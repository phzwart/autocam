"""Dask + PyTorch orchestrator for Autocam conference execution."""

from typing import Any
from typing import Dict
from typing import List

from dask.distributed import Client
from dask.distributed import LocalCluster

from .conference import Conference
from .trainer import DynamicSelfSupervisedTrainer


class AutocamDaskOrchestrator:
    """Orchestrates conference execution using Dask for resource management and PyTorch for training.

    This class manages the execution of a conference using Dask for resource
    management and PyTorch for training.
    """

    def __init__(
        self,
        n_workers: int = 4,
        gpus_per_worker: int = 1,
        gpu_memory_limit: str = "8GB",
    ):
        """Initialize the orchestrator.

        Args:
            n_workers: Number of Dask workers
            gpus_per_worker: GPUs per worker
            gpu_memory_limit: Memory limit per GPU
        """
        self.n_workers = n_workers
        self.gpus_per_worker = gpus_per_worker
        self.gpu_memory_limit = gpu_memory_limit

        # Initialize Dask cluster
        self.cluster = LocalCluster(
            n_workers=n_workers, resources={"GPU": gpus_per_worker * n_workers}
        )
        self.client = Client(self.cluster)

    def run_conference(
        self,
        conference: Conference,
        model_implementations: Dict[str, Any],
        dataloader: Any,
        loss_fn: Any,
        optimizer_factory: Any,
    ) -> Dict[str, Any]:
        """Run a conference using Dask for orchestration.

        Args:
            conference: Conference configuration
            model_implementations: Dict mapping model names to PyTorch models
            dataloader: Data loader for training
            loss_fn: Loss function
            optimizer_factory: Function to create optimizers

        Returns:
            Dictionary with results from all working groups
        """
        futures = []

        # Submit working groups as Dask tasks
        for session in conference.config.parallel_sessions:
            session_futures = self._submit_session_tasks(
                session, model_implementations, dataloader, loss_fn, optimizer_factory
            )
            futures.extend(session_futures)

        # Wait for all to complete
        results = self.client.gather(futures)
        return results

    def _submit_session_tasks(
        self, session, model_implementations, dataloader, loss_fn, optimizer_factory
    ):
        """Submit tasks for a session (handles nested subsessions)."""
        futures = []

        if session.working_groups:
            # Direct working groups
            for wg in session.working_groups:
                future = self.client.submit(
                    self._run_working_group,
                    wg,
                    model_implementations,
                    dataloader,
                    loss_fn,
                    optimizer_factory,
                    resources={"GPU": self.gpus_per_worker},
                )
                futures.append(future)

        elif session.subsessions:
            # Nested subsessions
            for subsession in session.subsessions:
                subsession_futures = self._submit_session_tasks(
                    subsession,
                    model_implementations,
                    dataloader,
                    loss_fn,
                    optimizer_factory,
                )
                futures.extend(subsession_futures)

        return futures

    def _run_working_group(
        self, wg, model_implementations, dataloader, loss_fn, optimizer_factory
    ):
        """Run a working group (executed on Dask worker with GPU allocation)."""
        # Get models for this working group
        models = {
            name: model_implementations[name]
            for name in wg.participants
            if name in model_implementations
        }

        # Create trainer
        trainer = DynamicSelfSupervisedTrainer(models)

        # Create optimizers
        optimizers = {}
        for name in wg.participants:
            if name in model_implementations:
                optimizers[name] = optimizer_factory(model_implementations[name])

        # Train based on mandate
        results = {}
        for batch in dataloader:
            if wg.training_mandate == "one_vs_one":
                batch_results = trainer.train_batch_one_vs_one(
                    batch,
                    wg.student_participants or [],
                    wg.target_participants or [],
                    loss_fn,
                    optimizers,
                )
            elif wg.training_mandate == "one_vs_random_mean":
                batch_results = trainer.train_batch_one_vs_random_mean(
                    batch,
                    wg.student_participants or [],
                    wg.random_mean_count or 2,
                    loss_fn,
                    optimizers,
                )
            elif wg.training_mandate == "one_vs_fixed":
                if wg.fixed_target:
                    batch_results = trainer.train_batch_one_vs_fixed(
                        batch,
                        wg.student_participants or [],
                        wg.fixed_target,
                        loss_fn,
                        optimizers,
                    )
            elif wg.training_mandate == "random_pairs":
                batch_results = trainer.train_batch_random_pairs(
                    batch, wg.participants, loss_fn, optimizers
                )
            elif wg.training_mandate == "barycentric_targets":
                batch_results = trainer.train_batch_barycentric_targets(
                    batch,
                    wg.student_participants or [],
                    wg.barycentric_min_models or 2,
                    wg.barycentric_max_models or 3,
                    loss_fn,
                    optimizers,
                )

            results.update(batch_results)

        return {
            "working_group": wg.name,
            "results": results,
            "participants": wg.participants,
        }

    def close(self):
        """Close the Dask cluster."""
        self.client.close()
        self.cluster.close()


class GPUResourceManager:
    """Manages GPU allocation for working groups."""

    def __init__(self, available_gpus: List[str], memory_per_gpu: str = "8GB"):
        """Initialize GPU resource manager.

        Args:
            available_gpus: List of GPU device names
            memory_per_gpu: Memory limit per GPU
        """
        self.available_gpus = available_gpus
        self.memory_per_gpu = memory_per_gpu
        self.allocations = {}

    def allocate_for_working_group(
        self, wg, strategy: str = "round_robin"
    ) -> List[str]:
        """Allocate GPUs for a working group.

        Args:
            wg: Working group
            strategy: Allocation strategy ("round_robin", "load_balanced")

        Returns:
            List of allocated GPU device names
        """
        if strategy == "round_robin":
            return self._allocate_round_robin(wg)
        elif strategy == "load_balanced":
            return self._allocate_load_balanced(wg)
        else:
            raise ValueError(f"Unknown allocation strategy: {strategy}")

    def _allocate_round_robin(self, wg) -> List[str]:
        """Simple round-robin allocation."""
        num_models = len(wg.participants)
        num_gpus = min(num_models, len(self.available_gpus))

        allocated = []
        for i in range(num_gpus):
            gpu_idx = i % len(self.available_gpus)
            allocated.append(self.available_gpus[gpu_idx])

        return allocated

    def _allocate_load_balanced(self, wg) -> List[str]:
        """Load-balanced allocation based on current usage."""
        # Simple implementation - in practice you'd track actual GPU usage
        num_models = len(wg.participants)
        num_gpus = min(num_models, len(self.available_gpus))

        # For now, just use first available GPUs
        return self.available_gpus[:num_gpus]
