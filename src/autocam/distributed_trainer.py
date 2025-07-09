"""Distributed trainer for Autocam with GPU allocation and mandate protocols."""

import asyncio
import multiprocessing as mp
from typing import Dict, List, Any, Optional, Tuple
import torch
import torch.distributed as dist
from dask.distributed import Client, get_client
import numpy as np

from .models import TrainingMandate, WorkingGroup
from .trainer import DynamicSelfSupervisedTrainer, ModelInterface
from .mlflow_tracker import AutocamMLflowTracker, MandateMetricsTracker


class GPUAllocator:
    """Manages GPU allocation for distributed training."""
    
    def __init__(self, gpu_ids: List[str] = None):
        """Initialize GPU allocator.
        
        Args:
            gpu_ids: List of GPU IDs to manage (e.g., ["gpu:0", "gpu:1"])
        """
        if gpu_ids is None:
            # Auto-detect available GPUs
            gpu_ids = [f"gpu:{i}" for i in range(torch.cuda.device_count())]
        
        self.gpu_ids = gpu_ids
        self.available_gpus = gpu_ids.copy()
        self.allocated_gpus = {}
    
    def allocate_gpu(self, entity_id: str, strategy: str = "round_robin") -> Optional[str]:
        """Allocate a GPU to an entity (working group, session, etc.).
        
        Args:
            entity_id: ID of the entity requesting GPU
            strategy: Allocation strategy ("round_robin", "least_used", "random")
            
        Returns:
            GPU ID if available, None otherwise
        """
        if not self.available_gpus:
            return None
        
        if strategy == "round_robin":
            gpu_id = self.available_gpus.pop(0)
        elif strategy == "random":
            gpu_id = np.random.choice(self.available_gpus)
            self.available_gpus.remove(gpu_id)
        elif strategy == "least_used":
            # Could implement more sophisticated logic here
            gpu_id = self.available_gpus.pop(0)
        else:
            gpu_id = self.available_gpus.pop(0)
        
        self.allocated_gpus[entity_id] = gpu_id
        return gpu_id
    
    def release_gpu(self, entity_id: str):
        """Release a GPU allocation."""
        if entity_id in self.allocated_gpus:
            gpu_id = self.allocated_gpus.pop(entity_id)
            self.available_gpus.append(gpu_id)
    
    def get_gpu_for_entity(self, entity_id: str) -> Optional[str]:
        """Get the GPU allocated to an entity."""
        return self.allocated_gpus.get(entity_id)


class DistributedMandateTrainer:
    """Distributed trainer that works with mandate protocols and GPU allocation."""
    
    def __init__(self, gpu_allocator: GPUAllocator = None, mlflow_tracker: AutocamMLflowTracker = None):
        """Initialize distributed trainer.
        
        Args:
            gpu_allocator: GPU allocator instance
            mlflow_tracker: MLflow tracker instance
        """
        self.gpu_allocator = gpu_allocator or GPUAllocator()
        self.mlflow_tracker = mlflow_tracker
        self.mandate_metrics_tracker = MandateMetricsTracker(mlflow_tracker) if mlflow_tracker else None
        self.training_history = {}
    
    def train_working_group_dask(
        self, 
        working_group: WorkingGroup,
        models: Dict[str, ModelInterface],
        dataloader,
        loss_fn,
        optimizer_factory,
        dask_client: Client = None,
        session_name: str = "unknown_session"
    ) -> Dict[str, Any]:
        """Train a working group using Dask distributed computing.
        
        Args:
            working_group: Working group to train
            models: Dictionary of model implementations
            dataloader: Data loader
            loss_fn: Loss function
            optimizer_factory: Function to create optimizers
            dask_client: Dask client (auto-detected if None)
            session_name: Name of the session for MLflow tracking
            
        Returns:
            Training results
        """
        if dask_client is None:
            dask_client = get_client()
        
        # Allocate GPU for this working group
        gpu_id = self.gpu_allocator.allocate_gpu(working_group.name)
        if gpu_id is None:
            raise RuntimeError(f"No GPU available for working group {working_group.name}")
        
        try:
            # Start MLflow tracking if available
            if self.mlflow_tracker:
                self.mlflow_tracker.start_working_group_run(working_group, session_name)
                
                # Log model parameters
                for model_name in working_group.participants:
                    if model_name in models:
                        # Create a dummy participant for logging (in practice, you'd have the actual participant)
                        from .models import Participant, ModelType, Dimension
                        participant = Participant(
                            name=model_name,
                            model_type=ModelType.CNN,
                            model_tag="encoder",
                            in_channels=3,
                            out_channels=128,
                            dimension=Dimension.D2
                        )
                        self.mlflow_tracker.log_model_parameters(model_name, models[model_name], participant)
            
            # Submit training task to Dask
            future = dask_client.submit(
                self._train_working_group_worker,
                working_group,
                models,
                dataloader,
                loss_fn,
                optimizer_factory,
                gpu_id,
                self.mlflow_tracker is not None,  # Enable MLflow tracking
                pure=False
            )
            
            # Wait for completion and get results
            results = future.result()
            return results
            
        finally:
            # Release GPU allocation
            self.gpu_allocator.release_gpu(working_group.name)
            
            # End MLflow tracking
            if self.mlflow_tracker:
                self.mlflow_tracker.end_working_group_run(working_group.name)
    
    def train_working_group_multiprocessing(
        self,
        working_group: WorkingGroup,
        models: Dict[str, ModelInterface],
        dataloader,
        loss_fn,
        optimizer_factory,
        num_processes: int = 1,
        session_name: str = "unknown_session"
    ) -> Dict[str, Any]:
        """Train a working group using multiprocessing.
        
        Args:
            working_group: Working group to train
            models: Dictionary of model implementations
            dataloader: Data loader
            loss_fn: Loss function
            optimizer_factory: Function to create optimizers
            num_processes: Number of processes to use
            session_name: Name of the session for MLflow tracking
            
        Returns:
            Training results
        """
        # Allocate GPU for this working group
        gpu_id = self.gpu_allocator.allocate_gpu(working_group.name)
        if gpu_id is None:
            raise RuntimeError(f"No GPU available for working group {working_group.name}")
        
        try:
            # Start MLflow tracking if available
            if self.mlflow_tracker:
                self.mlflow_tracker.start_working_group_run(working_group, session_name)
                
                # Log model parameters
                for model_name in working_group.participants:
                    if model_name in models:
                        from .models import Participant, ModelType, Dimension
                        participant = Participant(
                            name=model_name,
                            model_type=ModelType.CNN,
                            model_tag="encoder",
                            in_channels=3,
                            out_channels=128,
                            dimension=Dimension.D2
                        )
                        self.mlflow_tracker.log_model_parameters(model_name, models[model_name], participant)
            
            if num_processes == 1:
                # Single process training
                return self._train_working_group_worker(
                    working_group, models, dataloader, loss_fn, optimizer_factory, gpu_id, self.mlflow_tracker is not None
                )
            else:
                # Multi-process training
                with mp.Pool(num_processes) as pool:
                    # Split data across processes
                    data_chunks = self._split_dataloader(dataloader, num_processes)
                    
                    # Submit tasks to process pool
                    futures = []
                    for chunk in data_chunks:
                        future = pool.apply_async(
                            self._train_working_group_worker,
                            (working_group, models, chunk, loss_fn, optimizer_factory, gpu_id, self.mlflow_tracker is not None)
                        )
                        futures.append(future)
                    
                    # Collect results
                    results = {}
                    for future in futures:
                        chunk_results = future.get()
                        results.update(chunk_results)
                    
                    return results
                    
        finally:
            # Release GPU allocation
            self.gpu_allocator.release_gpu(working_group.name)
            
            # End MLflow tracking
            if self.mlflow_tracker:
                self.mlflow_tracker.end_working_group_run(working_group.name)
    
    def _train_working_group_worker(
        self,
        working_group: WorkingGroup,
        models: Dict[str, ModelInterface],
        dataloader,
        loss_fn,
        optimizer_factory,
        gpu_id: str,
        enable_mlflow: bool = False
    ) -> Dict[str, Any]:
        """Worker function for training a working group.
        
        This function runs on a worker (Dask or multiprocessing) and handles
        the actual training using the mandate protocols.
        """
        # Set GPU device
        if torch.cuda.is_available():
            torch.cuda.set_device(int(gpu_id.split(":")[1]))
        
        # Create trainer
        trainer = DynamicSelfSupervisedTrainer(models)
        
        # Create optimizers
        optimizers = {}
        for name in working_group.participants:
            if name in models:
                optimizers[name] = optimizer_factory(models[name])
        
        # Train using mandate protocols
        results = {}
        mandate_specific_results = []
        
        for epoch, batch in enumerate(dataloader):
            # Use mandate protocol to get reference targets
            targets = working_group.training_mandate.get_reference(
                batch, working_group, models
            )
            
            # Get optimizer assignments
            optimizer_assignments = working_group.training_mandate.get_optimizer_assignments(
                working_group, optimizers
            )
            
            # Train each student against its target
            batch_results = {}
            batch_losses = {}
            
            for student_name, target_tensor in targets.items():
                if student_name in models and student_name in optimizer_assignments:
                    student_model = models[student_name]
                    optimizer = optimizer_assignments[student_name]
                    
                    # Forward pass
                    student_output = student_model.forward(batch)
                    
                    # Compute loss
                    loss = loss_fn(student_output, target_tensor)
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    loss_value = loss.item()
                    batch_results[f"{student_name}_loss"] = loss_value
                    batch_results[f"{student_name}_output_shape"] = student_output.shape
                    batch_losses[student_name] = loss_value
            
            results.update(batch_results)
            
            # Track mandate-specific metrics
            if enable_mlflow and self.mandate_metrics_tracker:
                mandate_result = {
                    "epoch": epoch,
                    "losses": batch_losses,
                    "targets": list(targets.keys())
                }
                
                if working_group.training_mandate == TrainingMandate.ONE_VS_ONE:
                    pairs = working_group.training_mandate.get_student_target_pairs(working_group)
                    self.mandate_metrics_tracker.track_one_vs_one_metrics(
                        working_group.name, pairs, batch_losses, epoch
                    )
                elif working_group.training_mandate == TrainingMandate.BARYCENTRIC_TARGETS:
                    # Extract barycentric combinations from results
                    barycentric_combinations = []
                    for key, value in batch_results.items():
                        if "barycentric_models" in str(value):
                            barycentric_combinations.append({"chosen_models": value.get("barycentric_models", [])})
                    
                    self.mandate_metrics_tracker.track_barycentric_metrics(
                        working_group.name, barycentric_combinations, batch_losses, epoch
                    )
                elif working_group.training_mandate == TrainingMandate.ONE_VS_RANDOM_MEAN:
                    # Extract mean combinations from results
                    mean_combinations = []
                    for key, value in batch_results.items():
                        if "mean_models" in str(value):
                            mean_combinations.append({"mean_models": value.get("mean_models", [])})
                    
                    self.mandate_metrics_tracker.track_random_mean_metrics(
                        working_group.name, mean_combinations, batch_losses, epoch
                    )
            
            # Log training metrics if MLflow is enabled
            if enable_mlflow and self.mlflow_tracker:
                self.mlflow_tracker.log_training_metrics(
                    working_group.name, batch_results, epoch, 0
                )
                
                # Log GPU metrics
                memory_usage = get_gpu_memory_usage(gpu_id)
                self.mlflow_tracker.log_gpu_metrics(
                    working_group.name, gpu_id, memory_usage
                )
        
        return {
            "working_group": working_group.name,
            "results": results,
            "participants": working_group.participants,
            "gpu_used": gpu_id,
            "mandate_specific_results": mandate_specific_results
        }
    
    def _split_dataloader(self, dataloader, num_chunks: int) -> List:
        """Split dataloader into chunks for multiprocessing."""
        # This is a simplified version - in practice you'd want more sophisticated splitting
        data_list = list(dataloader)
        chunk_size = len(data_list) // num_chunks
        return [data_list[i:i + chunk_size] for i in range(0, len(data_list), chunk_size)]
    
    def train_conference_distributed(
        self,
        conference_config,
        models: Dict[str, ModelInterface],
        dataloader,
        loss_fn,
        optimizer_factory,
        strategy: str = "dask",
        conference_name: str = "autocam_conference",
        **kwargs
    ) -> Dict[str, Any]:
        """Train entire conference using distributed computing.
        
        Args:
            conference_config: Conference configuration
            models: Dictionary of model implementations
            dataloader: Data loader
            loss_fn: Loss function
            optimizer_factory: Function to create optimizers
            strategy: "dask" or "multiprocessing"
            conference_name: Name of the conference for MLflow tracking
            **kwargs: Additional arguments for specific strategies
            
        Returns:
            Training results for all working groups
        """
        # Start conference MLflow tracking
        if self.mlflow_tracker:
            self.mlflow_tracker.start_conference_run(
                conference_name=conference_name,
                conference_config=conference_config.__dict__ if hasattr(conference_config, '__dict__') else conference_config,
                metadata={
                    "strategy": strategy,
                    "num_models": len(models),
                    "gpu_count": len(self.gpu_allocator.gpu_ids)
                }
            )
        
        all_results = {}
        
        try:
            for session in conference_config.parallel_sessions:
                session_results = {}
                
                for working_group in session.working_groups:
                    print(f"Training working group: {working_group.name}")
                    
                    if strategy == "dask":
                        result = self.train_working_group_dask(
                            working_group, models, dataloader, loss_fn, optimizer_factory,
                            kwargs.get("dask_client"), session.name
                        )
                    elif strategy == "multiprocessing":
                        result = self.train_working_group_multiprocessing(
                            working_group, models, dataloader, loss_fn, optimizer_factory,
                            kwargs.get("num_processes", 1), session.name
                        )
                    else:
                        raise ValueError(f"Unknown strategy: {strategy}")
                    
                    session_results[working_group.name] = result
                
                all_results[session.name] = session_results
            
            # Log conference comparison metrics
            if self.mlflow_tracker:
                comparison_metrics = self._compute_conference_comparison_metrics(all_results)
                self.mlflow_tracker.log_conference_comparison(all_results, comparison_metrics)
            
            return all_results
            
        finally:
            # End conference MLflow tracking
            if self.mlflow_tracker:
                self.mlflow_tracker.end_conference_run()
    
    def _compute_conference_comparison_metrics(self, all_results: Dict[str, Any]) -> Dict[str, float]:
        """Compute comparison metrics across all working groups."""
        all_losses = []
        all_models = set()
        
        for session_name, session_results in all_results.items():
            for wg_name, wg_results in session_results.items():
                results = wg_results.get("results", {})
                for key, value in results.items():
                    if "loss" in key and isinstance(value, (int, float)):
                        all_losses.append(value)
                        model_name = key.split("_")[0]
                        all_models.add(model_name)
        
        if not all_losses:
            return {}
        
        return {
            "avg_loss_across_all_models": np.mean(all_losses),
            "min_loss_across_all_models": np.min(all_losses),
            "max_loss_across_all_models": np.max(all_losses),
            "std_loss_across_all_models": np.std(all_losses),
            "total_models_tracked": len(all_models)
        }


# Utility functions for GPU management
def setup_gpu_environment(gpu_id: str):
    """Setup GPU environment for distributed training."""
    if torch.cuda.is_available():
        torch.cuda.set_device(int(gpu_id.split(":")[1]))
        torch.cuda.empty_cache()


def get_gpu_memory_usage(gpu_id: str) -> Dict[str, float]:
    """Get GPU memory usage information."""
    if torch.cuda.is_available():
        device = torch.device(gpu_id)
        allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
        cached = torch.cuda.memory_reserved(device) / 1024**3  # GB
        return {
            "allocated_gb": allocated,
            "cached_gb": cached,
            "free_gb": cached - allocated
        }
    return {"allocated_gb": 0, "cached_gb": 0, "free_gb": 0}


def optimize_gpu_allocation(working_groups: List[WorkingGroup], gpu_ids: List[str]) -> Dict[str, str]:
    """Optimize GPU allocation based on working group characteristics."""
    # Simple round-robin allocation
    # In practice, you'd want more sophisticated logic based on:
    # - Model sizes and memory requirements
    # - Training mandate complexity
    # - Expected training time
    # - GPU memory availability
    
    allocation = {}
    for i, wg in enumerate(working_groups):
        gpu_id = gpu_ids[i % len(gpu_ids)]
        allocation[wg.name] = gpu_id
    
    return allocation 