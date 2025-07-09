"""MLflow integration for Autocam conference tracking."""

import mlflow
import mlflow.pytorch
from typing import Dict, List, Any, Optional, Union
import torch
import numpy as np
from datetime import datetime
import json

from .models import TrainingMandate, WorkingGroup, Participant
from .trainer import ModelInterface


class AutocamMLflowTracker:
    """MLflow tracker for Autocam conferences with per-model logging."""
    
    def __init__(
        self,
        experiment_name: str = "autocam_conference",
        tracking_uri: Optional[str] = None,
        log_artifacts: bool = True,
        log_models: bool = True
    ):
        """Initialize MLflow tracker.
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking URI (defaults to local)
            log_artifacts: Whether to log model artifacts
            log_models: Whether to log PyTorch models
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.log_artifacts = log_artifacts
        self.log_models = log_models
        
        # Set tracking URI if provided
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        # Set experiment
        mlflow.set_experiment(experiment_name)
        
        # Active run tracking
        self.active_runs = {}
        self.conference_run_id = None
    
    def start_conference_run(
        self,
        conference_name: str,
        conference_config: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Start a new MLflow run for the entire conference.
        
        Args:
            conference_name: Name of the conference
            conference_config: Conference configuration
            metadata: Additional metadata
            
        Returns:
            Run ID for the conference
        """
        run_name = f"conference_{conference_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with mlflow.start_run(run_name=run_name) as run:
            self.conference_run_id = run.info.run_id
            
            # Log conference parameters
            mlflow.log_params({
                "conference_name": conference_name,
                "num_participants": len(conference_config.get("participants", [])),
                "num_sessions": len(conference_config.get("parallel_sessions", [])),
                "total_working_groups": sum(
                    len(session.get("working_groups", [])) 
                    for session in conference_config.get("parallel_sessions", [])
                )
            })
            
            # Log conference configuration
            if self.log_artifacts:
                with open("conference_config.json", "w") as f:
                    json.dump(conference_config, f, indent=2)
                mlflow.log_artifact("conference_config.json")
            
            # Log metadata
            if metadata:
                mlflow.log_params(metadata)
            
            # Log conference structure
            self._log_conference_structure(conference_config)
        
        return self.conference_run_id
    
    def start_working_group_run(
        self,
        working_group: WorkingGroup,
        session_name: str,
        parent_run_id: Optional[str] = None
    ) -> str:
        """Start a new MLflow run for a working group.
        
        Args:
            working_group: Working group to track
            session_name: Name of the session
            parent_run_id: Parent run ID (conference run)
            
        Returns:
            Run ID for the working group
        """
        run_name = f"wg_{working_group.name}_{session_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Set parent run if provided
        if parent_run_id:
            mlflow.set_tag("mlflow.parentRunId", parent_run_id)
        
        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id
            self.active_runs[working_group.name] = run_id
            
            # Log working group parameters
            mlflow.log_params({
                "working_group_name": working_group.name,
                "session_name": session_name,
                "training_mandate": working_group.training_mandate.value,
                "training_epochs": working_group.training_epochs,
                "num_participants": len(working_group.participants),
                "student_participants": working_group.student_participants or [],
                "target_participants": working_group.target_participants or [],
                "fixed_target": working_group.fixed_target,
                "random_mean_count": working_group.random_mean_count,
                "barycentric_min_models": working_group.barycentric_min_models,
                "barycentric_max_models": working_group.barycentric_max_models
            })
            
            # Log mandate-specific parameters
            self._log_mandate_parameters(working_group)
            
            # Log working group configuration
            if self.log_artifacts:
                wg_config = {
                    "name": working_group.name,
                    "description": working_group.description,
                    "participants": working_group.participants,
                    "training_mandate": working_group.training_mandate.value,
                    "training_epochs": working_group.training_epochs,
                    "student_participants": working_group.student_participants,
                    "target_participants": working_group.target_participants,
                    "fixed_target": working_group.fixed_target,
                    "random_mean_count": working_group.random_mean_count,
                    "barycentric_min_models": working_group.barycentric_min_models,
                    "barycentric_max_models": working_group.barycentric_max_models
                }
                
                with open(f"wg_{working_group.name}_config.json", "w") as f:
                    json.dump(wg_config, f, indent=2)
                mlflow.log_artifact(f"wg_{working_group.name}_config.json")
        
        return run_id
    
    def log_model_parameters(
        self,
        model_name: str,
        model: ModelInterface,
        participant: Participant
    ):
        """Log model parameters and configuration.
        
        Args:
            model_name: Name of the model
            model: Model instance
            participant: Participant configuration
        """
        if model_name not in self.active_runs:
            return
        
        with mlflow.start_run(run_id=self.active_runs[model_name], nested=True):
            # Log model architecture parameters
            mlflow.log_params({
                f"model_{model_name}_type": participant.model_type.value,
                f"model_{model_name}_tag": participant.model_tag,
                f"model_{model_name}_in_channels": participant.in_channels,
                f"model_{model_name}_out_channels": participant.out_channels,
                f"model_{model_name}_dimension": participant.dimension.value,
                f"model_{model_name}_input_shape": str(model.get_input_shape()),
                f"model_{model_name}_output_shape": str(model.get_output_shape())
            })
            
            # Log model configuration
            if participant.config:
                for key, value in participant.config.items():
                    mlflow.log_param(f"model_{model_name}_config_{key}", value)
            
            # Log model to MLflow if enabled
            if self.log_models and hasattr(model, 'state_dict'):
                mlflow.pytorch.log_model(model, f"model_{model_name}")
    
    def log_training_metrics(
        self,
        working_group_name: str,
        batch_results: Dict[str, Any],
        epoch: int,
        batch: int
    ):
        """Log training metrics for a working group.
        
        Args:
            working_group_name: Name of the working group
            batch_results: Results from training batch
            epoch: Current epoch
            batch: Current batch
        """
        if working_group_name not in self.active_runs:
            return
        
        with mlflow.start_run(run_id=self.active_runs[working_group_name], nested=True):
            # Log batch-level metrics
            for metric_name, metric_value in batch_results.items():
                if isinstance(metric_value, (int, float)):
                    mlflow.log_metric(f"batch_{metric_name}", metric_value, step=epoch)
                elif isinstance(metric_value, dict):
                    for sub_metric, sub_value in metric_value.items():
                        if isinstance(sub_value, (int, float)):
                            mlflow.log_metric(f"batch_{metric_name}_{sub_metric}", sub_value, step=epoch)
            
            # Log epoch and batch info
            mlflow.log_metric("epoch", epoch, step=epoch)
            mlflow.log_metric("batch", batch, step=epoch)
    
    def log_mandate_specific_metrics(
        self,
        working_group_name: str,
        mandate_results: Dict[str, Any],
        epoch: int
    ):
        """Log mandate-specific metrics.
        
        Args:
            working_group_name: Name of the working group
            mandate_results: Results specific to the training mandate
            epoch: Current epoch
        """
        if working_group_name not in self.active_runs:
            return
        
        with mlflow.start_run(run_id=self.active_runs[working_group_name], nested=True):
            # Log mandate-specific metrics
            for metric_name, metric_value in mandate_results.items():
                if isinstance(metric_value, (int, float)):
                    mlflow.log_metric(f"mandate_{metric_name}", metric_value, step=epoch)
                elif isinstance(metric_value, dict):
                    for sub_metric, sub_value in metric_value.items():
                        if isinstance(sub_value, (int, float)):
                            mlflow.log_metric(f"mandate_{metric_name}_{sub_metric}", sub_value, step=epoch)
    
    def log_gpu_metrics(
        self,
        working_group_name: str,
        gpu_id: str,
        memory_usage: Dict[str, float]
    ):
        """Log GPU metrics for a working group.
        
        Args:
            working_group_name: Name of the working group
            gpu_id: GPU ID being used
            memory_usage: GPU memory usage information
        """
        if working_group_name not in self.active_runs:
            return
        
        with mlflow.start_run(run_id=self.active_runs[working_group_name], nested=True):
            mlflow.log_params({
                "gpu_id": gpu_id,
                "gpu_allocated_gb": memory_usage.get("allocated_gb", 0),
                "gpu_cached_gb": memory_usage.get("cached_gb", 0),
                "gpu_free_gb": memory_usage.get("free_gb", 0)
            })
    
    def log_conference_comparison(
        self,
        conference_results: Dict[str, Any],
        comparison_metrics: Dict[str, float]
    ):
        """Log conference-wide comparison metrics.
        
        Args:
            conference_results: Results from all working groups
            comparison_metrics: Comparison metrics between models
        """
        if not self.conference_run_id:
            return
        
        with mlflow.start_run(run_id=self.conference_run_id, nested=True):
            # Log comparison metrics
            for metric_name, metric_value in comparison_metrics.items():
                if isinstance(metric_value, (int, float)):
                    mlflow.log_metric(f"comparison_{metric_name}", metric_value)
            
            # Log conference results summary
            if self.log_artifacts:
                with open("conference_results.json", "w") as f:
                    json.dump(conference_results, f, indent=2)
                mlflow.log_artifact("conference_results.json")
    
    def end_working_group_run(self, working_group_name: str):
        """End MLflow run for a working group.
        
        Args:
            working_group_name: Name of the working group
        """
        if working_group_name in self.active_runs:
            mlflow.end_run()
            del self.active_runs[working_group_name]
    
    def end_conference_run(self):
        """End MLflow run for the conference."""
        if self.conference_run_id:
            mlflow.end_run()
            self.conference_run_id = None
    
    def _log_conference_structure(self, conference_config: Dict[str, Any]):
        """Log the structure of the conference."""
        sessions = conference_config.get("parallel_sessions", [])
        
        for i, session in enumerate(sessions):
            session_name = session.get("name", f"session_{i}")
            working_groups = session.get("working_groups", [])
            
            mlflow.log_param(f"session_{i}_name", session_name)
            mlflow.log_param(f"session_{i}_num_working_groups", len(working_groups))
            
            for j, wg in enumerate(working_groups):
                wg_name = wg.get("name", f"wg_{j}")
                mandate = wg.get("training_mandate", "unknown")
                mlflow.log_param(f"session_{i}_wg_{j}_name", wg_name)
                mlflow.log_param(f"session_{i}_wg_{j}_mandate", mandate)
    
    def _log_mandate_parameters(self, working_group: WorkingGroup):
        """Log parameters specific to the training mandate."""
        mandate = working_group.training_mandate
        
        if mandate == TrainingMandate.ONE_VS_ONE:
            mlflow.log_params({
                "mandate_type": "one_vs_one",
                "num_student_target_pairs": len(working_group.student_participants or [])
            })
        
        elif mandate == TrainingMandate.ONE_VS_RANDOM_MEAN:
            mlflow.log_params({
                "mandate_type": "one_vs_random_mean",
                "random_mean_count": working_group.random_mean_count
            })
        
        elif mandate == TrainingMandate.ONE_VS_FIXED:
            mlflow.log_params({
                "mandate_type": "one_vs_fixed",
                "fixed_target": working_group.fixed_target
            })
        
        elif mandate == TrainingMandate.RANDOM_PAIRS:
            mlflow.log_params({
                "mandate_type": "random_pairs",
                "num_participants": len(working_group.participants)
            })
        
        elif mandate == TrainingMandate.BARYCENTRIC_TARGETS:
            mlflow.log_params({
                "mandate_type": "barycentric_targets",
                "barycentric_min_models": working_group.barycentric_min_models,
                "barycentric_max_models": working_group.barycentric_max_models
            })


class MandateMetricsTracker:
    """Specialized tracker for mandate-specific metrics."""
    
    def __init__(self, mlflow_tracker: AutocamMLflowTracker):
        """Initialize mandate metrics tracker.
        
        Args:
            mlflow_tracker: MLflow tracker instance
        """
        self.mlflow_tracker = mlflow_tracker
        self.mandate_metrics = {}
    
    def track_one_vs_one_metrics(
        self,
        working_group_name: str,
        student_target_pairs: List[tuple],
        losses: Dict[str, float],
        epoch: int
    ):
        """Track metrics for one vs one training."""
        metrics = {
            "num_pairs": len(student_target_pairs),
            "avg_loss": np.mean(list(losses.values())),
            "min_loss": np.min(list(losses.values())),
            "max_loss": np.max(list(losses.values())),
            "std_loss": np.std(list(losses.values()))
        }
        
        self.mlflow_tracker.log_mandate_specific_metrics(working_group_name, metrics, epoch)
    
    def track_barycentric_metrics(
        self,
        working_group_name: str,
        barycentric_combinations: List[Dict[str, Any]],
        losses: Dict[str, float],
        epoch: int
    ):
        """Track metrics for barycentric training."""
        num_combinations = len(barycentric_combinations)
        avg_models_per_combination = np.mean([
            len(combo.get("chosen_models", [])) for combo in barycentric_combinations
        ])
        
        metrics = {
            "num_combinations": num_combinations,
            "avg_models_per_combination": avg_models_per_combination,
            "avg_loss": np.mean(list(losses.values())),
            "min_loss": np.min(list(losses.values())),
            "max_loss": np.max(list(losses.values())),
            "std_loss": np.std(list(losses.values()))
        }
        
        self.mlflow_tracker.log_mandate_specific_metrics(working_group_name, metrics, epoch)
    
    def track_random_mean_metrics(
        self,
        working_group_name: str,
        mean_combinations: List[Dict[str, Any]],
        losses: Dict[str, float],
        epoch: int
    ):
        """Track metrics for random mean training."""
        avg_models_per_mean = np.mean([
            len(combo.get("mean_models", [])) for combo in mean_combinations
        ])
        
        metrics = {
            "num_combinations": len(mean_combinations),
            "avg_models_per_mean": avg_models_per_mean,
            "avg_loss": np.mean(list(losses.values())),
            "min_loss": np.min(list(losses.values())),
            "max_loss": np.max(list(losses.values())),
            "std_loss": np.std(list(losses.values()))
        }
        
        self.mlflow_tracker.log_mandate_specific_metrics(working_group_name, metrics, epoch) 