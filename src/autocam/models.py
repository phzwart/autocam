"""Pydantic models for the Autocam conference system."""

from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from abc import ABC, abstractmethod
import random
import torch
import numpy as np

from pydantic import BaseModel
from pydantic import Field


class ModelType(str, Enum):
    """Supported model types."""

    CNN = "cnn"
    VIT = "vit"
    CUSTOM = "custom"


class Dimension(str, Enum):
    """Supported dimensions."""

    D2 = "2D"
    D3 = "3D"


class MandateProtocol(ABC):
    """Abstract base class for training mandate protocols."""
    
    @abstractmethod
    def get_reference(self, batch, working_group, models) -> Dict[str, torch.Tensor]:
        """Generate target vectors according to mandate protocol."""
        pass
    
    @abstractmethod
    def get_student_target_pairs(self, working_group) -> List[Tuple[str, str]]:
        """Return student-target pairs for this mandate."""
        pass
    
    @abstractmethod
    def get_optimizer_assignments(self, working_group, optimizers) -> Dict[str, Any]:
        """Assign optimizers to students."""
        pass


class OneVsOneProtocol(MandateProtocol):
    """Protocol for one vs one training mandate."""
    
    def get_reference(self, batch, working_group, models) -> Dict[str, torch.Tensor]:
        """Generate targets using fixed student-target pairs."""
        targets = {}
        student_participants = working_group.student_participants or []
        target_participants = working_group.target_participants or []
        
        for i, student_name in enumerate(student_participants):
            if i < len(target_participants):
                target_name = target_participants[i]
            else:
                # If no corresponding target, pick random one
                target_name = random.choice(target_participants)
            
            if student_name in models and target_name in models:
                with torch.no_grad():
                    targets[student_name] = models[target_name].forward(batch)
        
        return targets
    
    def get_student_target_pairs(self, working_group) -> List[Tuple[str, str]]:
        """Return fixed student-target pairs."""
        student_participants = working_group.student_participants or []
        target_participants = working_group.target_participants or []
        
        pairs = []
        for i, student in enumerate(student_participants):
            if i < len(target_participants):
                pairs.append((student, target_participants[i]))
            else:
                # If no corresponding target, pair with first target
                if target_participants:
                    pairs.append((student, target_participants[0]))
        
        return pairs
    
    def get_optimizer_assignments(self, working_group, optimizers) -> Dict[str, Any]:
        """Assign optimizers to students."""
        student_participants = working_group.student_participants or []
        return {student: optimizers.get(student) for student in student_participants}


class OneVsRandomMeanProtocol(MandateProtocol):
    """Protocol for one vs random mean training mandate."""
    
    def get_reference(self, batch, working_group, models) -> Dict[str, torch.Tensor]:
        """Generate targets using mean of random model outputs."""
        targets = {}
        student_participants = working_group.student_participants or []
        available_models = list(models.keys())
        random_mean_count = working_group.random_mean_count or 2
        
        for student_name in student_participants:
            if student_name in models and len(available_models) >= random_mean_count:
                # Choose random models for mean
                k = random.randint(2, min(random_mean_count, len(available_models)))
                chosen_models = random.sample(available_models, k)
                
                # Compute mean of random model outputs
                mean_output = None
                for model_name in chosen_models:
                    if model_name != student_name:
                        model_output = models[model_name].forward(batch)
                        if mean_output is None:
                            mean_output = model_output
                        else:
                            mean_output = (mean_output + model_output) / 2
                
                if mean_output is not None:
                    targets[student_name] = mean_output
        
        return targets
    
    def get_student_target_pairs(self, working_group) -> List[Tuple[str, str]]:
        """Return student-target pairs (dynamic, generated per batch)."""
        # For random mean, pairs are generated dynamically in get_reference
        return []
    
    def get_optimizer_assignments(self, working_group, optimizers) -> Dict[str, Any]:
        """Assign optimizers to students."""
        student_participants = working_group.student_participants or []
        return {student: optimizers.get(student) for student in student_participants}


class OneVsFixedProtocol(MandateProtocol):
    """Protocol for one vs fixed training mandate."""
    
    def get_reference(self, batch, working_group, models) -> Dict[str, torch.Tensor]:
        """Generate targets using fixed target model."""
        targets = {}
        student_participants = working_group.student_participants or []
        fixed_target = working_group.fixed_target
        
        if fixed_target not in models:
            return targets
        
        target_model = models[fixed_target]
        with torch.no_grad():
            fixed_output = target_model.forward(batch)
        
        for student_name in student_participants:
            if student_name in models and student_name != fixed_target:
                targets[student_name] = fixed_output
        
        return targets
    
    def get_student_target_pairs(self, working_group) -> List[Tuple[str, str]]:
        """Return student-target pairs (all students vs fixed target)."""
        student_participants = working_group.student_participants or []
        fixed_target = working_group.fixed_target
        
        if not fixed_target:
            return []
        
        return [(student, fixed_target) for student in student_participants]
    
    def get_optimizer_assignments(self, working_group, optimizers) -> Dict[str, Any]:
        """Assign optimizers to students."""
        student_participants = working_group.student_participants or []
        return {student: optimizers.get(student) for student in student_participants}


class RandomPairsProtocol(MandateProtocol):
    """Protocol for random pairs training mandate."""
    
    def get_reference(self, batch, working_group, models) -> Dict[str, torch.Tensor]:
        """Generate targets using random student-target pairs."""
        targets = {}
        participants = working_group.participants
        available_participants = [p for p in participants if p in models]
        
        if len(available_participants) < 2:
            return targets
        
        # Randomly pair participants
        random.shuffle(available_participants)
        pairs = []
        for i in range(0, len(available_participants) - 1, 2):
            pairs.append((available_participants[i], available_participants[i + 1]))
        
        for student_name, target_name in pairs:
            if student_name in models and target_name in models:
                with torch.no_grad():
                    targets[student_name] = models[target_name].forward(batch)
        
        return targets
    
    def get_student_target_pairs(self, working_group) -> List[Tuple[str, str]]:
        """Return random student-target pairs (generated per batch)."""
        # For random pairs, pairs are generated dynamically in get_reference
        return []
    
    def get_optimizer_assignments(self, working_group, optimizers) -> Dict[str, Any]:
        """Assign optimizers to students."""
        participants = working_group.participants
        return {participant: optimizers.get(participant) for participant in participants}


class BarycentricTargetsProtocol(MandateProtocol):
    """Protocol for barycentric targets training mandate."""
    
    def get_reference(self, batch, working_group, models) -> Dict[str, torch.Tensor]:
        """Generate targets using barycentric combinations."""
        targets = {}
        student_participants = working_group.student_participants or []
        available_models = list(models.keys())
        min_models = working_group.barycentric_min_models or 2
        max_models = working_group.barycentric_max_models or 3
        
        for student_name in student_participants:
            if student_name in models and len(available_models) >= min_models:
                # Choose random number of models for barycentric combination
                k = random.randint(min_models, min(max_models, len(available_models)))
                chosen_models = random.sample(available_models, k)
                
                # Compute barycentric combination
                barycentric_output = None
                weights = np.random.dirichlet(np.ones(k))
                
                for i, model_name in enumerate(chosen_models):
                    if model_name != student_name:
                        model_output = models[model_name].forward(batch)
                        if barycentric_output is None:
                            barycentric_output = weights[i] * model_output
                        else:
                            barycentric_output += weights[i] * model_output
                
                if barycentric_output is not None:
                    targets[student_name] = barycentric_output
        
        return targets
    
    def get_student_target_pairs(self, working_group) -> List[Tuple[str, str]]:
        """Return student-target pairs (dynamic, generated per batch)."""
        # For barycentric targets, pairs are generated dynamically in get_reference
        return []
    
    def get_optimizer_assignments(self, working_group, optimizers) -> Dict[str, Any]:
        """Assign optimizers to students."""
        student_participants = working_group.student_participants or []
        return {student: optimizers.get(student) for student in student_participants}


class TrainingMandate(str, Enum):
    """Training mandate types for working groups."""

    ONE_VS_ONE = "one_vs_one"  # One model trains against another specific model
    ONE_VS_RANDOM_MEAN = (
        "one_vs_random_mean"  # One model trains against mean of 2+ random models
    )
    ONE_VS_FIXED = "one_vs_fixed"  # One model always trains against a fixed target
    RANDOM_PAIRS = "random_pairs"  # Random student-target pairs each batch
    BARYCENTRIC_TARGETS = (
        "barycentric_targets"  # Random barycentric combinations as targets
    )
    
    @property
    def protocol(self) -> MandateProtocol:
        """Get the protocol instance for this mandate."""
        protocols = {
            TrainingMandate.ONE_VS_ONE: OneVsOneProtocol(),
            TrainingMandate.ONE_VS_RANDOM_MEAN: OneVsRandomMeanProtocol(),
            TrainingMandate.ONE_VS_FIXED: OneVsFixedProtocol(),
            TrainingMandate.RANDOM_PAIRS: RandomPairsProtocol(),
            TrainingMandate.BARYCENTRIC_TARGETS: BarycentricTargetsProtocol(),
        }
        return protocols[self]
    
    def get_reference(self, batch, working_group, models) -> Dict[str, torch.Tensor]:
        """Generate target vectors according to mandate protocol."""
        return self.protocol.get_reference(batch, working_group, models)
    
    def get_student_target_pairs(self, working_group) -> List[Tuple[str, str]]:
        """Return student-target pairs for this mandate."""
        return self.protocol.get_student_target_pairs(working_group)
    
    def get_optimizer_assignments(self, working_group, optimizers) -> Dict[str, Any]:
        """Assign optimizers to students."""
        return self.protocol.get_optimizer_assignments(working_group, optimizers)


class Participant(BaseModel):
    """A participant in the conference (a model)."""

    model_config = {"protected_namespaces": ()}

    name: str = Field(..., description="Unique name for the participant")
    model_type: ModelType = Field(..., description="Type of model")
    model_tag: str = Field(..., description="Tag for grouping similar models")
    in_channels: int = Field(..., description="Number of input channels")
    out_channels: int = Field(..., description="Number of output channels")
    dimension: Dimension = Field(..., description="Data dimension")
    config: Dict[str, Any] = Field(
        default_factory=dict, description="Model configuration"
    )
    training_epochs: Optional[int] = Field(
        default=None, description="Training epochs for this participant"
    )


class WorkingGroup(BaseModel):
    """A working group within a parallel session.

    Attributes:
        name: Name of the working group.
        description: Description of the working group.
        participants: List of participant names.
        training_mandate: Training mandate for this working group.
        training_epochs: Training epochs for this working group.
        student_participants: Participants acting as students in this session.
        target_participants: Participants acting as targets in this session.
        fixed_target: Fixed target model for ONE_VS_FIXED mandate.
        random_mean_count: Number of models to use in random mean.
        barycentric_min_models: Minimum number of models for barycentric combinations.
        barycentric_max_models: Maximum number of models for barycentric combinations.
    """

    model_config = {"protected_namespaces": ()}

    name: str = Field(..., description="Name of the working group")
    description: str = Field(..., description="Description of the working group")
    participants: List[str] = Field(..., description="List of participant names")
    training_mandate: TrainingMandate = Field(
        ...,
        description="Training mandate for this working group"
    )
    training_epochs: Optional[int] = Field(
        default=100, description="Training epochs for this working group"
    )

    # Configuration for different training mandates
    student_participants: Optional[List[str]] = Field(
        default=None, description="Participants acting as students in this session"
    )
    target_participants: Optional[List[str]] = Field(
        default=None,
        description="Participants acting as targets in this session"
    )
    fixed_target: Optional[str] = Field(
        default=None, description="Fixed target model for ONE_VS_FIXED mandate"
    )
    random_mean_count: Optional[int] = Field(
        default=2,
        description="Number of models to use in random mean for "
        "ONE_VS_RANDOM_MEAN",
    )
    barycentric_min_models: Optional[int] = Field(
        default=2,
        description="Minimum number of models for barycentric combinations",
    )
    barycentric_max_models: Optional[int] = Field(
        default=3,
        description="Maximum number of models for barycentric combinations",
    )


class ParallelSession(BaseModel):
    """A parallel session in the conference. Can contain multiple working groups."""

    model_config = {"protected_namespaces": ()}

    name: str = Field(..., description="Name of the parallel session")
    description: str = Field(..., description="Description of the parallel session")
    working_groups: Optional[List[WorkingGroup]] = Field(
        default=None, description="Working groups in this session (leaf)"
    )
    subsessions: Optional[List["ParallelSession"]] = Field(
        default=None, description="Nested subsessions (ballrooms)"
    )
    training_epochs: Optional[int] = Field(
        default=100, description="Training epochs for this session"
    )

    @classmethod
    def __get_validators__(cls):
        """Get validators for the ParallelSession model."""
        yield cls.validate_either_groups_or_subsessions

    @classmethod
    def validate_either_groups_or_subsessions(cls, values):
        """Validate that a session has either working groups 
        or subsessions, but not both."""
        if values.get("working_groups") and values.get("subsessions"):
            raise ValueError(
                "A ParallelSession can have either working_groups or "
                "subsessions, not both."
            )
        if not values.get("working_groups") and not values.get("subsessions"):
            raise ValueError(
                "A ParallelSession must have either working_groups or "
                "subsessions."
            )
        return values


ParallelSession.update_forward_refs()


class ConferenceConfig(BaseModel):
    """Configuration for a conference."""

    model_config = {"protected_namespaces": ()}

    name: str = Field(..., description="Name of the conference")
    description: str = Field(..., description="Description of the conference")
    participants: List[Participant] = Field(..., description="List of participants")
    parallel_sessions: List[ParallelSession] = Field(
        ...,
        description="Parallel sessions"
    )


class Conference(BaseModel):
    """A complete conference configuration."""

    model_config = {"protected_namespaces": ()}

    conference: ConferenceConfig = Field(..., description="Conference configuration")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata")
