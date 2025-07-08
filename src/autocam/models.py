"""Pydantic models for the Autocam conference system."""

from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

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
