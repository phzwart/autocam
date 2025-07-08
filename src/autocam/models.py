"""Pydantic models for conference configuration."""

from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from pydantic import BaseModel
from pydantic import Field


class ModelType(str, Enum):
    """Valid model types."""

    CNN = "cnn"
    VIT = "vit"
    CUSTOM = "custom"


class Dimension(str, Enum):
    """Valid dimensions."""

    D2 = "2D"
    D3 = "3D"


class Participant(BaseModel):
    """A participant in the conference."""

    model_config = {"protected_namespaces": ()}

    name: str = Field(..., description="Name of the participant")
    model_type: ModelType = Field(..., description="Type of model")
    model_tag: str = Field(..., description="Tag for grouping models")
    in_channels: int = Field(..., description="Number of input channels")
    out_channels: int = Field(..., description="Number of output channels")
    dimension: Dimension = Field(..., description="Dimension of the model")
    config: Optional[Dict[str, Any]] = Field(
        default=None, description="Model configuration"
    )


class WorkingGroup(BaseModel):
    """A working group within a session."""

    name: str = Field(..., description="Name of the working group")
    description: Optional[str] = Field(
        default=None, description="Description of the working group"
    )
    participants: List[str] = Field(..., description="List of participant names")


class ParallelSession(BaseModel):
    """A parallel session in the conference."""

    name: str = Field(..., description="Name of the session")
    description: Optional[str] = Field(
        default=None, description="Description of the session"
    )
    working_groups: List[WorkingGroup] = Field(
        ..., description="Working groups in this session"
    )


class Conference(BaseModel):
    """A conference configuration."""

    name: str = Field(..., description="Name of the conference")
    description: Optional[str] = Field(
        default=None, description="Description of the conference"
    )
    participants: List[Participant] = Field(..., description="List of participants")
    parallel_sessions: List[ParallelSession] = Field(
        ..., description="Parallel sessions"
    )


class ConferenceConfig(BaseModel):
    """Complete conference configuration."""

    conference: Conference = Field(..., description="Conference configuration")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Metadata")

    def validate_config(self) -> None:
        """Validate the configuration."""
        # Check that all participants referenced in working groups exist
        participant_names = {p.name for p in self.conference.participants}

        for session in self.conference.parallel_sessions:
            for wg in session.working_groups:
                for participant_name in wg.participants:
                    if participant_name not in participant_names:
                        raise ValueError(
                            f"Participant {participant_name!r} not found in "
                            f"participants list"
                        )

    def get_participant_by_name(self, name: str) -> Optional[Participant]:
        """Get a participant by name."""
        for participant in self.conference.participants:
            if participant.name == name:
                return participant
        return None

    def get_working_group_by_name(
        self, session_name: str, group_name: str
    ) -> Optional[WorkingGroup]:
        """Get a working group by session and group name."""
        for session in self.conference.parallel_sessions:
            if session.name == session_name:
                for wg in session.working_groups:
                    if wg.name == group_name:
                        return wg
        return None
