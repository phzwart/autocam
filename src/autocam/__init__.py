"""Autocam - Representation Learning Conference System."""

from .conference import Conference as ConferenceClass
from .conference import ModelInterface
from .conference import create_conference
from .config import load_conference_config
from .config import save_conference_config
from .config import validate_yaml_schema
from .models import Conference
from .models import ConferenceConfig
from .models import Dimension
from .models import ModelType
from .models import ParallelSession
from .models import Participant
from .models import WorkingGroup
from .orchestrator import AutocamDaskOrchestrator
from .orchestrator import GPUResourceManager


__all__ = [
    "__version__",
    "ConferenceConfig",
    "Conference",
    "ConferenceClass",
    "create_conference",
    "ModelInterface",
    "Participant",
    "ParallelSession",
    "WorkingGroup",
    "ModelType",
    "Dimension",
    "load_conference_config",
    "save_conference_config",
    "validate_yaml_schema",
    "AutocamDaskOrchestrator",
    "GPUResourceManager",
]

try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"
