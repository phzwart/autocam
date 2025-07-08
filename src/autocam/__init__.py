"""Autocam - Representation Learning Conference System."""

from .config import load_conference_config
from .config import save_conference_config
from .config import validate_yaml_schema
from .generator import CodeGenerator
from .generator import generate_from_yaml
from .models import Conference
from .models import ConferenceConfig
from .models import Dimension
from .models import ModelType
from .models import ParallelSession
from .models import Participant
from .models import WorkingGroup


__all__ = [
    "__version__",
    "ConferenceConfig",
    "Conference",
    "Participant",
    "ParallelSession",
    "WorkingGroup",
    "ModelType",
    "Dimension",
    "load_conference_config",
    "save_conference_config",
    "validate_yaml_schema",
    "CodeGenerator",
    "generate_from_yaml",
]

try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"
