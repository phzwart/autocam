"""Generated model interface for representation learning."""

from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import Optional


class ModelInterface(ABC):
    """Abstract interface for models that can be injected."""

    @abstractmethod
    def forward(self, x):
        """Forward pass through the model."""
        pass

    @abstractmethod
    def get_input_shape(self) -> tuple:
        """Get the expected input shape."""
        pass

    @abstractmethod
    def get_output_shape(self) -> tuple:
        """Get the output shape."""
        pass


class ParticipantRegistry:
    """Registry for managing participant models."""

    def __init__(self):
        """Initialize the registry."""
        self._models: Dict[str, ModelInterface] = {}
        self._specs: Dict[str, Dict[str, Any]] = {}

    def register_model(self, name: str, model: ModelInterface, spec: Dict[str, Any]):
        """Register a model with its specification."""
        self._models[name] = model
        self._specs[name] = spec

    def get_model(self, name: str) -> Optional[ModelInterface]:
        """Get a model by name."""
        return self._models.get(name)

    def get_spec(self, name: str) -> Optional[Dict[str, Any]]:
        """Get model specification by name."""
        return self._specs.get(name)

    def list_models(self) -> list[str]:
        """List all registered model names."""
        return list(self._models.keys())

    def list_models_by_tag(self, tag: str) -> list[str]:
        """List models by tag."""
        return [
            name for name, spec in self._specs.items() if spec.get("model_tag") == tag
        ]
