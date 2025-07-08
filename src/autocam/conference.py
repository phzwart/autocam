"""Direct conference object creation from YAML configurations."""

from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from .config import enforce_working_group_constraints
from .config import load_conference_config


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

    def list_models(self) -> List[str]:
        """List all registered model names."""
        return list(self._models.keys())

    def list_models_by_tag(self, tag: str) -> List[str]:
        """List models by tag."""
        return [
            name for name, spec in self._specs.items() if spec.get("model_tag") == tag
        ]


class Conference:
    """A conference loaded directly from YAML configuration."""

    def __init__(self, yaml_path: str):
        """Initialize conference from YAML file."""
        self.config = load_conference_config(yaml_path)
        self.name = self.config.conference.name
        self.registry = ParticipantRegistry()

        # Validate the configuration
        self.config.validate_config()
        enforce_working_group_constraints(self.config)

    def register_participant(
        self, name: str, model: ModelInterface, spec: Dict[str, Any]
    ):
        """Register a participant model."""
        self.registry.register_model(name, model, spec)

    def get_participant(self, name: str) -> Optional[ModelInterface]:
        """Get a participant model by name."""
        return self.registry.get_model(name)

    def get_participant_spec(self, name: str) -> Optional[Dict[str, Any]]:
        """Get participant specification by name."""
        return self.registry.get_spec(name)

    def get_working_group(self, session_name: str, group_name: str):
        """Get a working group by session and group name."""
        for session in self.config.conference.parallel_sessions:
            if session.name == session_name:
                for wg in session.working_groups:
                    if wg.name == group_name:
                        return wg
        return None

    def get_session_participants(self, session_name: str) -> List[ModelInterface]:
        """Get all participants in a session."""
        for session in self.config.conference.parallel_sessions:
            if session.name == session_name:
                participants = []
                for wg in session.working_groups:
                    for participant_name in wg.participants:
                        model = self.registry.get_model(participant_name)
                        if model:
                            participants.append(model)
                return participants
        return []

    def run_session(self, session_name: str, data: Any):
        """Run all models in a session on the given data."""
        participants = self.get_session_participants(session_name)
        results = {}
        for i, model in enumerate(participants):
            try:
                results[f"model_{i}"] = model.forward(data)
            except Exception as e:
                results[f"model_{i}"] = f"Error: {e}"
        return results

    def get_participants_by_tag(self, tag: str) -> List[str]:
        """Get participant names by tag."""
        return self.registry.list_models_by_tag(tag)

    def validate_registration(self) -> bool:
        """Validate that all expected participants are registered."""
        expected_participants = set()
        for session in self.config.conference.parallel_sessions:
            for wg in session.working_groups:
                expected_participants.update(wg.participants)

        registered_participants = set(self.registry.list_models())
        missing = expected_participants - registered_participants

        if missing:
            print(f"Missing participants: {missing}")
            return False
        return True

    def list_sessions(self) -> List[str]:
        """List all session names."""
        return [session.name for session in self.config.conference.parallel_sessions]

    def list_participants(self) -> List[str]:
        """List all participant names."""
        return [p.name for p in self.config.conference.participants]

    def get_participant_config(self, name: str):
        """Get participant configuration by name."""
        for participant in self.config.conference.participants:
            if participant.name == name:
                return participant
        return None


def create_conference(yaml_path: str) -> Conference:
    """Create a conference object directly from YAML file."""
    return Conference(yaml_path)
