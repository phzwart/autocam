"""Configuration loading and validation for the representation learning system."""

from enum import Enum

import yaml

from .models import ConferenceConfig


def _enum_to_str(obj):
    if isinstance(obj, dict):
        return {k: _enum_to_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_enum_to_str(v) for v in obj]
    elif isinstance(obj, Enum):
        return obj.value
    return obj


def load_conference_config(yaml_path: str) -> ConferenceConfig:
    """Load a conference configuration from a YAML file."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    return ConferenceConfig(**data)


def save_conference_config(config: ConferenceConfig, yaml_path: str) -> None:
    """Save a conference configuration to a YAML file."""
    data = config.model_dump()
    data = _enum_to_str(data)
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, indent=2)


def enforce_working_group_constraints(config: ConferenceConfig) -> None:
    """Enforce that all models in a working group have the same in_channels.

    out_channels, and model_tag.
    """
    participants_by_name = {p.name: p for p in config.conference.participants}
    for session in config.conference.parallel_sessions:
        for wg in session.working_groups:
            if not wg.participants:
                continue
            # Get the reference participant
            ref = participants_by_name[wg.participants[0]]
            ref_in = ref.in_channels
            ref_out = ref.out_channels
            ref_tag = ref.model_tag
            for pname in wg.participants[1:]:
                p = participants_by_name[pname]
                if p.in_channels != ref_in:
                    raise ValueError(
                        f"Working group {wg.name!r} in session {session.name!r}: "
                        f"All models must have the same in_channels. "
                        f"{p.name!r} has {p.in_channels}, expected {ref_in}."
                    )
                if p.out_channels != ref_out:
                    raise ValueError(
                        f"Working group {wg.name!r} in session {session.name!r}: "
                        f"All models must have the same out_channels. "
                        f"{p.name!r} has {p.out_channels}, expected {ref_out}."
                    )
                if p.model_tag != ref_tag:
                    raise ValueError(
                        f"Working group {wg.name!r} in session {session.name!r}: "
                        f"All models must have the same model_tag. "
                        f"{p.name!r} has {p.model_tag!r}, expected {ref_tag!r}."
                    )


def validate_yaml_schema(yaml_path: str) -> bool:
    """Validate that a YAML file conforms to the expected schema.

    And working group constraints.
    """
    try:
        config = load_conference_config(yaml_path)
        config.validate_config()
        enforce_working_group_constraints(config)
        return True
    except Exception as e:
        print(f"Validation error: {e}")
        return False
