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

    # Handle nested conference structure
    if "conference" in data:
        conference_data = data["conference"]
    else:
        conference_data = data

    return ConferenceConfig(**conference_data)


def save_conference_config(config: ConferenceConfig, yaml_path: str) -> None:
    """Save a conference configuration to a YAML file."""
    data = config.model_dump()
    data = _enum_to_str(data)
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, indent=2)


def enforce_no_duplicate_participants_in_subsessions(session, participants_by_name):
    """Ensure no participant appears in more than one direct subsession.

    Args:
        session: The session to check.
        participants_by_name: Mapping of participant names to participant objects.
    """
    if session.subsessions:
        # For each direct subsession, collect all participants
        subsession_participants = []
        for subsession in session.subsessions:
            models_in_subsession = set()
            if subsession.working_groups:
                for wg in subsession.working_groups:
                    models_in_subsession.update(wg.participants)
            subsession_participants.append(models_in_subsession)
        # Check for overlap between any two subsessions
        for i in range(len(subsession_participants)):
            for j in range(i + 1, len(subsession_participants)):
                overlap = subsession_participants[i] & subsession_participants[j]
                if overlap:
                    raise ValueError(
                        f"Duplicate participant(s) {overlap} found in multiple "
                        f"subsessions of session {session.name!r}"
                    )
        # Recursively check nested subsessions
        for subsession in session.subsessions:
            enforce_no_duplicate_participants_in_subsessions(
                subsession, participants_by_name
            )
    elif session.working_groups:
        # No subsessions, nothing to check at this level
        return


def enforce_working_group_constraints(config):
    """Enforce that all models in a working group are compatible.

    This includes:
    - All models have the same in_channels, out_channels, and model_tag.
    - No duplicate participants in subsessions.
    """
    participants_by_name = {p.name: p for p in config.participants}

    def check_groups(session):
        """Check the working groups in a session.

        Args:
            session: The session to check.
        """
        if session.working_groups:
            for wg in session.working_groups:
                if not wg.participants:
                    continue
                ref = participants_by_name[wg.participants[0]]
                ref_in = ref.in_channels
                ref_out = ref.out_channels
                ref_tag = ref.model_tag
                for pname in wg.participants[1:]:
                    p = participants_by_name[pname]
                    if p.in_channels != ref_in:
                        raise ValueError(
                            f"Working group {wg.name!r} in session "
                            f"{session.name!r}: All models must have the same "
                            f"in_channels. {p.name!r} has {p.in_channels}, "
                            f"expected {ref_in}."
                        )
                    if p.out_channels != ref_out:
                        raise ValueError(
                            f"Working group {wg.name!r} in session "
                            f"{session.name!r}: All models must have the same "
                            f"out_channels. {p.name!r} has {p.out_channels}, "
                            f"expected {ref_out}."
                        )
                    if p.model_tag != ref_tag:
                        raise ValueError(
                            f"Working group {wg.name!r} in session "
                            f"{session.name!r}: All models must have the same "
                            f"model_tag. {p.name!r} has {p.model_tag!r}, "
                            f"expected {ref_tag!r}."
                        )
        if session.subsessions:
            for subsession in session.subsessions:
                check_groups(subsession)

    for session in config.parallel_sessions:
        check_groups(session)
        enforce_no_duplicate_participants_in_subsessions(
            session, participants_by_name
        )


def validate_yaml_schema(yaml_path: str) -> bool:
    """Validate that a YAML file conforms to the expected schema.

    And working group constraints.
    """
    try:
        config = load_conference_config(yaml_path)
        enforce_working_group_constraints(config)
        return True
    except Exception as e:
        print(f"Validation error: {e}")
        return False
