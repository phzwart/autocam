"""Utility functions for building conferences with maximum model mixing."""

from typing import List, Optional
import numpy as np
import re

from .models import (
    Participant, 
    WorkingGroup, 
    ParallelSession, 
    ConferenceConfig, 
    TrainingMandate,
    ModelType,
    Dimension
)


def create_conference_grid(
    participants: List[Participant],
    num_sessions: int,
    working_groups_per_session: int,
    session_names: Optional[List[str]] = None,
    base_name: str = "model",
    training_mandate: TrainingMandate = TrainingMandate.ONE_VS_ONE,
    mixing_strategy: str = "max"
) -> ConferenceConfig:
    """Create a conference with maximum model mixing between sessions.
    
    Args:
        participants: List of existing Participant objects
        num_sessions: Number of parallel sessions
        working_groups_per_session: Number of working groups per session
        session_names: Optional list of session names
        base_name: Base name for model participants (used for validation)
        training_mandate: Training mandate for all working groups
        mixing_strategy: "max" for maximum mixing, "min" for minimal mixing
        
    Returns:
        ConferenceConfig with maximum model mixing
    """
    
    num_models = len(participants)
    
    # Get participant names for use in working groups
    participant_names = [p.name for p in participants]
    
    # Generate session names if not provided
    if session_names is None:
        session_names = [f"Session_{i+1}" for i in range(num_sessions)]
    else:
        # Validate that session_names length matches num_sessions
        if len(session_names) != num_sessions:
            raise ValueError(
                f"Number of session names ({len(session_names)}) must match "
                f"num_sessions ({num_sessions})"
            )
    
    # Validate that we have enough models for the requested configuration
    total_groups = num_sessions * working_groups_per_session
    
    if num_models < total_groups:
        raise ValueError(
            f"Not enough models ({num_models}) for {num_sessions} sessions with "
            f"{working_groups_per_session} groups per session. "
            f"Need at least {total_groups} models."
        )
    
    # Create sessions with maximum mixing
    sessions = []
    
    for session_idx in range(num_sessions):
        working_groups = []
        
        for group_idx in range(working_groups_per_session):
            # Each session contains ALL models, but grouped differently
            if mixing_strategy == "max":
                # Use different grouping strategies per session
                if session_idx == 0:
                    # First session: consecutive groups
                    start_idx = group_idx * (num_models // working_groups_per_session)
                    end_idx = start_idx + (num_models // working_groups_per_session)
                    model_indices = list(range(start_idx, min(end_idx, num_models)))
                else:
                    # Later sessions: strided patterns across all models
                    stride = session_idx + 1
                    start_offset = group_idx * (num_models // working_groups_per_session)
                    model_indices = []
                    for i in range(num_models // working_groups_per_session):
                        model_idx_strided = (start_offset + i * stride) % num_models
                        model_indices.append(model_idx_strided)
            else:
                # Minimal mixing: each session gets its own models
                start_idx = group_idx * (num_models // working_groups_per_session)
                end_idx = start_idx + (num_models // working_groups_per_session)
                model_indices = list(range(start_idx, min(end_idx, num_models)))
            
            # Create working group
            group_name = f"group{session_idx+1}{group_idx+1}"
            group_description = ",".join([participant_names[i] for i in model_indices])
            
            working_group = WorkingGroup(
                name=group_name,
                description=group_description,
                participants=[participant_names[i] for i in model_indices],
                training_mandate=training_mandate,
                student_participants=[participant_names[i] for i in model_indices],
                target_participants=[participant_names[i] for i in model_indices],
                training_epochs=100
            )
            working_groups.append(working_group)
        
        # Create session
        session = ParallelSession(
            name=session_names[session_idx],
            description=f"Session {session_idx + 1}",
            working_groups=working_groups,
            training_epochs=100
        )
        sessions.append(session)
    
    # Create conference config
    conference_config = ConferenceConfig(
        name="AutoConference",
        description=f"Auto-generated conference with {num_models} models",
        participants=participants,
        parallel_sessions=sessions
    )
    
    return conference_config


def create_conference_with_custom_mixing(
    num_models: int,
    sessions_config: List[dict]
) -> ConferenceConfig:
    """Create conference with custom mixing patterns.
    
    Args:
        num_models: Number of models
        sessions_config: List of session configurations
            Each dict should have:
            - name: session name
            - groups: list of group configs
              Each group config should have:
              - name: group name  
              - pattern: "consecutive", "strided", or custom slice
              - size: number of models per group
    
    Returns:
        ConferenceConfig with custom mixing
    """
    
    # Create participants
    participants = []
    for i in range(num_models):
        participant = Participant(
            name=f"model_{i}",
            model_type=ModelType.CNN,
            model_tag="encoder", 
            in_channels=1,
            out_channels=3,
            dimension=Dimension.D2,
            config={"layers": 5, "alpha": 0.5, "gamma": 0.0}
        )
        participants.append(participant)
    
    sessions = []
    for session_config in sessions_config:
        working_groups = []
        
        for group_config in session_config["groups"]:
            pattern = group_config["pattern"]
            size = group_config["size"]
            
            # Generate model indices based on pattern
            if pattern == "consecutive":
                model_indices = list(range(size))
            elif pattern == "strided":
                model_indices = list(range(0, num_models, 2))[:size]
            else:
                # Custom slice pattern
                model_indices = list(range(num_models))[pattern][:size]
            
            working_group = WorkingGroup(
                name=group_config["name"],
                description=",".join([f"model_{i}" for i in model_indices]),
                participants=[f"model_{i}" for i in model_indices],
                training_mandate=TrainingMandate.ONE_VS_ONE,
                student_participants=[f"model_{i}" for i in model_indices],
                target_participants=[f"model_{i}" for i in model_indices],
                training_epochs=100
            )
            working_groups.append(working_group)
        
        session = ParallelSession(
            name=session_config["name"],
            description=f"Custom session: {session_config['name']}",
            working_groups=working_groups,
            training_epochs=100
        )
        sessions.append(session)
    
    return ConferenceConfig(
        name="CustomConference",
        description="Conference with custom mixing patterns",
        participants=participants,
        parallel_sessions=sessions
    ) 