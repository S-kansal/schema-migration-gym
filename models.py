# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Data models for the Schema Migration Gym Environment.

Structured action space (no raw SQL) and structured observation with goal + state.
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field
from typing import List, Optional


# ================= ACTION =================

class SchemaMigrationGymAction(Action):
    """
    Structured action space for RL.

    Agent chooses an operation type and specifies the target table/column.
    6 action types: ADD_COLUMN, DROP_COLUMN, DROP_TABLE, RENAME_TABLE,
    SET_PRIMARY_KEY, SET_NOT_NULL.
    """

    action_type: str = Field(
        ...,
        description="Type of action: ADD_COLUMN, DROP_COLUMN, DROP_TABLE, RENAME_TABLE, SET_PRIMARY_KEY, SET_NOT_NULL"
    )

    table: Optional[str] = Field(
        None,
        description="Target table name"
    )

    column: Optional[str] = Field(
        None,
        description="Target column name"
    )

    new_table_name: Optional[str] = Field(
        None,
        description="New table name (for RENAME_TABLE only)"
    )

    column_type: Optional[str] = Field(
        None,
        description="Column type for ADD_COLUMN: INTEGER, TEXT, or BOOLEAN"
    )


# ================= OBSERVATION =================

class SchemaMigrationGymObservation(Observation):
    """
    Structured observation for RL.

    Agent sees current schema (DDL text), target schema (goal),
    and progress signals (step count, action success, error messages).
    """

    current_schema: str = Field(
        default="",
        description="Current database schema (DDL text)"
    )

    target_schema: str = Field(
        default="",
        description="Target schema the agent must achieve"
    )

    relationships: List[dict] = Field(
        default_factory=list,
        description="Foreign key relationships (reserved for future use)"
    )

    step_count: int = Field(
        default=0,
        description="Current step number"
    )

    max_steps: int = Field(
        default=30,
        description="Maximum allowed steps"
    )

    last_action_success: bool = Field(
        default=True,
        description="Whether last action succeeded"
    )

    error_message: Optional[str] = Field(
        default=None,
        description="Error message if action failed"
    )