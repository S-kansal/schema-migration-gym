# Copyright (c) Meta Platforms, Inc.
# All rights reserved.

"""Schema Migration Gym Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import SchemaMigrationGymAction, SchemaMigrationGymObservation
except ImportError:
    from models import SchemaMigrationGymAction, SchemaMigrationGymObservation


class SchemaMigrationGymClient(
    EnvClient[SchemaMigrationGymAction, SchemaMigrationGymObservation, State]
):
    """
    Client for the Schema Migration Gym Environment.
    """

    # ✅ FIXED (Pydantic v2 compliant)
    def _step_payload(self, action: SchemaMigrationGymAction) -> Dict:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict) -> StepResult[SchemaMigrationGymObservation]:
        obs_data = payload.get("observation", {})

        observation = SchemaMigrationGymObservation(
            current_schema=obs_data.get("current_schema", ""),
            target_schema=obs_data.get("target_schema", ""),
            relationships=obs_data.get("relationships", []),
            step_count=obs_data.get("step_count", 0),
            max_steps=obs_data.get("max_steps", 30),
            last_action_success=obs_data.get("last_action_success", True),
            error_message=obs_data.get("error_message"),
            done=payload.get("done", False),
            reward=payload.get("reward"),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )