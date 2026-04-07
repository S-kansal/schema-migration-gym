# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Schema Migration Gym Environment.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - GET /tasks: List available tasks with metadata
    - POST /baseline: Run baseline agents, return scores
    - POST /grader: Grade an episode history
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
"""

import copy
import random

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import SchemaMigrationGymAction, SchemaMigrationGymObservation
    from .schema_migration_gym_environment import SchemaMigrationGymEnvironment, TASKS
    from .heuristic import select_action as _heuristic_select
except (ImportError, ModuleNotFoundError):
    from models import SchemaMigrationGymAction, SchemaMigrationGymObservation
    from server.schema_migration_gym_environment import SchemaMigrationGymEnvironment, TASKS
    from server.heuristic import select_action as _heuristic_select


# Create the app with web interface and README integration
app = create_app(
    SchemaMigrationGymEnvironment,
    SchemaMigrationGymAction,
    SchemaMigrationGymObservation,
    env_name="schema_migration_gym",
    max_concurrent_envs=1,
)


# =====================================================================
#  CUSTOM ENDPOINTS
# =====================================================================

TASK_DIFFICULTY = {
    "migrate": "easy",
    "restructure": "medium",
    "full_migration": "hard",
    "trap_migration": "expert",
    "constraint_trap": "expert",
    "constraint_dependency_trap": "hard",
}


def _normalize_score(s):
    """Clamp score into strict (0, 1) open interval."""
    if s >= 1.0:
        return 0.99
    if s <= 0.0:
        return 0.01
    return round(s, 4)


@app.get("/tasks")
def get_tasks():
    """Returns list of available tasks with metadata and action schema."""
    task_list = []
    for task in TASKS:
        start = task["start"]
        target = task["target"]

        task_list.append({
            "name": task["name"],
            "difficulty": TASK_DIFFICULTY.get(task["name"], "medium"),
            "start_tables": list(start["tables"].keys()),
            "target_tables": list(target["tables"].keys()),
            "start_schema": SchemaMigrationGymEnvironment._render_schema(start),
            "target_schema": SchemaMigrationGymEnvironment._render_schema(target),
        })

    action_schema = {
        "action_type": "ADD_COLUMN | DROP_COLUMN | DROP_TABLE | RENAME_TABLE | SET_PRIMARY_KEY | SET_NOT_NULL",
        "table": "Target table name (required for all actions)",
        "column": "Target column name (required for column actions)",
        "new_table_name": "New table name (required for RENAME_TABLE only)",
        "column_type": "Column type for ADD_COLUMN: INTEGER, TEXT, or BOOLEAN (optional, inferred from target if omitted)",
    }

    return {"tasks": task_list, "action_schema": action_schema}


@app.post("/baseline")
def run_baseline():
    """Runs heuristic baseline on all tasks. Returns {task: score}."""
    scores = {}

    for task in TASKS:
        env = SchemaMigrationGymEnvironment()
        env._state.step_count = 0
        env.current_task = task
        env.current_state = copy.deepcopy(task["start"])
        env.target_state = copy.deepcopy(task["target"])
        env._prev_similarity = env._compute_similarity()
        env._visited_states = {env._hash_state(env.current_state)}

        for _ in range(env.max_steps):
            action = _heuristic_select(env.current_state, env.target_state)
            obs = env.step(action)
            if obs.done:
                break

        solved = env.current_state == env.target_state
        raw = 1.0 if solved else round(env._compute_similarity() * 0.5, 4)
        scores[task["name"]] = _normalize_score(raw)

    return scores


@app.post("/grader")
def grade_episode(episode: dict):
    """
    Grade an episode history.

    Input: {"task": "migrate", "actions": [{"action_type": "...", "table": "...", ...}, ...]}
    Output: {"score": 0.0-1.0, "solved": bool, "steps": int, "reward": float}
    """
    task_name = episode.get("task", "migrate")
    actions = episode.get("actions", [])

    # Find the task
    task = next((t for t in TASKS if t["name"] == task_name), None)
    if task is None:
        return {"error": f"Unknown task '{task_name}'", "score": 0.0}

    env = SchemaMigrationGymEnvironment()
    env._state.step_count = 0
    env.current_task = task
    env.current_state = copy.deepcopy(task["start"])
    env.target_state = copy.deepcopy(task["target"])
    env._prev_similarity = env._compute_similarity()
    env._visited_states = {env._hash_state(env.current_state)}

    cumulative_reward = 0.0
    valid_actions = 0
    invalid_actions = 0

    for action_data in actions:
        if env._state.step_count >= env.max_steps:
            break
        action = SchemaMigrationGymAction(**action_data)
        obs = env.step(action)
        cumulative_reward += obs.reward
        if obs.last_action_success:
            valid_actions += 1
        else:
            invalid_actions += 1
        if obs.done:
            break

    solved = env.current_state == env.target_state
    similarity = env._compute_similarity()

    # Score: 0.0-1.0 based on similarity + solve bonus
    score = similarity * 0.5
    if solved:
        score = 1.0

    return {
        "score": _normalize_score(round(score, 4)),
        "solved": solved,
        "steps": env._state.step_count,
        "total_reward": round(cumulative_reward, 4),
        "similarity": round(similarity, 4),
        "valid_actions": valid_actions,
        "invalid_actions": invalid_actions,
    }


import uvicorn


def main():
    """Entry point for direct execution."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run("server.app:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
