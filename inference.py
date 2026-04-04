"""
Schema Migration Gym — Inference Script.

Uses an LLM via the OpenAI-compatible API to solve schema migration tasks.
Falls back to a deterministic heuristic agent if the LLM is unavailable.

Required Environment Variables:
    API_BASE_URL  — API endpoint (default: https://router.huggingface.co/v1)
    MODEL_NAME    — Model identifier (e.g. meta-llama/Llama-3.1-8B-Instruct)
    HF_TOKEN      — HuggingFace API token

Output:
    {"migrate": float, "restructure": float, "full_migration": float, "trap_migration": float, "constraint_trap": float, "constraint_dependency_trap": float}

Usage:
    export API_BASE_URL=https://router.huggingface.co/v1
    export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
    export HF_TOKEN=hf_...
    python inference.py
"""

import copy
import json
import os
import sys
import time

sys.path.insert(0, ".")

from server.schema_migration_gym_environment import (
    SchemaMigrationGymEnvironment, TASKS,
)
from server.heuristic import select_action as heuristic_select
from models import SchemaMigrationGymAction

# =====================================================================
#  CONFIGURATION
# =====================================================================

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")

LLM_AVAILABLE = False
client = None

try:
    from openai import OpenAI
    if MODEL_NAME and HF_TOKEN:
        client = OpenAI(
            base_url=API_BASE_URL,
            api_key=HF_TOKEN,
        )
        LLM_AVAILABLE = True
        print(f"[inference] LLM mode: model={MODEL_NAME} base_url={API_BASE_URL}")
    else:
        print("[inference] LLM not configured (missing MODEL_NAME or HF_TOKEN)")
        print("[inference] Falling back to heuristic agent")
except ImportError:
    print("[inference] openai package not installed -- using heuristic fallback")


# =====================================================================
#  SYSTEM PROMPT
# =====================================================================

SYSTEM_PROMPT = """You are a database migration agent. You receive a current schema and a target schema.
You must output ONE action to transform the current schema toward the target.

Available actions (output as JSON):
- {"action_type": "ADD_COLUMN", "table": "TABLE", "column": "COLUMN", "column_type": "INTEGER|TEXT|BOOLEAN"}
- {"action_type": "DROP_COLUMN", "table": "TABLE", "column": "COLUMN"}
- {"action_type": "DROP_TABLE", "table": "TABLE"}
- {"action_type": "RENAME_TABLE", "table": "OLD_NAME", "new_table_name": "NEW_NAME"}
- {"action_type": "SET_NOT_NULL", "table": "TABLE", "column": "COLUMN"}
- {"action_type": "SET_PRIMARY_KEY", "table": "TABLE", "column": "COLUMN"}

IMPORTANT RULES:
1. SET_PRIMARY_KEY REQUIRES the column to already have NOT NULL set. Always SET_NOT_NULL first.
2. DROP_TABLE is irreversible. Only drop tables that are NOT in the target.
3. RENAME_TABLE before operating on columns of the renamed table.
4. For ADD_COLUMN, specify column_type matching the target schema type.
5. Output ONLY valid JSON. No explanation, no markdown, just the action object."""


# =====================================================================
#  ENVIRONMENT INIT
# =====================================================================

def init_env(task):
    env = SchemaMigrationGymEnvironment()
    env._state.step_count = 0
    env.current_task = task
    env.current_state = copy.deepcopy(task["start"])
    env.target_state = copy.deepcopy(task["target"])
    env._prev_similarity = env._compute_similarity()
    env._visited_states = {env._hash_state(env.current_state)}
    return env


# Heuristic is imported from server.heuristic as heuristic_select


# =====================================================================
#  LLM AGENT
# =====================================================================

def llm_select(env, history):
    """Call LLM for one action. Returns action or None on failure."""
    if not LLM_AVAILABLE or client is None:
        return None

    user_msg = (
        f"Current schema:\n{env._render_schema(env.current_state)}\n\n"
        f"Target schema:\n{env._render_schema(env.target_state)}\n\n"
        f"Step {env._state.step_count + 1}/{env.max_steps}."
    )

    if history:
        last = history[-1]
        feedback = "OK" if last["success"] else f"FAILED: {last.get('error', '')}"
        user_msg = f"Previous: {last['action']} -> {feedback}\n\n{user_msg}"

    user_msg += "\n\nOutput ONE action as JSON:"

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=200,
        )
        raw = response.choices[0].message.content.strip()

        # Handle markdown code blocks
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        if raw.startswith("{") and raw.endswith("}"):
            action_data = json.loads(raw)
            return SchemaMigrationGymAction(**action_data)
        else:
            # Try to extract JSON from the response
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                action_data = json.loads(raw[start:end])
                return SchemaMigrationGymAction(**action_data)
    except Exception as e:
        print(f"  [LLM error: {e}]")

    return None


# =====================================================================
#  RUN ONE EPISODE
# =====================================================================

def run_episode(task, use_llm=True):
    """Run one episode. Returns score (0.0-1.0)."""
    env = init_env(task)
    history = []
    cumulative = 0.0

    for step in range(env.max_steps):
        action = None

        # Try LLM first
        if use_llm and LLM_AVAILABLE:
            action = llm_select(env, history)

        # Fallback to heuristic
        if action is None:
            action = heuristic_select(env.current_state, env.target_state)

        obs = env.step(action)
        cumulative += obs.reward

        record = {
            "step": step + 1,
            "action": f"{action.action_type}({action.table}.{action.column})",
            "success": obs.last_action_success,
        }
        if obs.error_message:
            record["error"] = obs.error_message
        history.append(record)

        ok = "OK" if obs.last_action_success else "FAIL"
        print(f"  step {step+1}: {action.action_type}({action.table}"
              f"{'.' + action.column if action.column else ''}"
              f"{'->' + action.new_table_name if action.new_table_name else ''}"
              f") [{ok}]")
        print(f"[STEP] task={task['name']} step={step+1} action={action.action_type}")

        if obs.done:
            break

    solved = env.current_state == env.target_state
    similarity = env._compute_similarity()

    # Score: 1.0 if solved, else similarity * 0.5
    score = 1.0 if solved else similarity * 0.5
    return score


# =====================================================================
#  MAIN
# =====================================================================

def main():
    start_time = time.time()

    mode = "LLM" if LLM_AVAILABLE else "Heuristic (fallback)"
    print("[START] Inference started")
    print("=" * 60)
    print(f"INFERENCE -- {mode}")
    print("=" * 60)

    scores = {}

    for task in TASKS:
        print(f"\nTask: {task['name']}")
        score = run_episode(task, use_llm=LLM_AVAILABLE)
        scores[task["name"]] = round(score, 4)
        print(f"  -> score={score:.4f}")
        print(f"[END] task={task['name']} score={score:.4f}")

    elapsed = time.time() - start_time

    # Output required format
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(json.dumps(scores, indent=2))
    print(f"\nRuntime: {elapsed:.1f}s")

    # Also write to stdout as pure JSON (for automated parsing)
    # The last line of output is the JSON result
    return scores


if __name__ == "__main__":
    scores = main()
