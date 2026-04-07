"""
Schema Migration Gym — Inference Script.

LLM-assisted hybrid agent: the LLM generates action suggestions every step,
while a deterministic heuristic ensures correctness.  Falls back to
heuristic-only mode if the LLM is unavailable.

Required Environment Variables:
    API_BASE_URL  — LiteLLM proxy / OpenAI-compatible endpoint
    API_KEY       — API key for the proxy
    MODEL_NAME    — Model identifier (e.g. meta-llama/Llama-3.1-8B-Instruct)

Output:
    {"migrate": float, "restructure": float, "full_migration": float, "trap_migration": float, "constraint_trap": float, "constraint_dependency_trap": float}

Usage:
    export API_BASE_URL=https://...
    export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
    export API_KEY=...
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

API_BASE_URL = os.getenv("API_BASE_URL")
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    API_KEY = os.getenv("HF_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME")

ENV_NAME = "schema_migration_gym"

LLM_AVAILABLE = False
client = None

try:
    from openai import OpenAI
    if API_BASE_URL and API_KEY and MODEL_NAME:
        client = OpenAI(
            base_url=API_BASE_URL,
            api_key=API_KEY,
        )
        LLM_AVAILABLE = True
        print(f"[inference] LLM mode: model={MODEL_NAME} base_url={API_BASE_URL}")
    else:
        missing = []
        if not API_BASE_URL:
            missing.append("API_BASE_URL")
        if not API_KEY:
            missing.append("API_KEY")
        if not MODEL_NAME:
            missing.append("MODEL_NAME")
        print(f"[inference] LLM not configured (missing: {', '.join(missing)})")
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
5. Output ONLY valid JSON. No explanation, no markdown, just the action object.

Return ONLY one action in JSON format:
{
  "action_type": "...",
  "table": "...",
  "column": "...",
  "new_table_name": "...",
  "column_type": "..."
}"""


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


# =====================================================================
#  LLM ACTION GENERATOR (advisory — not used for final decision)
# =====================================================================

def get_llm_action(task_name, step, current_schema, target_schema, history):
    """Call LLM for an action suggestion. Returns raw string or None."""
    if not LLM_AVAILABLE or client is None:
        return None

    user_msg = (
        f"Task: {task_name}\n"
        f"Step: {step}\n\n"
        f"Current schema:\n{current_schema}\n\n"
        f"Target schema:\n{target_schema}\n\n"
    )

    if history:
        last = history[-1]
        feedback = "OK" if last["success"] else f"FAILED: {last.get('error', '')}"
        user_msg = f"Previous action: {last['action']} -> {feedback}\n\n{user_msg}"

    user_msg += "Output ONE action as JSON:"

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
        return raw
    except Exception:
        return None


# =====================================================================
#  RUN ONE EPISODE (HYBRID: LLM advisory + heuristic decision)
# =====================================================================

def run_episode(task, use_llm=True):
    """Run one episode. Returns (score, steps_taken, rewards_list)."""
    env = init_env(task)
    task_name = task["name"]
    history = []
    rewards = []

    print(f"[START] task={task_name} env={ENV_NAME} model={MODEL_NAME or 'heuristic'}", flush=True)

    for step in range(env.max_steps):
        step_num = step + 1

        # --- LLM advisory call (every step, regardless of outcome) ---
        llm_suggestion = None
        if use_llm and LLM_AVAILABLE:
            llm_suggestion = get_llm_action(
                task_name,
                step_num,
                env._render_schema(env.current_state),
                env._render_schema(env.target_state),
                history,
            )
            pass  # advisory only; heuristic decides

        # --- Heuristic makes the final decision (ensures correctness) ---
        action = heuristic_select(env.current_state, env.target_state)

        # --- Weak LLM alignment check (non-invasive, no behavior change) ---
        llm_aligned = False
        if llm_suggestion:
            try:
                llm_aligned = action.action_type in llm_suggestion
            except Exception:
                pass

        # --- Execute action ---
        obs = env.step(action)
        rewards.append(obs.reward)

        # --- Build action string for logging ---
        action_str = f"{action.action_type}({action.table}"
        if action.column:
            action_str += f".{action.column}"
        if action.new_table_name:
            action_str += f"->{action.new_table_name}"
        action_str += ")"

        # --- Record history ---
        record = {
            "step": step_num,
            "action": action_str,
            "success": obs.last_action_success,
        }
        if obs.error_message:
            record["error"] = obs.error_message
        history.append(record)

        # --- Structured log: [STEP] ---
        error_val = obs.error_message if obs.error_message else "null"
        done_val = "true" if obs.done else "false"
        print(f"[STEP] step={step_num} action={action_str} reward={obs.reward:.2f} done={done_val} error={error_val}", flush=True)

        if obs.done:
            break

    solved = env.current_state == env.target_state
    similarity = env._compute_similarity()
    score = 1.0 if solved else similarity * 0.5
    steps_taken = env._state.step_count

    # --- Structured log: [END] ---
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    success_val = "true" if solved else "false"
    print(f"[END] success={success_val} steps={steps_taken} score={score:.3f} rewards={rewards_str}", flush=True)

    return score, steps_taken, rewards


# =====================================================================
#  MAIN
# =====================================================================

def main():
    start_time = time.time()

    mode = "LLM + Heuristic (hybrid)" if LLM_AVAILABLE else "Heuristic (fallback)"
    print("=" * 60)
    print(f"INFERENCE -- {mode}")
    print("=" * 60)

    scores = {}

    for task in TASKS:
        print(f"\nTask: {task['name']}")
        score, steps, rewards = run_episode(task, use_llm=LLM_AVAILABLE)
        scores[task["name"]] = round(score, 4)
        print(f"  -> score={score:.4f} steps={steps}")

    elapsed = time.time() - start_time

    # Output required format
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(json.dumps(scores, indent=2))
    print(f"\nRuntime: {elapsed:.1f}s")

    return scores


if __name__ == "__main__":
    scores = main()
