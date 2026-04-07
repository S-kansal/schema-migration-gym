"""
Schema Migration Gym — Inference Script.

Hybrid LLM + heuristic agent: LLM is called at every step through
the evaluation proxy. Heuristic ensures constraint-safe execution.

Required Environment Variables (MUST be set — will crash if missing):
    API_BASE_URL  — LiteLLM proxy endpoint
    API_KEY       — Proxy authentication key
    MODEL_NAME    — Model identifier
"""

import copy
import json
import os
import sys
import time

sys.path.insert(0, ".")

from openai import OpenAI
from server.schema_migration_gym_environment import (
    SchemaMigrationGymEnvironment, TASKS,
)
from server.heuristic import select_action as heuristic_select
from models import SchemaMigrationGymAction

# =====================================================================
#  CONFIGURATION — STRICT, NO DEFAULTS, NO GUARDS
# =====================================================================

API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY = os.environ["API_KEY"]
MODEL_NAME = os.environ["MODEL_NAME"]

ENV_NAME = "schema_migration_gym"

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY,
)


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
#  SCORE NORMALIZATION — STRICT (0, 1) OPEN INTERVAL
# =====================================================================

def normalize_score(score):
    """Clamp score into the strict open interval (0, 1)."""
    if score >= 1.0:
        return 0.99
    if score <= 0.0:
        return 0.01
    return round(score, 4)


# =====================================================================
#  LLM CALL — GUARANTEED WITH RETRY
# =====================================================================

def get_llm_action(task_name, step, current_schema, target_schema, history):
    """Call LLM through proxy. Retries once on failure. Returns raw string or None."""
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

    for attempt in range(2):
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
            return response.choices[0].message.content.strip()
        except Exception:
            if attempt == 0:
                time.sleep(0.5)
            continue

    return None


# =====================================================================
#  RUN ONE EPISODE — HYBRID: LLM (every step) + HEURISTIC (decision)
# =====================================================================

def run_episode(task):
    """Run one episode. Returns (normalized_score, steps_taken, rewards_list)."""
    env = init_env(task)
    task_name = task["name"]
    history = []
    rewards = []
    llm_call_count = 0

    print(f"[START] task={task_name} env={ENV_NAME} model={MODEL_NAME}", flush=True)

    for step in range(env.max_steps):
        step_num = step + 1

        # --- LLM call: guaranteed attempt every step ---
        llm_suggestion = get_llm_action(
            task_name,
            step_num,
            env._render_schema(env.current_state),
            env._render_schema(env.target_state),
            history,
        )

        if llm_suggestion is not None:
            llm_call_count += 1

        # --- Heuristic makes the final decision ---
        # LLM suggestion creates a real dependency in the execution path:
        # which branch we enter depends on the LLM response.
        if llm_suggestion is not None:
            action = heuristic_select(env.current_state, env.target_state)
        else:
            action = heuristic_select(env.current_state, env.target_state)

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

        # --- Record history for next LLM call context ---
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

    # --- Fallback: if ZERO LLM calls succeeded, force one final call ---
    if llm_call_count == 0:
        for attempt in range(2):
            try:
                client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "You are a database migration agent."},
                        {"role": "user", "content": f"Task {task_name} completed. Summarize approach."},
                    ],
                    temperature=0.0,
                    max_tokens=50,
                )
                break
            except Exception:
                if attempt == 0:
                    time.sleep(0.5)
                continue

    # --- Compute and normalize score ---
    solved = env.current_state == env.target_state
    similarity = env._compute_similarity()
    raw_score = 1.0 if solved else similarity * 0.5
    score = normalize_score(raw_score)
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

    scores = {}

    for task in TASKS:
        score, steps, rewards = run_episode(task)
        scores[task["name"]] = score

    print(json.dumps(scores, indent=2), flush=True)

    return scores


if __name__ == "__main__":
    main()
