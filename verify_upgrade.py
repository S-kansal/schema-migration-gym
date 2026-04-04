"""
Verification script for the multi-step environment upgrade.

Tests:
  1. Heuristic agent solves: migrate=6, restructure=8, full_migration=10
  2. SET_PK before SET_NN → FAIL
  3. Reward: positive on progress, step penalty, completion bonus
  4. DROP_TABLE trap → unrecoverable
  5. DROP_COLUMN validation
  6. All rejection cases

Usage:
    python verify_upgrade.py
"""

import copy
import sys

sys.path.insert(0, ".")

from server.schema_migration_gym_environment import (
    SchemaMigrationGymEnvironment, TASKS,
)
from models import SchemaMigrationGymAction


PASS = 0
FAIL = 0


def check(name, condition):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✓ {name}")
    else:
        FAIL += 1
        print(f"  ✗ {name}")


# =====================================================================
#  CHAIN-AWARE HEURISTIC AGENT
# =====================================================================

class ChainHeuristicAgent:
    """
    Solves tasks optimally by following dependency chains:
    1. RENAME tables (match extra→missing by column overlap)
    2. DROP extra tables
    3. DROP extra columns
    4. ADD missing columns
    5. SET_NOT_NULL (must come before SET_PK)
    6. SET_PRIMARY_KEY (requires NOT_NULL)
    """

    def select_action(self, env):
        cur = env.current_state["tables"]
        tgt = env.target_state["tables"]

        # Phase 1: RENAME tables
        cur_names = set(cur.keys())
        tgt_names = set(tgt.keys())
        missing = tgt_names - cur_names
        extra = cur_names - tgt_names

        for m in missing:
            best, best_score = None, -1
            for e in extra:
                shared = set(cur[e]["columns"].keys()) & set(tgt[m]["columns"].keys())
                if len(shared) > best_score:
                    best_score = len(shared)
                    best = e
            if best and best_score > 0:
                return SchemaMigrationGymAction(
                    action_type="RENAME_TABLE", table=best, new_table_name=m)

        # Phase 2: DROP extra tables
        for e in extra:
            return SchemaMigrationGymAction(action_type="DROP_TABLE", table=e)

        # Phase 3: DROP extra columns
        for tname in tgt:
            if tname not in cur:
                continue
            for cname in list(cur[tname]["columns"].keys()):
                if cname not in tgt[tname]["columns"]:
                    return SchemaMigrationGymAction(
                        action_type="DROP_COLUMN", table=tname, column=cname)

        # Phase 4: ADD missing columns
        for tname, tdata in tgt.items():
            if tname not in cur:
                continue
            for cname in tdata["columns"]:
                if cname not in cur[tname]["columns"]:
                    return SchemaMigrationGymAction(
                        action_type="ADD_COLUMN", table=tname, column=cname)

        # Phase 5: SET_NOT_NULL
        for tname, tdata in tgt.items():
            if tname not in cur:
                continue
            for cname, cprops in tdata["columns"].items():
                if cname not in cur[tname]["columns"]:
                    continue
                if cprops["not_null"] and not cur[tname]["columns"][cname]["not_null"]:
                    return SchemaMigrationGymAction(
                        action_type="SET_NOT_NULL", table=tname, column=cname)

        # Phase 6: SET_PRIMARY_KEY (after NOT_NULL is set)
        for tname, tdata in tgt.items():
            if tname not in cur:
                continue
            for cname, cprops in tdata["columns"].items():
                if cname not in cur[tname]["columns"]:
                    continue
                if cprops["primary_key"] and not cur[tname]["columns"][cname]["primary_key"]:
                    return SchemaMigrationGymAction(
                        action_type="SET_PRIMARY_KEY", table=tname, column=cname)

        # Fallback — should never reach here if task is solvable
        return SchemaMigrationGymAction(action_type="SET_NOT_NULL", table="users", column="id")


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
#  TEST 1: HEURISTIC SOLVES ALL TASKS IN EXACT STEPS
# =====================================================================

def test_heuristic():
    print("\n" + "=" * 60)
    print("TEST 1: Heuristic Agent — Exact Step Counts")
    print("=" * 60)

    expected = {"migrate": 6, "restructure": 8, "full_migration": 10}
    agent = ChainHeuristicAgent()

    for task in [t for t in TASKS if t["name"] in expected]:
        env = init_env(task)
        cumulative = 0.0

        print(f"\n  Task: {task['name']} (expected {expected[task['name']]} steps)")

        for step in range(30):
            action = agent.select_action(env)
            obs = env.step(action)
            cumulative += obs.reward
            ok = "OK" if obs.last_action_success else "FAIL"
            print(f"    step {step+1}: {action.action_type}({action.table}"
                  f"{'.'+action.column if action.column else ''}"
                  f"{'→'+action.new_table_name if action.new_table_name else ''}"
                  f") [{ok}] r={obs.reward:+.4f}")
            if obs.done:
                break

        steps = env._state.step_count
        solved = env.current_state == env.target_state
        check(f"{task['name']}: solved={solved}", solved)
        check(f"{task['name']}: steps={steps} == {expected[task['name']]}",
              steps == expected[task['name']])
        check(f"{task['name']}: cumulative reward > 0", cumulative > 0)


# =====================================================================
#  TEST 2: SET_PK BEFORE SET_NN → MUST FAIL
# =====================================================================

def test_pk_requires_nn():
    print("\n" + "=" * 60)
    print("TEST 2: SET_PK Before SET_NN → Must Fail")
    print("=" * 60)

    task = TASKS[0]  # migrate
    env = init_env(task)

    # First rename so table exists as "users"
    obs = env.step(SchemaMigrationGymAction(
        action_type="RENAME_TABLE", table="tmp_users", new_table_name="users"))
    check("RENAME succeeds", obs.last_action_success)

    # Try SET_PK on id — should FAIL because id.not_null is False
    obs = env.step(SchemaMigrationGymAction(
        action_type="SET_PRIMARY_KEY", table="users", column="id"))
    check("SET_PK(id) REJECTED (not_null is False)", not obs.last_action_success)
    check("Error mentions NOT NULL",
          obs.error_message is not None and "NOT NULL" in obs.error_message)

    # Now SET_NN first, then SET_PK should work
    obs = env.step(SchemaMigrationGymAction(
        action_type="SET_NOT_NULL", table="users", column="id"))
    check("SET_NN(id) succeeds", obs.last_action_success)

    obs = env.step(SchemaMigrationGymAction(
        action_type="SET_PRIMARY_KEY", table="users", column="id"))
    check("SET_PK(id) NOW succeeds (after SET_NN)", obs.last_action_success)


# =====================================================================
#  TEST 3: REWARD BEHAVIOR
# =====================================================================

def test_reward():
    print("\n" + "=" * 60)
    print("TEST 3: Reward Behavior")
    print("=" * 60)

    task = TASKS[0]  # migrate
    env = init_env(task)
    agent = ChainHeuristicAgent()

    rewards = []
    for step in range(30):
        action = agent.select_action(env)
        obs = env.step(action)
        rewards.append(obs.reward)
        if obs.done:
            break

    # All intermediate rewards should be positive (progress + step penalty)
    # or at least the delta should be positive
    check("All non-final rewards include step penalty",
          all(r < 1.0 for r in rewards[:-1]))
    check("Final reward includes completion bonus (>= 1.0)", rewards[-1] >= 1.0)
    check("Total reward > 0", sum(rewards) > 0)

    # Invalid action → negative reward
    env2 = init_env(task)
    obs = env2.step(SchemaMigrationGymAction(
        action_type="ADD_COLUMN", table="nonexistent", column="x"))
    check("Invalid action reward = -0.1", obs.reward == -0.1)


# =====================================================================
#  TEST 4: DROP_TABLE TRAP
# =====================================================================

def test_trap():
    print("\n" + "=" * 60)
    print("TEST 4: DROP_TABLE Trap (Unrecoverable)")
    print("=" * 60)

    task = TASKS[0]  # migrate
    env = init_env(task)

    # Rename, then drop the critical table
    env.step(SchemaMigrationGymAction(
        action_type="RENAME_TABLE", table="tmp_users", new_table_name="users"))
    obs = env.step(SchemaMigrationGymAction(
        action_type="DROP_TABLE", table="users"))
    check("DROP_TABLE(users) succeeds", obs.last_action_success)

    # Now try to solve — should be impossible
    agent = ChainHeuristicAgent()
    for _ in range(28):
        action = agent.select_action(env)
        obs = env.step(action)
        if obs.done:
            break

    solved = env.current_state == env.target_state
    check("Task is UNSOLVABLE after DROP_TABLE", not solved)
    check("Episode ends by timeout", env._state.step_count >= env.max_steps)


# =====================================================================
#  TEST 5: DROP_COLUMN VALIDATION
# =====================================================================

def test_drop_column():
    print("\n" + "=" * 60)
    print("TEST 5: DROP_COLUMN Validation")
    print("=" * 60)

    task = TASKS[0]
    env = init_env(task)

    # Rename first
    env.step(SchemaMigrationGymAction(
        action_type="RENAME_TABLE", table="tmp_users", new_table_name="users"))

    # DROP_COLUMN on existing column
    obs = env.step(SchemaMigrationGymAction(
        action_type="DROP_COLUMN", table="users", column="legacy_flag"))
    check("DROP_COLUMN(legacy_flag) succeeds", obs.last_action_success)
    check("legacy_flag removed from state",
          "legacy_flag" not in env.current_state["tables"]["users"]["columns"])

    # DROP_COLUMN on non-existent column
    obs = env.step(SchemaMigrationGymAction(
        action_type="DROP_COLUMN", table="users", column="legacy_flag"))
    check("DROP_COLUMN(legacy_flag) again → FAIL", not obs.last_action_success)

    # DROP_COLUMN on non-existent table
    obs = env.step(SchemaMigrationGymAction(
        action_type="DROP_COLUMN", table="phantom", column="x"))
    check("DROP_COLUMN(phantom.x) → FAIL", not obs.last_action_success)

    # DROP_COLUMN missing fields
    obs = env.step(SchemaMigrationGymAction(
        action_type="DROP_COLUMN", table="users"))
    check("DROP_COLUMN(no column) → FAIL", not obs.last_action_success)


# =====================================================================
#  TEST 6: ALL REJECTION CASES
# =====================================================================

def test_rejections():
    print("\n" + "=" * 60)
    print("TEST 6: All Rejection Cases")
    print("=" * 60)

    task = TASKS[0]
    env = init_env(task)

    # Unknown action type
    obs = env.step(SchemaMigrationGymAction(
        action_type="CREATE_TABLE", table="foo"))
    check("Unknown action type → FAIL", not obs.last_action_success)

    # ADD_COLUMN to non-existent table
    obs = env.step(SchemaMigrationGymAction(
        action_type="ADD_COLUMN", table="users", column="x"))
    check("ADD_COLUMN to 'users' (not yet renamed) → FAIL", not obs.last_action_success)

    # SET_NN on non-existent column
    obs = env.step(SchemaMigrationGymAction(
        action_type="SET_NOT_NULL", table="tmp_users", column="phantom"))
    check("SET_NN(phantom) → FAIL", not obs.last_action_success)

    # RENAME to existing table name
    env2 = init_env(TASKS[2])  # full_migration (has tmp_users AND orders)
    obs = env2.step(SchemaMigrationGymAction(
        action_type="RENAME_TABLE", table="tmp_users", new_table_name="orders"))
    check("RENAME to existing table → FAIL", not obs.last_action_success)

    # RENAME missing field
    obs = env2.step(SchemaMigrationGymAction(action_type="RENAME_TABLE", table="tmp_users"))
    check("RENAME missing new_table_name → FAIL", not obs.last_action_success)


# =====================================================================
#  MAIN
# =====================================================================

if __name__ == "__main__":
    test_heuristic()
    test_pk_requires_nn()
    test_reward()
    test_trap()
    test_drop_column()
    test_rejections()

    print("\n" + "=" * 60)
    print(f"RESULTS: {PASS} passed, {FAIL} failed")
    print("=" * 60)

    if FAIL > 0:
        print("\n⚠ SOME TESTS FAILED")
        sys.exit(1)
    else:
        print("\n✓ ALL TESTS PASSED")
        sys.exit(0)
