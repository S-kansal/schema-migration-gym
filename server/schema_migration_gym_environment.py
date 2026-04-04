# Copyright (c) Meta Platforms, Inc.
# All rights reserved.

"""
Schema Migration Gym Environment — Deterministic RL environment.

State is represented as structured Python dicts (no SQLite).
All transitions are explicit, validated, and use deep copy for safety.

Reward system: delta-based (progress-oriented) with step and loop penalties.

Action ordering enforced:
  - SET_PRIMARY_KEY requires column.not_null == True
  - Creates mandatory chains: ADD_COLUMN → SET_NOT_NULL → SET_PRIMARY_KEY
"""

import copy
import json
import random
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import SchemaMigrationGymAction, SchemaMigrationGymObservation
except ImportError:
    from models import SchemaMigrationGymAction, SchemaMigrationGymObservation


# =====================================================================
#  TASKS — multi-step dependency chains (6/8/10 optimal steps)
# =====================================================================

TASKS = [
    # ------------------------------------------------------------------
    # TASK 1: "migrate" — 6 optimal steps
    #
    # Dependency DAG:
    #   1. RENAME(tmp_users → users)       ← must be first
    #      ├── 2. SET_NN(users.id)
    #      │   └── 3. SET_PK(users.id)     ← PK requires NN (chain len 3)
    #      ├── 4. SET_NN(users.name)
    #      ├── 5. DROP_COL(users.legacy_flag)
    #      └── 6. ADD_COL(users.email)
    # ------------------------------------------------------------------
    {
        "name": "migrate",
        "start": {"tables": {
            "tmp_users": {"columns": {
                "id":          {"type": "INTEGER", "primary_key": False, "not_null": False},
                "name":        {"type": "TEXT",    "primary_key": False, "not_null": False},
                "legacy_flag": {"type": "TEXT",    "primary_key": False, "not_null": False},
            }},
        }},
        "target": {"tables": {
            "users": {"columns": {
                "id":    {"type": "INTEGER", "primary_key": True,  "not_null": True},
                "name":  {"type": "TEXT",    "primary_key": False, "not_null": True},
                "email": {"type": "TEXT",    "primary_key": False, "not_null": False},
            }},
        }},
    },
    # ------------------------------------------------------------------
    # TASK 2: "restructure" — 8 optimal steps
    #
    # Dependency DAG:
    #   1. RENAME(tmp_accounts → users)    ← must be first for users.*
    #      ├── 2. SET_NN(users.id)
    #      │   └── 3. SET_PK(users.id)    ← chain: 1→2→3
    #      ├── 4. SET_NN(users.name)
    #      ├── 5. DROP_COL(users.old_notes)
    #      ├── 6. ADD_COL(users.age)
    #      │   └── 7. SET_NN(users.age)   ← chain: 1→6→7
    #   8. DROP_TABLE(junk_logs)           ← independent
    # ------------------------------------------------------------------
    {
        "name": "restructure",
        "start": {"tables": {
            "tmp_accounts": {"columns": {
                "id":        {"type": "INTEGER", "primary_key": False, "not_null": False},
                "name":      {"type": "TEXT",    "primary_key": False, "not_null": False},
                "old_notes": {"type": "TEXT",    "primary_key": False, "not_null": False},
            }},
            "junk_logs": {"columns": {
                "x": {"type": "INTEGER", "primary_key": False, "not_null": False},
                "y": {"type": "INTEGER", "primary_key": False, "not_null": False},
            }},
        }},
        "target": {"tables": {
            "users": {"columns": {
                "id":   {"type": "INTEGER", "primary_key": True,  "not_null": True},
                "name": {"type": "TEXT",    "primary_key": False, "not_null": True},
                "age":  {"type": "INTEGER", "primary_key": False, "not_null": True},
            }},
        }},
    },
    # ------------------------------------------------------------------
    # TASK 3: "full_migration" — 10 optimal steps
    #
    # Dependency DAG:
    #   1. RENAME(tmp_users → users)       ← must be first for users.*
    #      ├── 2. SET_NN(users.uid)
    #      │   └── 3. SET_PK(users.uid)   ← chain: 1→2→3
    #      ├── 4. SET_NN(users.uname)
    #      ├── 5. DROP_COL(users.legacy)
    #      ├── 6. ADD_COL(users.email)
    #      ├── 7. ADD_COL(users.age)
    #      │   └── 8. SET_NN(users.age)   ← chain: 1→7→8
    #   9.  DROP_TABLE(old_cache)          ← independent
    #   10. DROP_COL(orders.tmp_note)      ← independent
    # ------------------------------------------------------------------
    {
        "name": "full_migration",
        "start": {"tables": {
            "tmp_users": {"columns": {
                "uid":    {"type": "INTEGER", "primary_key": False, "not_null": False},
                "uname":  {"type": "TEXT",    "primary_key": False, "not_null": False},
                "legacy": {"type": "TEXT",    "primary_key": False, "not_null": False},
            }},
            "old_cache": {"columns": {
                "a": {"type": "INTEGER", "primary_key": False, "not_null": False},
                "b": {"type": "INTEGER", "primary_key": False, "not_null": False},
            }},
            "orders": {"columns": {
                "oid":      {"type": "INTEGER", "primary_key": True,  "not_null": True},
                "buyer_id": {"type": "INTEGER", "primary_key": False, "not_null": True},
                "tmp_note": {"type": "TEXT",    "primary_key": False, "not_null": False},
            }},
        }},
        "target": {"tables": {
            "users": {"columns": {
                "uid":   {"type": "INTEGER", "primary_key": True,  "not_null": True},
                "uname": {"type": "TEXT",    "primary_key": False, "not_null": True},
                "email": {"type": "TEXT",    "primary_key": False, "not_null": False},
                "age":   {"type": "INTEGER", "primary_key": False, "not_null": True},
            }},
            "orders": {"columns": {
                "oid":      {"type": "INTEGER", "primary_key": True,  "not_null": True},
                "buyer_id": {"type": "INTEGER", "primary_key": False, "not_null": True},
            }},
        }},
    },
    # ------------------------------------------------------------------
    # TASK 4: "trap_migration" — 14 optimal steps
    #
    # TRAP: temp_holding has 0 column overlap with the "metrics" target.
    #       Heuristic's rename logic requires shared_cols > 0 (line 36),
    #       so it DROPS temp_holding instead of renaming it.
    #       Since there is no CREATE_TABLE action, "metrics" can never
    #       be created. Task unsolvable for the heuristic.
    #
    #       A smart agent renames temp_holding -> metrics (valid even
    #       with 0 overlap), then drops old cols and adds target cols.
    #
    # Optimal path:
    #   1. RENAME(raw_data -> accounts)
    #   2. RENAME(raw_logs -> audit_log)
    #   3. RENAME(temp_holding -> metrics)   <- heuristic skips this!
    #   4. DROP_COL(accounts.old_ref)
    #   5. DROP_COL(metrics.ts)
    #   6. DROP_COL(metrics.batch)
    #   7. ADD_COL(metrics.metric_id, INTEGER)
    #   8. ADD_COL(metrics.value, INTEGER)
    #   9. SET_NN(accounts.acc_id)
    #  10. SET_NN(accounts.acc_name)
    #  11. SET_NN(accounts.acc_type)
    #  12. SET_NN(audit_log.log_id) + SET_NN(audit_log.msg) + SET_NN(audit_log.level)
    #  13-14. SET_PK(accounts.acc_id) + remaining SET_NN + SET_PK(audit_log.log_id)
    #         + SET_NN/PK on metrics
    # ------------------------------------------------------------------
    {
        "name": "trap_migration",
        "start": {"tables": {
            "raw_data": {"columns": {
                "acc_id":   {"type": "INTEGER", "primary_key": False, "not_null": False},
                "acc_name": {"type": "TEXT",    "primary_key": False, "not_null": False},
                "acc_type": {"type": "TEXT",    "primary_key": False, "not_null": False},
                "old_ref":  {"type": "TEXT",    "primary_key": False, "not_null": False},
            }},
            "raw_logs": {"columns": {
                "log_id": {"type": "INTEGER", "primary_key": False, "not_null": False},
                "msg":    {"type": "TEXT",    "primary_key": False, "not_null": False},
                "level":  {"type": "TEXT",    "primary_key": False, "not_null": False},
            }},
            "temp_holding": {"columns": {
                "ts":    {"type": "INTEGER", "primary_key": False, "not_null": False},
                "batch": {"type": "TEXT",    "primary_key": False, "not_null": False},
            }},
        }},
        "target": {"tables": {
            "accounts": {"columns": {
                "acc_id":   {"type": "INTEGER", "primary_key": True,  "not_null": True},
                "acc_name": {"type": "TEXT",    "primary_key": False, "not_null": True},
                "acc_type": {"type": "TEXT",    "primary_key": False, "not_null": True},
            }},
            "audit_log": {"columns": {
                "log_id": {"type": "INTEGER", "primary_key": True,  "not_null": True},
                "msg":    {"type": "TEXT",    "primary_key": False, "not_null": True},
                "level":  {"type": "TEXT",    "primary_key": False, "not_null": True},
            }},
            "metrics": {"columns": {
                "metric_id": {"type": "INTEGER", "primary_key": True, "not_null": True},
                "value":     {"type": "INTEGER", "primary_key": False, "not_null": True},
            }},
        }},
    },
    # ------------------------------------------------------------------
    # TASK 5: "constraint_trap" — 20 optimal steps
    #
    # DUAL TRAP:
    #   A) Wrong column names: email_wrong, amount_wrong
    #      must be DROPPED and re-created as email, amount.
    #      Heuristic handles this correctly (Phase 3+4).
    #
    #   B) junk_table has 0 column overlap with "audit" target.
    #      Heuristic requires shared_cols > 0 to rename,
    #      so it DROPS junk_table instead of renaming it.
    #      "audit" table never gets created. Unsolvable for heuristic.
    #
    #   A smart agent renames junk_table -> audit (valid even
    #   with 0 overlap), drops {x,y}, adds {event_id, detail}.
    #
    # Optimal path:
    #   1. RENAME(tmp_users -> users)
    #   2. RENAME(tmp_orders -> orders)
    #   3. RENAME(junk_table -> audit)    <- heuristic skips this!
    #   4. DROP_COL(users.email_wrong)
    #   5. DROP_COL(users.temp_flag)
    #   6. DROP_COL(orders.amount_wrong)
    #   7. DROP_COL(orders.temp_note)
    #   8. DROP_COL(audit.x)
    #   9. DROP_COL(audit.y)
    #  10. ADD_COL(users.email, TEXT)
    #  11. ADD_COL(orders.amount, INTEGER)
    #  12. ADD_COL(audit.event_id, INTEGER)
    #  13. ADD_COL(audit.detail, TEXT)
    #  14. SET_NN(users.id)
    #  15. SET_NN(users.name)
    #  16. SET_NN(orders.order_id)
    #  17. SET_NN(orders.user_id)
    #  18. SET_NN(audit.event_id)
    #  19. SET_PK(users.id)
    #  20. SET_PK(orders.order_id)
    #  21. SET_PK(audit.event_id)
    # ------------------------------------------------------------------
    {
        "name": "constraint_trap",
        "start": {"tables": {
            "tmp_users": {"columns": {
                "id":          {"type": "INTEGER", "primary_key": False, "not_null": False},
                "name":        {"type": "TEXT",    "primary_key": False, "not_null": False},
                "email_wrong": {"type": "INTEGER", "primary_key": False, "not_null": False},
                "temp_flag":   {"type": "BOOLEAN", "primary_key": False, "not_null": False},
            }},
            "tmp_orders": {"columns": {
                "order_id":     {"type": "INTEGER", "primary_key": False, "not_null": False},
                "user_id":      {"type": "INTEGER", "primary_key": False, "not_null": False},
                "amount_wrong": {"type": "TEXT",    "primary_key": False, "not_null": False},
                "temp_note":    {"type": "TEXT",    "primary_key": False, "not_null": False},
            }},
            "junk_table": {"columns": {
                "x": {"type": "INTEGER", "primary_key": False, "not_null": False},
                "y": {"type": "INTEGER", "primary_key": False, "not_null": False},
            }},
        }},
        "target": {"tables": {
            "users": {"columns": {
                "id":    {"type": "INTEGER", "primary_key": True,  "not_null": True},
                "name":  {"type": "TEXT",    "primary_key": False, "not_null": True},
                "email": {"type": "TEXT",    "primary_key": False, "not_null": False},
            }},
            "orders": {"columns": {
                "order_id": {"type": "INTEGER", "primary_key": True,  "not_null": True},
                "user_id":  {"type": "INTEGER", "primary_key": False, "not_null": True},
                "amount":   {"type": "INTEGER", "primary_key": False, "not_null": False},
            }},
            "audit": {"columns": {
                "event_id": {"type": "INTEGER", "primary_key": True,  "not_null": True},
                "detail":   {"type": "TEXT",    "primary_key": False, "not_null": False},
            }},
        }},
    },
    # ------------------------------------------------------------------
    # TASK 6: "constraint_dependency_trap" — 9 optimal steps
    #
    # TRAP (NEW mechanism): wrong constraints that can't be un-set.
    #   users.email starts with primary_key=True, not_null=True
    #   but target says primary_key=False, not_null=True.
    #   There is NO UNSET_PRIMARY_KEY action.
    #   Heuristic only SETS constraints (Phase 5→6), never UNSETS.
    #   So email.primary_key stays True → solved=False.
    #
    #   A smart agent must: DROP_COLUMN(email) + ADD_COLUMN(email,TEXT)
    #   + SET_NN(email) to clear the wrong PK flag.
    #
    # Optimal path:
    #   1. SET_NN(users.id)
    #   2. DROP_COL(users.email)         <- clear wrong PK
    #   3. ADD_COL(users.email, TEXT)     <- re-add clean
    #   4. SET_NN(users.email)
    #   5. SET_NN(users.age)
    #   6. SET_PK(users.id)
    #   7. SET_NN(orders.order_id)
    #   8. SET_PK(orders.order_id)
    #   9. SET_NN(orders.user_id)
    # ------------------------------------------------------------------
    {
        "name": "constraint_dependency_trap",
        "start": {"tables": {
            "users": {"columns": {
                "id":    {"type": "INTEGER", "primary_key": False, "not_null": False},
                "email": {"type": "TEXT",    "primary_key": True,  "not_null": True},
                "age":   {"type": "INTEGER", "primary_key": False, "not_null": False},
            }},
            "orders": {"columns": {
                "order_id": {"type": "INTEGER", "primary_key": False, "not_null": False},
                "user_id":  {"type": "INTEGER", "primary_key": False, "not_null": False},
            }},
        }},
        "target": {"tables": {
            "users": {"columns": {
                "id":    {"type": "INTEGER", "primary_key": True,  "not_null": True},
                "email": {"type": "TEXT",    "primary_key": False, "not_null": True},
                "age":   {"type": "INTEGER", "primary_key": False, "not_null": True},
            }},
            "orders": {"columns": {
                "order_id": {"type": "INTEGER", "primary_key": True,  "not_null": True},
                "user_id":  {"type": "INTEGER", "primary_key": False, "not_null": True},
            }},
        }},
    },
]

# =====================================================================
#  CONSTANTS
# =====================================================================

ALLOWED_TYPES = {"INTEGER", "TEXT", "BOOLEAN"}

INVALID_ACTION_REWARD = -0.1
STEP_PENALTY = -0.01
REVISIT_PENALTY = -0.05
COMPLETION_BONUS = 1.0


# =====================================================================
#  ENVIRONMENT
# =====================================================================

class SchemaMigrationGymEnvironment(Environment):

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.current_state: dict = {"tables": {}}
        self.target_state: dict = {"tables": {}}
        self.current_task: dict | None = None
        self.max_steps: int = 30
        # Reward tracking
        self._prev_similarity: float = 0.0
        self._visited_states: set = set()

    # ======================== RESET ========================

    def reset(self) -> SchemaMigrationGymObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)

        self.current_task = random.choice(TASKS)
        self.current_state = copy.deepcopy(self.current_task["start"])
        self.target_state = copy.deepcopy(self.current_task["target"])

        # Initialize reward tracking
        self._prev_similarity = self._compute_similarity()
        self._visited_states = {self._hash_state(self.current_state)}

        return self._build_observation(success=True, error=None, reward=0.0, done=False)

    # ======================== STEP ========================

    def step(self, action: SchemaMigrationGymAction) -> SchemaMigrationGymObservation:
        self._state.step_count += 1

        # Work on a deep copy — commit only if action succeeds
        candidate = copy.deepcopy(self.current_state)
        success, error = self._apply_action(candidate, action)

        if success:
            self.current_state = candidate

            # Delta-based reward
            new_similarity = self._compute_similarity()
            delta = new_similarity - self._prev_similarity
            self._prev_similarity = new_similarity

            # Revisit penalty
            state_hash = self._hash_state(self.current_state)
            revisit = REVISIT_PENALTY if state_hash in self._visited_states else 0.0
            self._visited_states.add(state_hash)

            reward = delta + STEP_PENALTY + revisit
        else:
            reward = INVALID_ACTION_REWARD

        done = self._is_done()

        # Completion bonus
        if done and self.current_state == self.target_state:
            reward += COMPLETION_BONUS

        return self._build_observation(success, error, reward, done)

    @property
    def state(self) -> State:
        return self._state

    # ======================== ACTION DISPATCH ========================

    def _apply_action(self, state: dict, action: SchemaMigrationGymAction):
        """Apply *action* to *state* (mutated in-place). Returns (success, error)."""

        dispatch = {
            "ADD_COLUMN":      self._action_add_column,
            "DROP_COLUMN":     self._action_drop_column,
            "DROP_TABLE":      self._action_drop_table,
            "RENAME_TABLE":    self._action_rename_table,
            "SET_PRIMARY_KEY": self._action_set_primary_key,
            "SET_NOT_NULL":    self._action_set_not_null,
        }

        handler = dispatch.get(action.action_type)
        if handler is None:
            return False, f"Unknown action_type '{action.action_type}'"

        return handler(state, action)

    # -------------------- individual handlers --------------------

    def _action_add_column(self, state, action):
        if not action.table or not action.column:
            return False, "ADD_COLUMN requires 'table' and 'column'"
        tables = state["tables"]
        if action.table not in tables:
            return False, f"Table '{action.table}' does not exist"
        if action.column in tables[action.table]["columns"]:
            return False, f"Column '{action.column}' already exists in '{action.table}'"

        # Determine column type
        col_type = action.column_type
        if col_type is None:
            # Infer from target schema
            tgt_cols = self.target_state.get("tables", {}).get(
                action.table, {}).get("columns", {})
            if action.column in tgt_cols:
                col_type = tgt_cols[action.column].get("type", "INTEGER")
            else:
                return False, "ADD_COLUMN requires 'column_type' (cannot infer from target)"
        if col_type not in ALLOWED_TYPES:
            return False, f"Invalid column_type '{col_type}'. Must be one of {sorted(ALLOWED_TYPES)}"

        tables[action.table]["columns"][action.column] = {
            "type": col_type,
            "primary_key": False,
            "not_null": False,
        }
        return True, None

    @staticmethod
    def _action_drop_column(state, action):
        if not action.table or not action.column:
            return False, "DROP_COLUMN requires 'table' and 'column'"
        tables = state["tables"]
        if action.table not in tables:
            return False, f"Table '{action.table}' does not exist"
        if action.column not in tables[action.table]["columns"]:
            return False, f"Column '{action.column}' does not exist in '{action.table}'"
        del tables[action.table]["columns"][action.column]
        return True, None

    @staticmethod
    def _action_drop_table(state, action):
        if not action.table:
            return False, "DROP_TABLE requires 'table'"
        tables = state["tables"]
        if action.table not in tables:
            return False, f"Table '{action.table}' does not exist"
        del tables[action.table]
        return True, None

    @staticmethod
    def _action_rename_table(state, action):
        if not action.table or not action.new_table_name:
            return False, "RENAME_TABLE requires 'table' and 'new_table_name'"
        tables = state["tables"]
        if action.table not in tables:
            return False, f"Table '{action.table}' does not exist"
        if action.new_table_name in tables:
            return False, f"Table '{action.new_table_name}' already exists"
        tables[action.new_table_name] = tables.pop(action.table)
        return True, None

    @staticmethod
    def _action_set_primary_key(state, action):
        if not action.table or not action.column:
            return False, "SET_PRIMARY_KEY requires 'table' and 'column'"
        tables = state["tables"]
        if action.table not in tables:
            return False, f"Table '{action.table}' does not exist"
        columns = tables[action.table]["columns"]
        if action.column not in columns:
            return False, f"Column '{action.column}' does not exist in '{action.table}'"
        if not columns[action.column]["not_null"]:
            return False, f"Cannot SET_PRIMARY_KEY: column '{action.column}' must be NOT NULL first"
        columns[action.column]["primary_key"] = True
        return True, None

    @staticmethod
    def _action_set_not_null(state, action):
        if not action.table or not action.column:
            return False, "SET_NOT_NULL requires 'table' and 'column'"
        tables = state["tables"]
        if action.table not in tables:
            return False, f"Table '{action.table}' does not exist"
        columns = tables[action.table]["columns"]
        if action.column not in columns:
            return False, f"Column '{action.column}' does not exist in '{action.table}'"
        columns[action.column]["not_null"] = True
        return True, None

    # ======================== SIMILARITY ========================

    def _compute_similarity(self) -> float:
        """
        Structural similarity between current_state and target_state.
        Returns a score in [0.0, 1.0].

        Scoring:
          +1  per target table that exists in current
          +1  per target column that exists in current (within matching table)
          +1  per matching constraint (primary_key, not_null) on a target column

        Penalties (added to denominator only):
          +1  per extra table in current not in target
          +1  per extra column in current not in target
        """
        cur_tables = self.current_state["tables"]
        tgt_tables = self.target_state["tables"]

        earned = 0
        total = 0

        for tname, tdata in tgt_tables.items():
            total += 1
            if tname in cur_tables:
                earned += 1
                for cname, cprops in tdata["columns"].items():
                    total += 1
                    if cname in cur_tables[tname]["columns"]:
                        earned += 1
                        cur_col = cur_tables[tname]["columns"][cname]
                        for key in ("type", "primary_key", "not_null"):
                            total += 1
                            if cur_col.get(key) == cprops.get(key):
                                earned += 1

        for tname in cur_tables:
            if tname not in tgt_tables:
                total += 1
            else:
                for cname in cur_tables[tname]["columns"]:
                    if cname not in tgt_tables[tname]["columns"]:
                        total += 1

        return earned / total if total > 0 else 1.0

    # ======================== DONE ========================

    def _is_done(self) -> bool:
        if self._state.step_count >= self.max_steps:
            return True
        return self.current_state == self.target_state

    # ======================== OBSERVATION ========================

    def _build_observation(self, success, error, reward, done):
        return SchemaMigrationGymObservation(
            current_schema=self._render_schema(self.current_state),
            target_schema=self._render_schema(self.target_state),
            relationships=[],
            step_count=self._state.step_count,
            max_steps=self.max_steps,
            last_action_success=success,
            error_message=error,
            reward=reward,
            done=done,
        )

    # ======================== HELPERS ========================

    @staticmethod
    def _render_schema(state: dict) -> str:
        """Convert structured state dict into a human-readable DDL-like string."""
        lines = []
        for tname, tdata in state["tables"].items():
            col_parts = []
            for cname, cprops in tdata["columns"].items():
                parts = [cname, cprops["type"]]
                if cprops.get("primary_key"):
                    parts.append("PRIMARY KEY")
                if cprops.get("not_null"):
                    parts.append("NOT NULL")
                col_parts.append(" ".join(parts))
            lines.append(f"TABLE {tname} ({', '.join(col_parts)})")
        return "\n".join(sorted(lines))

    @staticmethod
    def _hash_state(state: dict) -> str:
        """Deterministic hash for visited-state tracking."""
        return json.dumps(state, sort_keys=True)

    def get_valid_actions(self) -> list:
        """Returns all structurally valid actions from the current state."""
        actions = []
        tables = self.current_state["tables"]
        tgt_tables = self.target_state["tables"]
        cur_names = set(tables.keys())
        tgt_names = set(tgt_tables.keys())

        for tname, tdata in tables.items():
            cols = list(tdata["columns"].keys())

            # Constraint actions on existing columns
            for col in cols:
                actions.append(SchemaMigrationGymAction(
                    action_type="SET_PRIMARY_KEY", table=tname, column=col))
                actions.append(SchemaMigrationGymAction(
                    action_type="SET_NOT_NULL", table=tname, column=col))
                actions.append(SchemaMigrationGymAction(
                    action_type="DROP_COLUMN", table=tname, column=col))

            # DROP_TABLE
            actions.append(SchemaMigrationGymAction(
                action_type="DROP_TABLE", table=tname))

            # RENAME_TABLE — only to target names not yet present
            for target_name in tgt_names - cur_names:
                actions.append(SchemaMigrationGymAction(
                    action_type="RENAME_TABLE", table=tname,
                    new_table_name=target_name))

            # ADD_COLUMN for target columns missing in current
            target_cols = tgt_tables.get(tname, {}).get("columns", {})
            for tcol, tcprops in target_cols.items():
                if tcol not in cols:
                    actions.append(SchemaMigrationGymAction(
                        action_type="ADD_COLUMN", table=tname, column=tcol,
                        column_type=tcprops.get("type", "INTEGER")))

        return actions


# =====================================================================
#  STANDALONE SMOKE TEST
# =====================================================================

if __name__ == "__main__":
    env = SchemaMigrationGymEnvironment()
    obs = env.reset()

    print("=== RESET ===")
    print(f"Task: {env.current_task['name']}")
    print(f"Current:\n  {obs.current_schema}")
    print(f"Target:\n  {obs.target_schema}")
    print(f"Initial similarity: {env._prev_similarity:.4f}")
    print(f"Max steps: {env.max_steps}")
    print()

    demo_actions = [
        SchemaMigrationGymAction(action_type="SET_PRIMARY_KEY", table="users", column="id"),
        SchemaMigrationGymAction(action_type="SET_NOT_NULL",    table="users", column="name"),
        SchemaMigrationGymAction(action_type="ADD_COLUMN",      table="users", column="age"),
        SchemaMigrationGymAction(action_type="DROP_COLUMN",     table="users", column="age"),
        SchemaMigrationGymAction(action_type="ADD_COLUMN",      table="nonexistent", column="x"),
    ]

    cumulative = 0.0
    for i, action in enumerate(demo_actions, 1):
        obs = env.step(action)
        cumulative += obs.reward
        status = "OK" if obs.last_action_success else "FAIL"
        print(f"Step {i} [{action.action_type}] {status}")
        print(f"  Reward: {obs.reward:+.4f}  Cumulative: {cumulative:+.4f}  Done: {obs.done}")
        if obs.error_message:
            print(f"  Error: {obs.error_message}")
        print()

    print(f"Valid actions available: {len(env.get_valid_actions())}")