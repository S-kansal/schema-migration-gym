# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Shared heuristic agent for the Schema Migration Gym.

Single source of truth — imported by both server/app.py and inference.py.
"""

try:
    from ..models import SchemaMigrationGymAction
except ImportError:
    from models import SchemaMigrationGymAction


def select_action(current_state: dict, target_state: dict) -> SchemaMigrationGymAction:
    """
    Chain-aware heuristic: RENAME -> DROP_TABLE -> DROP_COL -> ADD_COL -> SET_NN -> SET_PK.

    Processes actions in dependency order so that prerequisite steps
    (e.g. SET_NOT_NULL) complete before dependent ones (e.g. SET_PRIMARY_KEY).
    """
    cur = current_state["tables"]
    tgt = target_state["tables"]

    cur_names, tgt_names = set(cur.keys()), set(tgt.keys())
    missing, extra = tgt_names - cur_names, cur_names - tgt_names

    # Phase 1: RENAME — match extra tables to missing target names by shared columns
    for m in missing:
        best, best_score = None, -1
        for e in extra:
            shared = set(cur[e]["columns"].keys()) & set(tgt[m]["columns"].keys())
            if len(shared) > best_score:
                best_score, best = len(shared), e
        if best and best_score > 0:
            return SchemaMigrationGymAction(
                action_type="RENAME_TABLE", table=best, new_table_name=m)

    # Phase 2: DROP extra tables
    for e in extra:
        return SchemaMigrationGymAction(action_type="DROP_TABLE", table=e)

    # Phase 3: DROP extra columns
    for t in tgt:
        if t not in cur:
            continue
        for c in list(cur[t]["columns"].keys()):
            if c not in tgt[t]["columns"]:
                return SchemaMigrationGymAction(
                    action_type="DROP_COLUMN", table=t, column=c)

    # Phase 4: ADD missing columns (with type from target)
    for t, td in tgt.items():
        if t not in cur:
            continue
        for c, cprops in td["columns"].items():
            if c not in cur[t]["columns"]:
                return SchemaMigrationGymAction(
                    action_type="ADD_COLUMN", table=t, column=c,
                    column_type=cprops.get("type", "INTEGER"))

    # Phase 5: SET_NOT_NULL
    for t, td in tgt.items():
        if t not in cur:
            continue
        for c, cp in td["columns"].items():
            if c not in cur[t]["columns"]:
                continue
            if cp["not_null"] and not cur[t]["columns"][c]["not_null"]:
                return SchemaMigrationGymAction(
                    action_type="SET_NOT_NULL", table=t, column=c)

    # Phase 6: SET_PRIMARY_KEY
    for t, td in tgt.items():
        if t not in cur:
            continue
        for c, cp in td["columns"].items():
            if c not in cur[t]["columns"]:
                continue
            if cp["primary_key"] and not cur[t]["columns"][c]["primary_key"]:
                return SchemaMigrationGymAction(
                    action_type="SET_PRIMARY_KEY", table=t, column=c)

    # Fallback (should not be reached if task is solvable)
    return SchemaMigrationGymAction(
        action_type="SET_NOT_NULL", table="users", column="id")
