---
title: Schema Migration Gym Environment Server
emoji: 🗄️
colorFrom: green
colorTo: purple
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Schema Migration Gym

A reinforcement learning environment where agents must plan and execute database schema migrations — exposing failure modes in greedy planning systems through adversarial task design.

---

## Problem: Why Schema Migration Is Hard

Database schemas evolve constantly in production systems. Columns are added, tables renamed, constraints tightened. The process of transforming one schema into another — schema migration — is a routine but high-stakes operation for DBAs, backend engineers, and DevOps teams.

What makes it difficult:

- **Dependency ordering.** A primary key cannot be set on a column that isn't marked NOT NULL. Constraints must be applied in the correct sequence.
- **Irreversibility.** Dropping a table in production is permanent. There is no undo. Agents that act greedily without planning can make tasks unsolvable.
- **Multiple valid paths.** Several action orderings may reach the target, but they differ in efficiency. The optimal path requires lookahead.
- **Structural ambiguity.** When table names change and columns are added or removed simultaneously, identifying which source table maps to which target requires reasoning beyond surface-level matching.

This environment distills these challenges into a deterministic, fully observable RL problem with structured actions, constraint enforcement, and adversarial tasks designed to defeat naive strategies.

---

## Environment Design

### State Representation

The state is a structured Python dictionary representing a database schema:

```python
{
    "tables": {
        "users": {
            "columns": {
                "id":   {"type": "INTEGER", "primary_key": True,  "not_null": True},
                "name": {"type": "TEXT",    "primary_key": False, "not_null": True},
            }
        }
    }
}
```

Both the current schema and the target schema are fully visible to the agent at every step. All transitions are deterministic — there is no stochasticity.

### Action Space

Six parametric action types:

| Action | Parameters | Effect |
|--------|-----------|--------|
| `ADD_COLUMN` | table, column, column_type | Adds a new column (INTEGER, TEXT, or BOOLEAN) |
| `DROP_COLUMN` | table, column | Removes an existing column |
| `DROP_TABLE` | table | Removes an entire table (irreversible) |
| `RENAME_TABLE` | table, new_table_name | Renames a table |
| `SET_NOT_NULL` | table, column | Sets the NOT NULL constraint on a column |
| `SET_PRIMARY_KEY` | table, column | Sets the primary key (requires NOT NULL first) |

The constraint `SET_PRIMARY_KEY requires NOT NULL` creates mandatory action chains: an agent must call `SET_NOT_NULL` before `SET_PRIMARY_KEY` on any column. This mirrors real database behavior in PostgreSQL and MySQL, where primary keys implicitly require NOT NULL.

### Transition Logic

- State is deep-copied before each action. Changes are committed only on success.
- Invalid actions (wrong table name, missing column, type violation) are rejected with descriptive error messages. The state remains unchanged.
- All transitions are deterministic and reproducible.

### Observation Space

Each observation returned by `step()` and `reset()` includes:

| Field | Type | Description |
|-------|------|-------------|
| `current_schema` | string | DDL-style rendering of the current state |
| `target_schema` | string | DDL-style rendering of the goal state |
| `step_count` | int | Current step number |
| `max_steps` | int | Episode length limit (30) |
| `last_action_success` | bool | Whether the previous action was valid |
| `error_message` | string | Explanation if the action failed |
| `reward` | float | Step reward signal |
| `done` | bool | Whether the episode has ended |

---

## Tasks and Difficulty Progression

6 tasks (3 standard + 3 adversarial), ranging from 6 to 21 optimal steps.

### Standard Tasks

**`migrate`** (easy, 6 steps) — Single table: rename, drop a legacy column, add a new column, set constraints.

```
START:  tmp_users(id, name, legacy_flag)
TARGET: users(id PK+NN, name NN, email)
```

**`restructure`** (medium, 8 steps) — Two tables: rename one, drop the other, add a column with constraints. Introduces multi-table coordination.

```
START:  tmp_accounts(id, name, old_notes) + junk_logs(x, y)
TARGET: users(id PK+NN, name NN, age NN)
```

**`full_migration`** (hard, 10 steps) — Three tables, cross-table drop operations, multiple dependency chains running in parallel.

```
START:  tmp_users(uid, uname, legacy) + old_cache(a, b) + orders(oid PK+NN, buyer_id NN, tmp_note)
TARGET: users(uid PK+NN, uname NN, email, age NN) + orders(oid PK+NN, buyer_id NN)
```

A greedy heuristic agent solves all three standard tasks optimally.

### Adversarial Tasks

These tasks are specifically designed to break heuristic and greedy planning agents. Each exploits a distinct failure mode:

**`trap_migration`** (expert, ~14 steps) — Zero-overlap rename trap.
`temp_holding` shares zero columns with the target table `metrics`. A greedy heuristic matches tables by column overlap, finds no match, and drops the table instead of renaming it. Since there is no `CREATE_TABLE` action, the target becomes permanently unreachable. A reasoning agent recognizes that renaming is valid regardless of column overlap, then drops old columns and adds the correct ones.

**`constraint_trap`** (expert, ~21 steps) — Structural mismatch plus zero-overlap trap.
Combines wrong column names (`email_wrong`, `amount_wrong`) with a second zero-overlap rename (`junk_table` to `audit`). Requires both column replacement and the non-obvious rename. The heuristic handles the column fix but fails on the rename — same mechanism as above, different context.

**`constraint_dependency_trap`** (hard, 9 steps) — Constraint repair trap.
`users.email` starts with `primary_key=True`, but the target requires `primary_key=False`. There is no `UNSET_PRIMARY_KEY` action. The only solution is to drop the column entirely and re-add it with the correct constraints. A heuristic agent only sets constraints forward — it never considers dropping a column to fix an incorrect one. This task introduces an entirely different failure mode from the rename traps.

These tasks ensure that only agents capable of multi-step reasoning can achieve full scores.

---

## Reward Design

The reward function uses a **delta-based** signal derived from structural similarity between the current and target schemas:

| Component | Value | Purpose |
|-----------|-------|---------|
| Progress delta | `sim(t) - sim(t-1)` | Positive reward for actions that move closer to the target |
| Step penalty | `-0.01` | Discourages inefficiency |
| Revisit penalty | `-0.05` | Discourages returning to previously visited states |
| Completion bonus | `+1.0` | Strong terminal reward for solving the task |
| Invalid action | `-0.1` | Penalizes illegal operations |

Similarity is computed as the structural overlap between the current and target schemas: matching tables, columns, types, and constraint flags, normalized to [0.0, 1.0].

The reward function provides continuous feedback, enabling learning across partial progress rather than relying on sparse success signals. Every valid action that moves the schema closer to the target produces a positive delta; every misstep produces a clear negative signal.

---

## Baseline Performance

The included heuristic agent follows a fixed priority chain: `RENAME → DROP_TABLE → DROP_COLUMN → ADD_COLUMN → SET_NOT_NULL → SET_PRIMARY_KEY`. It matches tables by column overlap for renames.

```json
{
  "migrate": 1.0,
  "restructure": 1.0,
  "full_migration": 1.0,
  "trap_migration": 0.4815,
  "constraint_trap": 0.4815,
  "constraint_dependency_trap": 0.4773
}
```

The heuristic achieves perfect scores on all standard tasks but fails on every adversarial task — scoring below 0.50 on each. This demonstrates a clear performance ceiling that cannot be crossed without planning beyond greedy action selection.

---

## Key Insight

This environment is designed not just to train agents, but to expose failure modes in greedy and heuristic-based planning systems.

The adversarial tasks create a measurable gap between agents that follow fixed rules and agents that reason about state. Specifically:

- **Greedy agents** achieve ~0.48 on adversarial tasks (partial progress only).
- **Reasoning agents** (LLM-based or Q-learning) can achieve 1.0 on all tasks.
- The environment **differentiates agent capability** across three levels: random (~0.0), heuristic (~0.75 average), and reasoning (~1.0).

This makes the environment suitable for benchmarking LLM planning ability, testing RL exploration strategies, and identifying where rule-based systems break down.

---

## Setup and Usage

### Docker

```bash
docker build -t schema-gym -f server/Dockerfile .
docker run -p 8000:8000 schema-gym
```

### API

```bash
# Health check
curl http://localhost:8000/health

# Reset environment
curl -X POST http://localhost:8000/reset

# Execute an action
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action_type": "RENAME_TABLE", "table": "tmp_users", "new_table_name": "users"}}'

# List all tasks
curl http://localhost:8000/tasks

# Run baseline agent
curl -X POST http://localhost:8000/baseline
```

### Inference

```bash
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
export HF_TOKEN=hf_...

python inference.py
```

The inference script uses an OpenAI-compatible client to call the LLM for each step. If the LLM is unavailable (missing credentials), it falls back to the deterministic heuristic agent.

---

## Evaluation

- Graders are deterministic: the same action sequence always produces the same score.
- Scores are bounded to [0.0, 1.0]. A score of 1.0 means the target schema was reached exactly.
- Partial progress is measured via structural similarity: `score = similarity * 0.5` for unsolved tasks.
- All tasks are verified solvable through explicit action sequences in the test suite.

---

## Project Structure

```
schema_migration_gym/
├── openenv.yaml                 ← OpenEnv manifest
├── pyproject.toml               ← Build config + dependencies
├── models.py                    ← Action + Observation Pydantic models
├── client.py                    ← WebSocket/HTTP client
├── inference.py                 ← LLM inference script (HF-compatible)
├── __init__.py                  ← Package exports
├── validate_openenv.py          ← OpenEnv spec validation
├── test_adversarial.py          ← Adversarial solvability + edge case tests
├── verify_upgrade.py            ← Heuristic + reward verification
└── server/
    ├── schema_migration_gym_environment.py  ← Core environment logic
    ├── app.py                   ← FastAPI application
    ├── heuristic.py             ← Baseline heuristic agent
    ├── Dockerfile               ← Multi-stage container build
    └── requirements.txt         ← Server dependencies
```

---

## Compliance

- OpenEnv spec version 1 compliant
- Typed Pydantic models extending `openenv.core.env_server.types.Action` and `Observation`
- Environment extends `openenv.core.env_server.interfaces.Environment`
- Dockerized with multi-stage build and health checks
- Hugging Face Spaces ready (`sdk: docker`, `app_port: 8000`)

---

## License

Copyright (c) Meta Platforms, Inc. All rights reserved.
BSD-style license — see LICENSE file.
