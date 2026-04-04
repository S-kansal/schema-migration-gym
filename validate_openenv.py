"""OpenEnv spec compliance validation — programmatic substitute for `openenv validate`."""
import sys
import os
import json

sys.path.insert(0, ".")

passed = 0
failed = 0

def check(name, cond):
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        print(f"  FAIL: {name}")

print("=" * 60)
print("OPENENV SPEC VALIDATION")
print("=" * 60)

# 1. openenv.yaml
import yaml
with open("openenv.yaml") as f:
    manifest = yaml.safe_load(f)
check("openenv.yaml exists", manifest is not None)
check("spec_version = 1", manifest.get("spec_version") == 1)
check("name = schema_migration_gym", manifest.get("name") == "schema_migration_gym")
check("runtime = fastapi", manifest.get("runtime") == "fastapi")
check("app = server.app:app", manifest.get("app") == "server.app:app")
check("port = 8000", manifest.get("port") == 8000)

# 2. Pydantic models
from pydantic import BaseModel
from models import SchemaMigrationGymAction, SchemaMigrationGymObservation
check("Action is Pydantic", issubclass(SchemaMigrationGymAction, BaseModel))
check("Observation is Pydantic", issubclass(SchemaMigrationGymObservation, BaseModel))

# 3. Action fields
a = SchemaMigrationGymAction(action_type="ADD_COLUMN", table="t", column="c")
check("Action.action_type", a.action_type == "ADD_COLUMN")
check("Action.table", a.table == "t")
check("Action.column", a.column == "c")
a2 = SchemaMigrationGymAction(action_type="RENAME_TABLE", table="t", new_table_name="t2")
check("Action.new_table_name", a2.new_table_name == "t2")

# 4. Observation fields
o = SchemaMigrationGymObservation()
check("Obs.current_schema", hasattr(o, "current_schema"))
check("Obs.target_schema", hasattr(o, "target_schema"))
check("Obs.step_count", hasattr(o, "step_count"))
check("Obs.max_steps = 30", o.max_steps == 30)
check("Obs.last_action_success", hasattr(o, "last_action_success"))
check("Obs.error_message", hasattr(o, "error_message"))

# 5. Environment interface
from server.schema_migration_gym_environment import SchemaMigrationGymEnvironment, TASKS
from openenv.core.env_server.interfaces import Environment
check("Env inherits Environment", issubclass(SchemaMigrationGymEnvironment, Environment))
env = SchemaMigrationGymEnvironment()
check("Has reset()", hasattr(env, "reset"))
check("Has step()", hasattr(env, "step"))
check("Has state property", hasattr(type(env), "state"))

# 6. reset()
obs = env.reset()
check("reset returns Observation", isinstance(obs, SchemaMigrationGymObservation))
check("reset reward=0", obs.reward == 0.0)
check("reset done=False", obs.done == False)
check("reset step_count=0", obs.step_count == 0)

# 7. step()
tables = list(env.current_state["tables"].keys())
cols = list(env.current_state["tables"][tables[0]]["columns"].keys())
action = SchemaMigrationGymAction(action_type="SET_NOT_NULL", table=tables[0], column=cols[0])
obs2 = env.step(action)
check("step returns Observation", isinstance(obs2, SchemaMigrationGymObservation))
check("step has reward (float)", isinstance(obs2.reward, float))
check("step has done (bool)", isinstance(obs2.done, bool))

# 8. Tasks
check("3+ tasks defined", len(TASKS) >= 3)
task_names = [t["name"] for t in TASKS]
check("Task: migrate", "migrate" in task_names)
check("Task: restructure", "restructure" in task_names)
check("Task: full_migration", "full_migration" in task_names)
for t in TASKS:
    check(f"Task '{t['name']}' has start", "start" in t)
    check(f"Task '{t['name']}' has target", "target" in t)

# 9. App routes
from server.app import app
check("FastAPI app loads", app is not None)
routes = [getattr(r, "path", "") for r in app.routes]
for ep in ["/reset", "/step", "/state", "/health", "/tasks", "/baseline", "/grader"]:
    check(f"Route {ep} exists", ep in routes)

# 10. Files
check("inference.py in root", os.path.isfile("inference.py"))
check("openenv.yaml in root", os.path.isfile("openenv.yaml"))
check("pyproject.toml in root", os.path.isfile("pyproject.toml"))
check("Dockerfile exists", os.path.isfile("server/Dockerfile"))
check("models.py exists", os.path.isfile("models.py"))

# 11. Stability (20 random actions)
import copy, random
random.seed(42)
env2 = SchemaMigrationGymEnvironment()
env2._state.step_count = 0
env2.current_task = TASKS[0]
env2.current_state = copy.deepcopy(TASKS[0]["start"])
env2.target_state = copy.deepcopy(TASKS[0]["target"])
env2._prev_similarity = env2._compute_similarity()
env2._visited_states = {env2._hash_state(env2.current_state)}

crash = False
for i in range(20):
    at = random.choice(["ADD_COLUMN","DROP_COLUMN","DROP_TABLE","RENAME_TABLE","SET_NOT_NULL","SET_PRIMARY_KEY","INVALID_TYPE"])
    try:
        a = SchemaMigrationGymAction(action_type=at, table="tmp_users", column="id", new_table_name="foo")
        obs = env2.step(a)
    except Exception as e:
        crash = True
        print(f"  CRASH at step {i}: {e}")
        break
check("20 random actions: no crash", not crash)

print()
print("=" * 60)
print(f"RESULTS: {passed} passed, {failed} failed")
print("=" * 60)
if failed > 0:
    print("SOME CHECKS FAILED")
    sys.exit(1)
else:
    print("ALL CHECKS PASSED")
