# Complex Worlds Hackathon — DeceptionSearch-v0

Find a hidden AWS access key in a simulated dev laptop populated with
LLM-generated decoys, under a 100-action budget. One env, two Searchers
(GPT-5.4, Claude Haiku 4.5), one Deceiver (Gemini 3 Flash), two baselines
(random, exhaustive).

Spec: [`docs/brief.md`](docs/brief.md).

## Setup

```bash
# 1. Create venv and install deps
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Clone the OSS repo we use as the world
git clone --depth 1 https://github.com/getsentry/self-hosted \
  scenarios/compromised_laptop/_source

# 3. Build the base tree (real Sentry self-hosted + synthesised home dir)
python scenarios/build_base_tree.py

# 4. API keys (gitignored, auto-loaded by build_tasks.py and agents/run.py)
cp .env.example .env
$EDITOR .env
```

Required keys (`.env` or shell):

| Var | Used by | Where to get |
|---|---|---|
| `OPENROUTER_API_KEY` | Haiku 4.5 Searcher + Gemini 3 Flash Deceiver | https://openrouter.ai/keys |
| `OPENAI_API_KEY` | GPT-5.4 Searcher (direct) | https://platform.openai.com/api-keys |
| `OPENREWARD_API_KEY` | `orwd` CLI publish | https://openreward.ai/settings |

Native Anthropic / Google AI Studio keys are *alternates* (`ANTHROPIC_API_KEY`,
`GEMINI_API_KEY`) — the harness only falls back to them if the OpenRouter key
is missing.

## Generate task specs

The Deceiver runs once at task-build time (Gemini 3 Flash). Set
`GEMINI_API_KEY` (or `GOOGLE_API_KEY`) and:

```bash
SCENARIO_DIR=scenarios/compromised_laptop \
  python build_tasks.py --split smoke --seeds 0      --out tasks/smoke.json
SCENARIO_DIR=scenarios/compromised_laptop \
  python build_tasks.py --split dev   --seeds 0-2    --out tasks/dev.json
SCENARIO_DIR=scenarios/compromised_laptop \
  python build_tasks.py --split test  --seeds 0-19   --out tasks/test.json
```

For harness development without API access, pass `--mock` to use
path-templated fake honeypots instead.

## Run a Searcher

```bash
# Baselines (no API key needed)
python -m agents.run --agent random      --task tasks/smoke.json -v
python -m agents.run --agent exhaustive  --task tasks/smoke.json -v

# Frontier Searchers (set ANTHROPIC_API_KEY / OPENAI_API_KEY)
python -m agents.run --agent haiku       --task tasks/smoke.json -v
python -m agents.run --agent gpt54       --task tasks/smoke.json -v

# All four:
python -m agents.run --agent all --task tasks/smoke.json
```

Per-session logs land at `runs/<run_id>.json`.

## Analyze a Trace

Use the trace analyzer to inspect one run, derive trace-quality metrics, and
compare it against a baseline on the same task.

```bash
# Analyze one run and auto-pick a baseline from runs/
python scripts/analyze_trace.py \
  --run runs/<run_id>.json \
  --task tasks/smoke_v2.json

# Analyze one run against an explicit baseline
python scripts/analyze_trace.py \
  --run runs/1777125694-gpt-5.4-task-0-v2-184f7c.json \
  --task tasks/smoke_v2.json \
  --baseline-run runs/1777125081-exhaustive-task-0-v2-208c91.json
```

The analyzer emits JSON including:

- `speculative_unlocks_after_key_seen`
- `first_real_key_turn`
- `first_real_vault_turn`
- `candidate_binding_count_over_time`
- `composite_score`
- `baseline_delta`

## Tests

```bash
python -m pytest tests/ -v
```

## Publish to OpenReward

The env is registered as **`atman/DeceptionSearch-v0`** at
https://openreward.ai/atman/DeceptionSearch-v0.

```bash
# Confirm auth
orwd whoami

# Register the env (one-time)
orwd create DeceptionSearch-v0 \
  --namespace atman \
  --description "Find a hidden AWS access key in a simulated dev laptop populated with LLM-generated decoys."

# Wire to this GitHub repo (auto-deploys on push)
orwd link atman/DeceptionSearch-v0 AnalystTom/ComplexWorld-Hackathon

# Upload the world + task specs to the env's file store (mounted at /orwd_data)
orwd upload atman/DeceptionSearch-v0 \
  scenarios/compromised_laptop/base_tree.json \
  tasks/smoke.json \
  tasks/dev.json \
  tasks/test.json
```

After link + upload, hosted runs invoke the env at
`https://openreward.ai/atman/DeceptionSearch-v0`. The env class identifies
itself with `name() == "DeceptionSearch-v0"` (see `server.py`).

## Layout

```
server.py                              # the env (Phase B runtime)
build_tasks.py                         # offline task generator (Phase A)
scenarios/
  build_base_tree.py                   # ingest OSS repo -> base_tree.json
  compromised_laptop/
    base_tree.json                     # 138 files, 39 dirs
    candidate_locations.json           # 8 secret-planting sites with templates
    allowed_honeypot_locations.json    # 60 paths
    scenario_description.txt
agents/
  tool_schema.py                       # canonical tool defs (1 source of truth)
  harness.py                           # Searcher loop + JSON logger
  run.py                               # CLI
  providers/
    anthropic_provider.py              # Haiku 4.5
    openai_provider.py                 # GPT-5.4
  baselines/
    random_agent.py
    exhaustive_agent.py
tasks/                                 # generated task specs
runs/                                  # per-session logs
tests/
```
