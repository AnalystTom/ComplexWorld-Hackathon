# Network Benchmark Design

## Goal

Add a second benchmark beside the existing Linux/filesystem environment on a
separate `network_benchmark` branch. The new benchmark should model a small,
verifiable, partially observable cyber world where an agent performs
multi-step exploitation chains over time.

This work must stay compatible with the repo's RL environment constraints:
there should be a pure internal step core and a thin ORS/OpenReward adapter on
top of it.

## Non-Goals

- Replacing the existing Linux/filesystem benchmark
- Introducing a defender agent or multi-agent runtime
- Building a large procedural simulator before the curated benchmark works
- Exposing raw shell commands as the action interface

## Architecture

The new benchmark should live beside the current environment, not inside it.
It should mirror the repo's existing split between offline task specs and
runtime environment logic.

Proposed layout:

- `network_benchmark/scenarios/`: checked-in `MiniCyberBench v1` task files
- `network_benchmark/schema.py`: validates scenario documents
- `network_benchmark/world.py`: loads one scenario and constructs immutable
  world state
- `network_benchmark/step_env.py`: pure `step(action) -> (obs, reward, done,
  info)` core
- `network_benchmark/ors_env.py`: thin ORS/OpenReward adapter exposing the
  action space as tools
- `network_benchmark/generate_tasks.py`: emits schema-valid variants from the
  same scenario model

The step core is the source of truth. The ORS adapter should translate tool
calls into typed actions and surface the resulting observation, reward, done
flag, and metadata without adding benchmark logic.

## Dataset Shape

The dataset is a collection of scenario files, not rows or static examples.
`MiniCyberBench v1` should ship with five curated scenarios:

- `task_01_linear`
- `task_02_hidden_vuln`
- `task_03_decoy`
- `task_04_multi_path`
- `task_05_delayed_exploit`

These five tasks are the canonical benchmark core. The generator exists to
produce additional tasks from the same schema, not to replace the curated set.

## Scenario Schema

Each scenario file should be a checked-in YAML document containing static,
inspectable data only:

- `id`
- `description`
- `difficulty`
- `nodes`
- `start_node`
- `goal`
- `rules` (optional)
- `expected_optimal_path` (optional)

Each node should contain:

- `id`
- `services`
- `neighbors`
- `loot` (optional)
- `host_state` (optional)

Each service should contain:

- `name`
- `vuln`

Allowed vulnerability types for the first version should be a constrained set:

- `sql_injection`
- `hidden_sql_injection`
- `fake_sql_injection`
- `weak_auth`
- `setup_required`
- `none`

Rules should stay declarative. They can unlock future state transitions, but
must not embed arbitrary executable code in the task files.

For the first version, `goal` should be a small declarative object rather than
a free-form string. A sufficient shape is:

- `type`: terminal condition kind such as `exfiltrate_from_node`
- `target_node`: node required for success
- `requires_privilege`: boolean

## Runtime State

Runtime state should be separate from scenario definition. The mutable state
tracked by the step core should include:

- `current_node`
- `known_nodes`
- `scanned_nodes`
- `discovered_vulns`
- `compromised_nodes`
- `privileged_nodes`
- `exfiltrated`
- `alerts`
- `step_count`
- `trajectory`

This split keeps tasks reproducible while preserving partial observability and
long-horizon reasoning. Hidden vulnerabilities become actionable only after
the required discovery step. Fake vulnerabilities remain discoverable but do
not progress the task. Delayed rules can unlock new opportunities after a
specific sequence or wait period.

## Action Model

The benchmark should expose a small abstract action layer:

- `scan(node)`
- `exploit(node)`
- `move(node)`
- `escalate(node)`
- `exfiltrate(node)`

Action semantics:

- `scan(node)`: reveals visible services and may reveal hidden vulnerabilities
- `exploit(node)`: succeeds only if the vulnerability is known and any
  prerequisites are satisfied
- `move(node)`: requires a valid reachable path from a compromised foothold
- `escalate(node)`: upgrades access when privilege conditions are met
- `exfiltrate(node)`: succeeds only when the scenario goal condition is met at
  the target node

The first release should remain single-agent against a static environment.

## Observation Model

Observations should expose only partial state, not the full world. A typical
observation payload should contain:

- currently known nodes
- newly revealed scan findings
- current access state on reachable nodes
- high-level alert or failure feedback
- step count and remaining episode budget

The observation model should make exploration necessary while staying
deterministic for a fixed scenario and action sequence.

## Reward Model

Rewards should be dense enough to guide search but conservative enough to
avoid trivial reward hacking.

Default branch-wide reward schedule:

- successful first-time scan: `+0.2`
- successful exploit: `+1.0`
- valid move into a new node: `+0.5`
- successful privilege escalation: `+1.0`
- failed exploit: `-0.5`
- invalid action: `-0.2`
- wrong exfiltration: `-1.0`
- final goal: `+5.0`

The goal reward should be large enough that winning dominates local shaping,
while repeated low-value actions should not outscore a clean successful trace.

## Verification And Metrics

Each checked-in scenario must be:

- buildable by the loader
- solvable by a scripted golden trace
- checkable by a deterministic goal verifier

The generator must only emit tasks that pass the same validation and solution
checks.

Terminal `info` should include trajectory-based metrics in addition to success:

- `success`
- `steps`
- `useful_scans`
- `failed_actions`
- `unique_nodes_compromised`
- `path_optimality`
- `expected_vs_actual_trace`

This is the benchmark's main differentiator: it should make failure and
reward-hacking behavior legible, not just report a binary outcome.

## Task Suite

`MiniCyberBench v1` should start with five hand-authored tasks:

1. Linear chain baseline
2. Hidden vulnerability requiring scan-first behavior
3. Decoy vulnerability that punishes naive exploit behavior
4. Multi-path scenario with multiple valid strategies
5. Delayed exploit scenario with temporal dependency

The generator can later vary visibility, decoys, and path structure while
preserving the same benchmark semantics.

## Testing Plan

Add targeted tests for:

- schema validation rejects malformed scenarios
- each checked-in scenario is solvable by a golden trace
- step-core determinism for a fixed task and action sequence
- ORS adapter parity with the step core
- generator outputs only schema-valid, solvable tasks

The tests should focus on benchmark integrity, not on broad refactors to the
existing Linux/filesystem environment.

## Implementation Order

1. Add scenario fixtures and schema validation
2. Implement world loader
3. Implement the pure step core
4. Add scripted-task tests over the curated dataset
5. Implement the ORS/OpenReward adapter
6. Add the constrained scenario generator
7. Update README/docs with a short second-benchmark section

## Branch Plan

Create `network_benchmark` from `opus-spec` and keep the benchmark isolated on
that branch. The Linux/filesystem environment should remain intact and usable
throughout this work.
