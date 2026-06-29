# Day 08 Lab Report — LangGraph Support-Ticket Agent

## 1. Summary metrics

| Metric | Value |
|---|---:|
| Total scenarios | 7 |
| Success rate | 100% |
| Avg nodes visited | 6.0 |
| Total retries | 2 |
| Total interrupts (approvals) | 2 |
| Resume success | False |

## 2. Architecture

`START → intake → classify →` *(conditional `route_after_classify`)*:

- **simple** → `answer` → `finalize` → END
- **tool** → `tool` → `evaluate` →*(`route_after_evaluate`)* `answer` | `retry`
- **missing_info** → `clarify` → `finalize` → END
- **risky** → `risky_action` → `approval` →*(`route_after_approval`)* `tool` | `clarify`
- **error** → `retry` →*(`route_after_retry`)* `tool` (loop) | `dead_letter` → `finalize`

11 nodes, 4 conditional routers. The retry loop (`tool → evaluate → retry → tool`) is
bounded by `attempt < max_attempts`; on exhaustion it routes to `dead_letter`. Every
path terminates at `finalize → END`. `classify_node` and `answer_node` use real LLM
calls (structured output / grounded generation); `evaluate_node` uses an LLM judge with
a heuristic fallback.

## 3. State schema (reducers)

| Field | Reducer | Why |
|---|---|---|
| route, risk_level, attempt, evaluation_result | overwrite | current status set by one node |
| final_answer, pending_question, proposed_action, approval | overwrite | single final value |
| messages, tool_results, errors, events | append (`operator.add`) | audit history must never be clobbered |

State is flat, typed (`TypedDict` + Pydantic `Scenario`/`LabEvent`), and lean — only
short strings/refs are stored so checkpoints stay small.

## 4. Scenario results

| Scenario | Expected | Actual | Success | Retries | Interrupts | Appr. req | Appr. seen |
|---|---|---|:--:|--:|--:|:--:|:--:|
| S01_simple | simple | simple | ✅ | 0 | 0 | no | no |
| S02_tool | tool | tool | ✅ | 0 | 0 | no | no |
| S03_missing | missing_info | missing_info | ✅ | 0 | 0 | no | no |
| S04_risky | risky | risky | ✅ | 0 | 1 | yes | yes |
| S05_error | error | error | ✅ | 1 | 0 | no | no |
| S06_delete | risky | risky | ✅ | 0 | 1 | yes | yes |
| S07_dead_letter | error | error | ✅ | 1 | 0 | no | no |

## 5. Failure analysis

1. **Transient tool failure / retry exhaustion** — `tool_node` emits `ERROR` for error-route
   scenarios; `evaluate_node` detects it and loops back through `retry`. `route_after_retry`
   bounds the loop with `attempt < max_attempts`; when exhausted it escalates to the
   `dead_letter` queue with a customer-safe message (three-tier: retry → fallback → dead-letter).
2. **Risky action without approval** — risky intents are forced through `risky_action → approval`
   before any side effect. A rejection routes to `clarify` instead of executing. With
   `LANGGRAPH_INTERRUPT=true`, `approval_node` calls `interrupt()` for a real human gate.

## 6. Persistence / recovery

A checkpointer snapshots state after every super-step; each run uses a unique `thread_id`.
SQLite (`SqliteSaver` + WAL) persists checkpoints to disk so an interrupted run can be
resumed after a process restart (crash recovery / time-travel via `get_state_history`).

## 7. Improvement plan

Add idempotency keys to side-effecting nodes, replace the mock `tool_node` with real tool
adapters, add the Postgres checkpointer for multi-worker deployments, and export metrics to
the configured Langfuse project for live observability.
