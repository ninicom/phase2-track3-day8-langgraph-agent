"""Tiny stdlib web UI to visualize the LangGraph agent.

Run:  python web/app.py   (then open http://127.0.0.1:8000)

No web framework dependency — http.server only. Builds one in-process graph with a
MemorySaver so HITL interrupt/resume works across the /api/run and /api/resume calls.
"""

from __future__ import annotations

import contextvars
from datetime import datetime
import json
import logging
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from langgraph.types import Command

from langgraph_agent_lab.graph import build_graph
from langgraph_agent_lab.llm import load_env
from langgraph_agent_lab.persistence import build_checkpointer
from langgraph_agent_lab.scenarios import load_scenarios
from langgraph_agent_lab.state import Route, Scenario, initial_state

# Setup thread-safe context logging
current_logs = contextvars.ContextVar("current_logs", default=None)

class ContextVarLogHandler(logging.Handler):
    def emit(self, record):
        logs = current_logs.get()
        if logs is not None:
            time_str = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
            msg = f"[{time_str}] [{record.levelname}] {record.getMessage()}"
            logs.append(msg)

agent_logger = logging.getLogger("langgraph_agent_lab")
agent_logger.setLevel(logging.INFO)
# Clear existing handlers to prevent duplication in reload scenarios
for h in list(agent_logger.handlers):
    agent_logger.removeHandler(h)
agent_logger.addHandler(ContextVarLogHandler())


ROOT = Path(__file__).resolve().parent.parent
INDEX = Path(__file__).resolve().parent / "index.html"

# Build the graph once. MemorySaver keeps interrupted threads alive between requests.
GRAPH = build_graph(checkpointer=build_checkpointer("memory"))

_SECRET_HINTS = ("KEY", "SECRET", "TOKEN", "PASSWORD")


def _mask(key: str, value: str) -> str:
    if any(h in key.upper() for h in _SECRET_HINTS) and len(value) > 10:
        return f"{value[:6]}…{value[-4:]} ({len(value)} chars)"
    return value


def _interrupt_payload(result: dict):
    items = result.get("__interrupt__") if isinstance(result, dict) else None
    if not items:
        return None
    first = items[0]
    return getattr(first, "value", first)


def _state_view(config: dict) -> dict:
    values = GRAPH.get_state(config).values
    return {
        "route": values.get("route"),
        "risk_level": values.get("risk_level"),
        "attempt": values.get("attempt"),
        "max_attempts": values.get("max_attempts"),
        "final_answer": values.get("final_answer"),
        "pending_question": values.get("pending_question"),
        "proposed_action": values.get("proposed_action"),
        "approval": values.get("approval"),
        "tool_results": values.get("tool_results", []),
        "errors": values.get("errors", []),
        "events": values.get("events", []),
    }


def _run(query: str, max_attempts: int) -> dict:
    logs_list = []
    token = current_logs.set(logs_list)
    try:
        agent_logger.info("Initializing run for query: '%s'", query)
        scenario = Scenario(id=f"web-{uuid.uuid4().hex[:8]}", query=query,
                            expected_route=Route.SIMPLE, max_attempts=max_attempts)
        state = initial_state(scenario)
        thread_id = f"web-{uuid.uuid4().hex}"
        config = {"configurable": {"thread_id": thread_id}}
        
        agent_logger.info("Starting graph execution. thread_id: %s", thread_id)
        result = GRAPH.invoke(state, config=config)
        payload = _interrupt_payload(result)
        out = {"thread_id": thread_id, "state": _state_view(config)}
        if payload is not None:
            out["interrupted"] = True
            out["approval_request"] = payload
        agent_logger.info("Graph execution completed/interrupted.")
        out["logs"] = list(logs_list)
        return out
    except Exception as exc:
        agent_logger.error("Error during graph execution: %s", str(exc), exc_info=True)
        return {"error": str(exc), "logs": list(logs_list)}
    finally:
        current_logs.reset(token)


def _resume(thread_id: str, approved: bool, comment: str) -> dict:
    logs_list = []
    token = current_logs.set(logs_list)
    try:
        agent_logger.info("Resuming thread %s. approved=%s, comment='%s'", thread_id, approved, comment)
        config = {"configurable": {"thread_id": thread_id}}
        GRAPH.invoke(
            Command(resume={"approved": approved, "reviewer": "web-user", "comment": comment}),
            config=config,
        )
        out = {"thread_id": thread_id, "state": _state_view(config)}
        agent_logger.info("Graph resumed successfully.")
        out["logs"] = list(logs_list)
        return out
    except Exception as exc:
        agent_logger.error("Error during graph resume: %s", str(exc), exc_info=True)
        return {"error": str(exc), "logs": list(logs_list)}
    finally:
        current_logs.reset(token)


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False, default=str).encode(), "application/json")

    def log_message(self, *args) -> None:  # noqa: D401 — quiet console
        pass

    def do_GET(self) -> None:  # noqa: N802
        try:
            if self.path in ("/", "/index.html"):
                self._send(200, INDEX.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/scenarios":
                self._json([s.model_dump(mode="json") for s in load_scenarios(ROOT / "data/sample/scenarios.jsonl")])
            elif self.path == "/api/env":
                env = load_env(ROOT / ".env")
                self._json({k: _mask(k, v) for k, v in env.items()})
            elif self.path == "/api/graph":
                self._json({"mermaid": GRAPH.get_graph().draw_mermaid()})
            elif self.path == "/api/metrics":
                p = ROOT / "outputs/metrics.json"
                self._json(json.loads(p.read_text(encoding="utf-8")) if p.exists() else {})
            else:
                self._json({"error": "not found"}, 404)
        except Exception as exc:  # noqa: BLE001
            self._json({"error": str(exc)}, 500)

    def do_POST(self) -> None:  # noqa: N802
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            if self.path == "/api/run":
                self._json(_run(body.get("query", "").strip(), int(body.get("max_attempts", 3))))
            elif self.path == "/api/resume":
                self._json(_resume(body["thread_id"], bool(body.get("approved")), body.get("comment", "")))
            else:
                self._json({"error": "not found"}, 404)
        except Exception as exc:  # noqa: BLE001
            self._json({"error": str(exc)}, 500)


if __name__ == "__main__":
    addr = ("127.0.0.1", 8000)
    print(f"Agent visualizer running at http://{addr[0]}:{addr[1]}  (Ctrl+C to stop)")
    ThreadingHTTPServer(addr, Handler).serve_forever()
