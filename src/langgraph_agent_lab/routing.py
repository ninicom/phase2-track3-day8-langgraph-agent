"""Routing functions for conditional edges.

Each function takes AgentState and returns a string — the name of the next node.
These strings MUST match node names registered in graph.py.
"""

from __future__ import annotations

import logging
from .state import AgentState

agent_logger = logging.getLogger("langgraph_agent_lab")

_CLASSIFY_NEXT = {
    "simple": "answer",
    "tool": "tool",
    "missing_info": "clarify",
    "risky": "risky_action",
    "error": "retry",
}


def route_after_classify(state: AgentState) -> str:
    """Map the classified route to the next graph node (default: answer)."""
    route = state.get("route", "")
    target = _CLASSIFY_NEXT.get(route, "answer")
    agent_logger.info("[routing] route_after_classify: input_route='%s' -> next_node='%s'", route, target)
    return target


def route_after_evaluate(state: AgentState) -> str:
    """The retry-loop gate: retry on a bad tool result, else answer."""
    eval_res = state.get("evaluation_result")
    target = "retry" if eval_res == "needs_retry" else "answer"
    agent_logger.info("[routing] route_after_evaluate: evaluation_result='%s' -> next_node='%s'", eval_res, target)
    return target


def route_after_retry(state: AgentState) -> str:
    """Bounded retry: try the tool again while under the limit, else dead-letter."""
    attempt = state.get("attempt", 0)
    max_attempts = state.get("max_attempts", 3)
    if attempt < max_attempts:
        target = "tool"
    else:
        target = "dead_letter"
    agent_logger.info("[routing] route_after_retry: attempt=%d, max_attempts=%d -> next_node='%s'", attempt, max_attempts, target)
    return target


def route_after_approval(state: AgentState) -> str:
    """Proceed with the risky action if approved, otherwise ask the user."""
    approval = state.get("approval") or {}
    approved = approval.get("approved", False)
    target = "tool" if approved else "clarify"
    agent_logger.info("[routing] route_after_approval: approved=%s -> next_node='%s'", approved, target)
    return target

