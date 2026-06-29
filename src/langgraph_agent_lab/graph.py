"""Graph construction.

This module is intentionally import-safe. It imports LangGraph only inside the builder so unit tests
that check schema/metrics can run even if students are still debugging graph wiring.
"""

from __future__ import annotations

from typing import Any

from .state import AgentState

# Conditional-edge targets, declared once so add_conditional_edges has an explicit
# path map (keeps the compiled graph diagram readable).
_CLASSIFY_PATHS = {
    "answer": "answer",
    "tool": "tool",
    "clarify": "clarify",
    "risky_action": "risky_action",
    "retry": "retry",
}


def build_graph(checkpointer: Any | None = None):
    """Build and compile the LangGraph workflow.

    TODO(student): Build the complete graph with this architecture:

    START → intake → classify → [conditional: route_after_classify]
      simple       → answer → finalize → END
      tool         → tool → evaluate → [conditional: route_after_evaluate]
                                          success → answer → finalize → END
                                          needs_retry → retry → [conditional: route_after_retry]
                                                                  tool (retry)
                                                                  dead_letter → finalize → END
      missing_info → clarify → finalize → END
      risky        → risky_action → approval → [conditional: route_after_approval]
                                                  approved → tool → evaluate → ...
                                                  rejected → clarify → finalize → END
      error        → retry → [conditional: route_after_retry] → ...

    Steps:
    1. Import StateGraph, START, END from langgraph.graph
    2. Create StateGraph(AgentState)
    3. Import and add all nodes from nodes.py (11 nodes total)
    4. Import and use routing functions from routing.py for conditional edges
    5. Add fixed edges (e.g., START→intake, intake→classify, tool→evaluate, etc.)
    6. Add conditional edges using add_conditional_edges()
    7. Compile with checkpointer: graph.compile(checkpointer=checkpointer)

    Reference: https://langchain-ai.github.io/langgraph/how-tos/create-react-agent/
    """
    from langgraph.graph import END, START, StateGraph

    from . import nodes, routing

    g = StateGraph(AgentState)

    g.add_node("intake", nodes.intake_node)
    g.add_node("classify", nodes.classify_node)
    g.add_node("tool", nodes.tool_node)
    g.add_node("evaluate", nodes.evaluate_node)
    g.add_node("answer", nodes.answer_node)
    g.add_node("clarify", nodes.ask_clarification_node)
    g.add_node("risky_action", nodes.risky_action_node)
    g.add_node("approval", nodes.approval_node)
    g.add_node("retry", nodes.retry_or_fallback_node)
    g.add_node("dead_letter", nodes.dead_letter_node)
    g.add_node("finalize", nodes.finalize_node)

    # Fixed edges
    g.add_edge(START, "intake")
    g.add_edge("intake", "classify")
    g.add_edge("tool", "evaluate")
    g.add_edge("risky_action", "approval")
    g.add_edge("answer", "finalize")
    g.add_edge("clarify", "finalize")
    g.add_edge("dead_letter", "finalize")
    g.add_edge("finalize", END)

    # Conditional edges
    g.add_conditional_edges("classify", routing.route_after_classify, _CLASSIFY_PATHS)
    g.add_conditional_edges(
        "evaluate", routing.route_after_evaluate, {"retry": "retry", "answer": "answer"}
    )
    g.add_conditional_edges(
        "retry", routing.route_after_retry, {"tool": "tool", "dead_letter": "dead_letter"}
    )
    g.add_conditional_edges(
        "approval", routing.route_after_approval, {"tool": "tool", "clarify": "clarify"}
    )

    return g.compile(checkpointer=checkpointer)


def run_to_completion(graph: Any, state: dict, config: dict, decision: dict | None = None) -> dict:
    """Invoke the graph, auto-resuming through any HITL interrupt() pauses.

    Used by the CLI (auto-approve) and tests. The web UI runs its own loop so a
    human can actually approve/reject at the interrupt point.
    """
    from langgraph.types import Command

    result = graph.invoke(state, config=config)
    while isinstance(result, dict) and result.get("__interrupt__"):
        resume = decision or {"approved": True, "reviewer": "cli-auto", "comment": "auto-approved"}
        result = graph.invoke(Command(resume=resume), config=config)
    return result
