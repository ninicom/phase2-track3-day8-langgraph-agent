"""Node functions for the LangGraph workflow.

Each function receives AgentState and returns a partial state update dict.
Do NOT mutate input state — return new values only.

LLM REQUIREMENT:
- classify_node MUST use a real LLM call (structured output for intent classification)
- answer_node MUST use a real LLM call (grounded response generation)
- evaluate_node uses LLM-as-judge with a heuristic fallback (bonus)
"""

from __future__ import annotations

import logging
import os
import time
from typing import Literal

from pydantic import BaseModel, Field

from .llm import get_llm
from .state import AgentState, ApprovalDecision, make_event

agent_logger = logging.getLogger("langgraph_agent_lab")


def _truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


# ─── EXAMPLE: working node (provided for reference) ──────────────────
def intake_node(state: AgentState) -> dict:
    """Normalize raw query. This node is provided as a working example."""
    query = state.get("query", "").strip()
    agent_logger.info("[intake] Normalizing raw query: '%s'", query)
    return {
        "query": query,
        "messages": [f"intake:{query[:40]}"],
        "events": [make_event("intake", "completed", "đã chuẩn hóa truy vấn")],
    }


# ─── LLM classification ──────────────────────────────────────────────
class Classification(BaseModel):
    """Structured classification result the LLM is forced to return."""

    route: Literal["simple", "tool", "missing_info", "risky", "error"] = Field(
        description="Intent route for the support ticket."
    )
    risk_level: Literal["low", "high"] = Field(
        description="high only for risky actions with side effects, else low."
    )


_CLASSIFY_SYSTEM = (
    "You are a support-ticket triage classifier. Classify the user's message into "
    "exactly one route. Apply this PRIORITY when more than one could fit: "
    "risky > tool > missing_info > error > simple.\n"
    "- risky: actions with side effects (refunds, deletions, cancellations, sending emails, "
    "charging/processing payments, account changes). risk_level=high.\n"
    "- tool: information lookups (order status, tracking, search) needing an external tool.\n"
    "- missing_info: vague/incomplete requests lacking the context needed to act "
    "(e.g. 'can you fix it?').\n"
    "- error: reports of system failures (timeouts, crashes, service unavailable, cannot recover).\n"
    "- simple: general questions answerable directly without tools or actions.\n"
    "Set risk_level=high only when route=risky; otherwise low.\n"
    'Respond ONLY with a JSON object: {"route": <one of simple|tool|missing_info|risky|error>, '
    '"risk_level": <low|high>}.'
)


def classify_node(state: AgentState) -> dict:
    """Classify the query into a route using an LLM with structured output."""
    start = time.perf_counter()
    query = state.get("query", "")
    agent_logger.info("[classify] Calling LLM structured output to classify query: '%s'", query)
    # method="json_mode": the DeepSeek proxy runs in thinking mode, which rejects both
    # json_schema response_format and forced tool_choice — but json_object mode works.
    llm = get_llm().with_structured_output(Classification, method="json_mode")
    result: Classification = llm.invoke(
        [
            ("system", _CLASSIFY_SYSTEM),
            ("human", query),
        ]
    )
    latency = int((time.perf_counter() - start) * 1000)
    risk = "high" if result.route == "risky" else result.risk_level
    agent_logger.info("[classify] LLM classification: route='%s', risk_level='%s' (Latency: %dms)", result.route, risk, latency)
    return {
        "route": result.route,
        "risk_level": risk,
        "messages": [f"classify:{result.route}"],
        "events": [
            make_event(
                "classify",
                "completed",
                f"tuyến={result.route} rủi ro={risk}",
                latency_ms=latency,
                route=result.route,
                risk_level=risk,
            )
        ],
    }


# ─── Mock tool execution ─────────────────────────────────────────────
def tool_node(state: AgentState) -> dict:
    """Execute a mock tool call."""
    attempt = state.get("attempt", 0)
    query = state.get("query", "")
    agent_logger.info("[tool] Executing mock tool. attempt=%d", attempt)
    result = f"TOOL_OK: kết quả tra cứu cho '{query[:60]}' (lần thử {attempt})"
    event = make_event("tool", "completed", "công cụ đã trả về kết quả", attempt=attempt)
    agent_logger.info("[tool] Mock tool executed successfully. Result: '%s'", result)
    return {"tool_results": [result], "events": [event]}


# ─── Evaluate (LLM-as-judge with heuristic fallback) ─────────────────
class Verdict(BaseModel):
    should_retry: bool = Field(
        description="True ONLY if the tool result indicates a failure/error/empty result "
        "that warrants a retry. False if the tool executed and returned a usable result."
    )


def evaluate_node(state: AgentState) -> dict:
    """Evaluate the latest tool result — the retry-loop gate."""
    results = state.get("tool_results") or []
    latest = results[-1] if results else ""
    agent_logger.info("[evaluate] Evaluating latest tool result: '%s'", latest)
    # Cheap, deterministic short-circuit: an explicit ERROR always needs retry.
    if "ERROR" in latest:
        verdict = "needs_retry"
        detail = "công cụ báo lỗi ERROR"
        agent_logger.warning("[evaluate] Short-circuit check: Tool reported ERROR. Verdict: '%s'", verdict)
    else:
        try:  # LLM-as-judge (bonus); fall back to heuristic on any failure.
            agent_logger.info("[evaluate] Calling LLM judge to analyze result...")
            judge = get_llm().with_structured_output(Verdict, method="json_mode")
            retry = judge.invoke(
                [
                    ("system", "You judge a tool execution result. A result is a FAILURE only if it "
                     "reports an error, timeout, or is empty. A normal tool result that returned data "
                     'is a SUCCESS. Respond ONLY with JSON: {"should_retry": true|false}.'),
                    ("human", f"Query: {state.get('query','')}\nTool result: {latest}"),
                ]
            ).should_retry
            verdict = "needs_retry" if retry else "success"
            detail = f"llm-judge should_retry={retry}"
            agent_logger.info("[evaluate] LLM Judge result: should_retry=%s -> Verdict: '%s'", retry, verdict)
        except Exception as exc:  # noqa: BLE001 — degrade gracefully, never break the graph
            verdict = "success"
            detail = f"heuristic fallback ({type(exc).__name__})"
            agent_logger.error("[evaluate] LLM Judge call failed. Falling back to heuristic: 'success'. error=%s", str(exc))
    return {
        "evaluation_result": verdict,
        "events": [make_event("evaluate", "completed", detail, verdict=verdict)],
    }


# ─── LLM grounded answer ─────────────────────────────────────────────
def answer_node(state: AgentState) -> dict:
    """Generate a final response using an LLM, grounded in available context."""
    start = time.perf_counter()
    tool_results = state.get("tool_results") or []
    approval = state.get("approval")
    agent_logger.info("[answer] Generating grounded final reply. Tool results count: %d, has_approval: %s", len(tool_results), approval is not None)
    context_lines = [f"User request: {state.get('query', '')}"]
    if tool_results:
        context_lines.append("Tool results:\n" + "\n".join(f"- {r}" for r in tool_results))
    if approval:
        context_lines.append(
            f"Human approval: approved={approval.get('approved')} "
            f"by {approval.get('reviewer')} ({approval.get('comment')})"
        )
    llm = get_llm()
    response = llm.invoke(
        [
            ("system", "You are a concise, helpful support agent. Write a short final reply to the "
             "customer. If tool results or an approval decision are provided, ground your answer in "
             "them and do not contradict them. For general how-to questions with no tool context, "
             "answer helpfully from general product-support knowledge. "
             "Always reply in the same language as the user's request."),
            ("human", "\n\n".join(context_lines)),
        ]
    )
    answer = getattr(response, "content", str(response))
    latency = int((time.perf_counter() - start) * 1000)
    agent_logger.info("[answer] LLM generated final answer (Latency: %dms, Length: %d chars)", latency, len(answer))
    return {
        "final_answer": answer,
        "messages": ["answer:generated"],
        "events": [make_event("answer", "completed", "đã tạo câu trả lời cuối", latency_ms=latency)],
    }


def ask_clarification_node(state: AgentState) -> dict:
    """Ask for missing information instead of hallucinating."""
    query = state.get("query", "")
    agent_logger.info("[clarify] Missing information in query. Requesting clarification LLM prompt. Query: '%s'", query)
    try:
        llm = get_llm()
        resp = llm.invoke(
            [
                ("system", "The user's support request is vague or incomplete. Ask ONE specific "
                 "clarifying question to get the detail you need. Output only the question. "
                 "Ask in the same language as the user's request."),
                ("human", query),
            ]
        )
        question = getattr(resp, "content", str(resp)).strip()
        agent_logger.info("[clarify] Clarification question generated: '%s'", question)
    except Exception as exc:  # noqa: BLE001
        question = "Bạn có thể cung cấp thêm chi tiết (sản phẩm/đơn hàng nào và bạn mong muốn kết quả gì) không?"
        agent_logger.error("[clarify] LLM call failed. Falling back to default question. error=%s", str(exc))
    return {
        "pending_question": question,
        "final_answer": question,
        "messages": ["clarify:asked"],
        "events": [make_event("clarify", "completed", "đã yêu cầu làm rõ")],
    }


def risky_action_node(state: AgentState) -> dict:
    """Prepare a risky action for human approval."""
    query = state.get("query", "")
    agent_logger.info("[risky_action] Preparing risky action node. User query: '%s'", query)
    action = (
        f"Hành động cần phê duyệt: '{query}'. "
        "Hành động này có hậu quả không thể hoàn tác và phải được con người xem xét."
    )
    agent_logger.info("[risky_action] Action prepared: '%s'", action)
    return {
        "proposed_action": action,
        "messages": ["risky:prepared"],
        "events": [make_event("risky_action", "completed", "đã chuẩn bị hành động rủi ro", risk="high")],
    }


def approval_node(state: AgentState) -> dict:
    """Human-in-the-loop approval step.

    Default: mock approval so CI/tests run offline. If LANGGRAPH_INTERRUPT is set,
    pause the graph via interrupt() and use the resumed value as the decision.
    """
    agent_logger.info("[approval] Checking approval constraints. LANGGRAPH_INTERRUPT=%s", os.getenv("LANGGRAPH_INTERRUPT"))
    if _truthy("LANGGRAPH_INTERRUPT"):
        from langgraph.types import interrupt

        proposed = state.get("proposed_action", "")
        agent_logger.warning("[approval] Pausing execution! Raising interrupt() for proposed action: '%s'", proposed)
        decision = interrupt(
            {
                "proposed_action": proposed,
                "query": state.get("query", ""),
                "prompt": "Phê duyệt hành động rủi ro này? Tiếp tục với {'approved': bool, 'comment': str}",
            }
        )
        if isinstance(decision, dict):
            approval = ApprovalDecision(
                approved=bool(decision.get("approved", True)),
                reviewer=decision.get("reviewer", "human"),
                comment=decision.get("comment", ""),
            )
        else:
            approval = ApprovalDecision(approved=bool(decision), reviewer="human")
        agent_logger.info("[approval] Interrupt resumed. Decision received: approved=%s, comment='%s'", approval.approved, approval.comment)
    else:
        approval = ApprovalDecision(approved=True, comment="tự động phê duyệt (người duyệt giả lập)")
        agent_logger.info("[approval] LANGGRAPH_INTERRUPT is not set, mock auto-approved. Comment: '%s'", approval.comment)
    return {
        "approval": approval.model_dump(),
        "messages": [f"approval:{approval.approved}"],
        "events": [
            make_event(
                "approval",
                "completed",
                f"đã phê duyệt={approval.approved}",
                approved=approval.approved,
                reviewer=approval.reviewer,
            )
        ],
    }


def retry_or_fallback_node(state: AgentState) -> dict:
    """Record a retry attempt (tier 1: retry; logs feed the fallback/dead-letter tiers)."""
    attempt = state.get("attempt", 0) + 1
    max_att = state.get("max_attempts", 3)
    msg = f"lỗi tạm thời — lần thử lại {attempt}/{max_att}"
    agent_logger.warning("[retry] Registering retry state: attempt %d/%d. Status: '%s'", attempt, max_att, msg)
    return {
        "attempt": attempt,
        "errors": [msg],
        "messages": [f"retry:{attempt}"],
        "events": [make_event("retry", "retrying", msg, attempt=attempt)],
    }


def dead_letter_node(state: AgentState) -> dict:
    """Handle unresolvable failures after max retries (tier 3: dead letter)."""
    attempt = state.get("attempt", 0)
    agent_logger.error("[dead_letter] Execution reached maximum retry attempts (%d). Redirecting to dead-letter queue.", attempt)
    answer = (
        "Chúng tôi không thể tự động hoàn tất yêu cầu của bạn sau "
        f"{attempt} lần thử. Yêu cầu đã được chuyển cho nhân viên hỗ trợ "
        "(hàng đợi dead-letter) để xử lý tiếp."
    )
    return {
        "final_answer": answer,
        "messages": ["dead_letter:escalated"],
        "events": [make_event("dead_letter", "escalated", "vượt số lần thử tối đa — đã chuyển xử lý")],
    }


def finalize_node(state: AgentState) -> dict:
    """Emit a final audit event. All routes pass through here before END."""
    agent_logger.info("[finalize] Graph execution complete. Performing final serialization/audit.")
    return {
        "messages": ["finalize:done"],
        "events": [make_event("finalize", "completed", "quy trình đã hoàn tất")],
    }
