import re
from langchain_ollama import OllamaLLM
from graph.state import AgentState

llm = OllamaLLM(model="llama3.2", temperature=0.3)

LEGAL_KEYWORDS  = ["legal action", "consumer court", "lawyer", "police", "chargeback", "fraud", "sue", "court"]
REFUND_KEYWORDS = ["refund", "money back", "return", "damaged", "broken"]
AMOUNT_THRESHOLD = 5000


def escalation_agent(state: AgentState) -> AgentState:
    """
    Escalation Agent — HITL trigger karta hai aur AI draft banata hai.
    Input  : state["user_message"], state["customer_data"], state["sentiment"]
    Output : state["should_escalate"], state["escalation_reason"], state["draft_reply"]
    """
    message   = state.get("user_message", "").lower()
    customer  = state.get("customer_data", {})
    sentiment = state.get("sentiment", "neutral")

    should_escalate = False
    reason = ""

    # Check 1 — Legal keywords
    for kw in LEGAL_KEYWORDS:
        if kw in message:
            should_escalate = True
            reason = f"Legal threat detected: '{kw}'"
            break

    # Check 2 — High refund amount
    if not should_escalate:
        amounts = re.findall(r'rs\.?\s*(\d+)', message)
        for amt in amounts:
            if int(amt) > AMOUNT_THRESHOLD:
                should_escalate = True
                reason = f"High refund amount: Rs {amt} (limit Rs {AMOUNT_THRESHOLD})"
                break

    # Check 3 — Repeat complaints + refund intent
    if not should_escalate:
        complaints = int(customer.get("complaints", 0))
        has_refund = any(kw in message for kw in REFUND_KEYWORDS)
        if complaints >= 3 and has_refund:
            should_escalate = True
            reason = f"Customer has {complaints} unresolved complaints with refund intent"

    # Check 4 — Angry VIP
    if not should_escalate:
        if sentiment == "angry" and customer.get("tier", "") == "VIP":
            should_escalate = True
            reason = "Angry VIP customer — priority human attention needed"

    # Generate AI draft if escalating
    draft = ""
    if should_escalate:
        name = customer.get("name", "Customer")
        tier = customer.get("tier", "regular")
        draft = llm.invoke(f"""Write a short empathetic draft reply for a human support agent.
Keep under 60 words. Be apologetic and personal. Do not mention being an AI.

Customer: {name} ({tier.upper()} member)
Issue: {state.get('user_message')}
Reason: {reason}

Draft reply:""")

    print(f"[Escalation Agent] escalate={should_escalate} | reason={reason}")

    return {
        **state,
        "should_escalate":   should_escalate,
        "escalation_reason": reason,
        "draft_reply":       draft,
    }
