import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from langchain_ollama import OllamaLLM
from graph.state import AgentState
import json, re

llm = OllamaLLM(model="llama3.2", temperature=0.0)


def intent_agent(state: AgentState) -> AgentState:

    # Simple rule-based fallback + LLM
    message = state['user_message'].lower()

    # Rule-based intent (fast + reliable)
    if any(w in message for w in ["where", "order", "track", "status", "delivered"]):
        intent = "order_tracking"
    elif any(w in message for w in ["refund", "money back", "reimburse"]):
        intent = "refund_request"
    elif any(w in message for w in ["return", "send back", "replace"]):
        intent = "return_item"
    elif any(w in message for w in ["cancel"]):
        intent = "cancellation"
    elif any(w in message for w in ["damaged", "broken", "defective", "complaint"]):
        intent = "complaint"
    elif any(w in message for w in ["policy", "how", "what", "when"]):
        intent = "product_query"
    else:
        intent = "general"

    # Rule-based sentiment
    if any(w in message for w in ["angry", "furious", "unacceptable", "ridiculous", "legal", "sue", "worst"]):
        sentiment = "angry"
    elif any(w in message for w in ["frustrated", "disappointed", "terrible", "horrible"]):
        sentiment = "frustrated"
    elif any(w in message for w in ["worried", "concerned", "late", "delayed", "still"]):
        sentiment = "concerned"
    else:
        sentiment = "neutral"

    # Rule-based urgency
    if any(w in message for w in ["legal", "court", "immediately", "urgent", "asap", "now"]):
        urgency = "critical"
    elif any(w in message for w in ["late", "delayed", "waiting", "days"]):
        urgency = "high"
    elif any(w in message for w in ["when", "how long", "update"]):
        urgency = "medium"
    else:
        urgency = "low"

    print(f"[Intent Agent] intent={intent} | sentiment={sentiment} | urgency={urgency}")

    return {
        **state,
        "intent":    intent,
        "sentiment": sentiment,
        "urgency":   urgency,
    }