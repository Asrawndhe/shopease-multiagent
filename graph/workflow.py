from langgraph.graph import StateGraph, END
from graph.state import AgentState
from graph.nodes.intent_agent import intent_agent
from graph.nodes.crm_agent import crm_agent
from graph.nodes.rag_agent import rag_agent
from graph.nodes.escalation_agent import escalation_agent
from graph.supervisor import supervisor_agent


def route_after_supervisor(state: AgentState) -> str:
    return state.get("next", "end")


def build_graph():
    graph = StateGraph(AgentState)

    # Nodes
    graph.add_node("intent_agent",     intent_agent)
    graph.add_node("crm_agent",        crm_agent)
    graph.add_node("rag_agent",        rag_agent)
    graph.add_node("escalation_agent", escalation_agent)
    graph.add_node("supervisor",       supervisor_agent)

    # Entry point
    graph.set_entry_point("intent_agent")

    # Edges — sequential flow
    graph.add_edge("intent_agent",     "crm_agent")
    graph.add_edge("crm_agent",        "rag_agent")
    graph.add_edge("rag_agent",        "escalation_agent")
    graph.add_edge("escalation_agent", "supervisor")

    # Conditional edge — supervisor decides end or escalate
    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {"end": END, "escalate": END}
    )

    return graph.compile()


compiled_graph = build_graph()


def run_graph(user_message: str, customer_id: str, chat_history: list) -> AgentState:
    """Main entry point called by app.py"""

    initial_state: AgentState = {
        "user_message":      user_message,
        "customer_id":       customer_id,
        "chat_history":      chat_history,
        "intent":            None,
        "sentiment":         None,
        "urgency":           None,
        "customer_data":     None,
        "customer_context":  None,
        "policy_chunks":     None,
        "should_escalate":   None,
        "escalation_reason": None,
        "draft_reply":       None,
        "final_response":    None,
        "next":              None,
    }

    print(f"\n{'='*50}")
    print(f"[Graph] Starting: {user_message[:60]}...")
    result = compiled_graph.invoke(initial_state)
    print(f"[Graph] Done — escalate={result.get('should_escalate')} | intent={result.get('intent')}")
    print(f"{'='*50}\n")

    return result
