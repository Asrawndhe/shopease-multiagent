from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM
from graph.state import AgentState

llm    = OllamaLLM(model="llama3.2", temperature=0.3)
parser = StrOutputParser()

PROMPT = ChatPromptTemplate.from_template("""You are a helpful and empathetic customer support agent for ShopEase.

CUSTOMER INFORMATION (from CRM):
{customer_context}

RELEVANT COMPANY POLICY (from knowledge base):
{policy_chunks}

DETECTED INTENT: {intent}
CUSTOMER SENTIMENT: {sentiment}

CONVERSATION HISTORY:
{history}

CUSTOMER MESSAGE:
{user_message}

INSTRUCTIONS:
- Address the customer by their first name only.
- Use EXACT values from customer info — no placeholders like [Courier Name] or XXXXXXXX.
- If order is Shipped, mention the specific order ID and current status.
- Apply the correct policy based on their situation.
- If customer is VIP, acknowledge their loyalty warmly.
- If sentiment is angry or frustrated, be extra empathetic.
- Sign off as: ShopEase Support Team
- Keep response under 100 words.
- Never invent information not present above.

YOUR RESPONSE:""")

chain = PROMPT | llm | parser


def supervisor_agent(state: AgentState) -> AgentState:
    """
    Supervisor Agent — sab agents ka output leke final response banata hai.
    Input  : full state
    Output : state["final_response"], state["next"]
    """
    if state.get("should_escalate"):
        print("[Supervisor] Escalation triggered — routing to HITL")
        return {**state, "final_response": None, "next": "escalate"}

    history = "\n".join(
        [f"{'User' if m['role']=='user' else 'Agent'}: {m['content']}"
         for m in state.get("chat_history", [])[-6:]]
    ) or "No previous conversation."

    response = chain.invoke({
        "customer_context": state.get("customer_context", "Not available"),
        "policy_chunks":    state.get("policy_chunks", "Not available"),
        "intent":           state.get("intent", "general"),
        "sentiment":        state.get("sentiment", "neutral"),
        "history":          history,
        "user_message":     state.get("user_message", ""),
    })

    print(f"[Supervisor] Response generated ({len(response)} chars)")
    return {**state, "final_response": response, "next": "end"}
