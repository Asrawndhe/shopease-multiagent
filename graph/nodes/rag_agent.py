import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from graph.state import AgentState

VECTOR_STORE_PATH = "data/faiss_index"
_vectorstore = None


def _get_vectorstore():
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    embeddings = OllamaEmbeddings(model="llama3.2")

    if os.path.exists(VECTOR_STORE_PATH):
        _vectorstore = FAISS.load_local(
            VECTOR_STORE_PATH, embeddings,
            allow_dangerous_deserialization=True
        )
        print("[RAG Agent] Loaded existing vector store.")
    else:
        print("[RAG Agent] Building vector store...")
        loader = TextLoader("data/policies.txt")
        docs   = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        _vectorstore = FAISS.from_documents(chunks, embeddings)
        _vectorstore.save_local(VECTOR_STORE_PATH)
        print(f"[RAG Agent] Built! {len(chunks)} chunks indexed.")

    return _vectorstore


def rag_agent(state: AgentState) -> AgentState:
    """
    RAG Agent — FAISS se relevant policy chunks retrieve karta hai.
    Input  : state["user_message"], state["intent"]
    Output : state["policy_chunks"]
    """
    intent  = state.get("intent", "general")
    message = state.get("user_message", "")
    query   = f"{intent} {message}"

    try:
        vs      = _get_vectorstore()
        results = vs.similarity_search(query, k=3)
        chunks  = "\n\n".join([doc.page_content for doc in results])
        print(f"[RAG Agent] Retrieved {len(results)} chunks for: {intent}")
    except Exception as e:
        chunks = f"Policy retrieval error: {str(e)}"
        print(f"[RAG Agent] Error: {e}")

    return {
        **state,
        "policy_chunks": chunks,
    }
