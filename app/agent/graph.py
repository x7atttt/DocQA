from langgraph.graph import END, StateGraph

from app.agent.nodes import general_answer, generate_answer, intent_router, retrieve_documents, rewrite_query
from app.agent.state import AgentState


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("intent_router", intent_router)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("retrieve_documents", retrieve_documents)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("general_answer", general_answer)

    graph.set_entry_point("intent_router")
    # 需检索时：先改写 query（多轮指代消解）再检索
    graph.add_conditional_edges(
        "intent_router",
        lambda state: state.get("should_retrieve", False),
        {True: "rewrite_query", False: "general_answer"},
    )
    graph.add_edge("rewrite_query", "retrieve_documents")
    graph.add_edge("retrieve_documents", "generate_answer")
    graph.add_edge("generate_answer", END)
    graph.add_edge("general_answer", END)
    return graph.compile()
