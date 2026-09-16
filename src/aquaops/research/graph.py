from langgraph.graph import END, StateGraph

from aquaops.agent.state import AgentState
from aquaops.rag.citations import build_grounded_answer


def _safe_refusal(state: AgentState) -> AgentState:
    grounded = build_grounded_answer("", [])
    return {
        **state,
        "evidence": grounded.citations,
        "answer": grounded.answer,
        "requires_human_review": grounded.requires_human_review,
    }


def research_node(state: AgentState) -> AgentState:
    """Refuse research answers until public, citable search evidence is connected."""
    question = state.get("question")
    if not isinstance(question, str) or not question.strip():
        return _safe_refusal(state)

    grounded = build_grounded_answer(question, [])
    return {
        **state,
        "evidence": grounded.citations,
        "answer": grounded.answer,
        "requires_human_review": grounded.requires_human_review,
    }


def run_research_graph(state: AgentState) -> AgentState:
    return research_graph.invoke(state)


_research_builder = StateGraph(AgentState)
_research_builder.add_node("research", research_node)
_research_builder.set_entry_point("research")
_research_builder.add_edge("research", END)
research_graph = _research_builder.compile()
