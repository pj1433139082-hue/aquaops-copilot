from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, Request, status

from aquaops.agent.state import (
    AgentRunCompleted,
    AgentRunRequest,
    AgentState,
    InMemoryAgentRunStore,
)

router = APIRouter(prefix="/v1/agent", tags=["agent"])


@router.post("/runs", response_model=AgentRunCompleted, status_code=status.HTTP_200_OK)
def submit_run(payload: AgentRunRequest, request: Request) -> AgentRunCompleted:
    request_id = request.state.request_id
    store = _run_store(request)
    while True:
        run_id = uuid4()
        result = _run_safely(
            payload,
            request_id,
            run_id,
            operations_graph=request.app.state.operations_graph,
        )
        if store.save(result):
            return result


@router.get("/runs/{run_id}", response_model=AgentRunCompleted)
def get_run(run_id: str, request: Request) -> AgentRunCompleted:
    try:
        parsed_run_id = UUID(run_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Agent run not found"
        )

    result = _run_store(request).get(parsed_run_id)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Agent run not found"
        )
    return result


def _run_safely(
    payload: AgentRunRequest,
    request_id: str,
    run_id: UUID,
    *,
    operations_graph: object | None = None,
) -> AgentRunCompleted:
    state: AgentState = {
        "request_id": request_id,
        "mode": payload.mode,
        "question": payload.question,
    }
    if payload.mode == "operations":
        selected_graph = operations_graph
        if selected_graph is None:
            from aquaops.agent.graph import operations_graph as default_operations_graph

            selected_graph = default_operations_graph
        completed_state = selected_graph.invoke(state)
    else:
        from aquaops.research.graph import research_graph

        completed_state = research_graph.invoke(state)

    return AgentRunCompleted(
        run_id=run_id,
        request_id=request_id,
        status="completed",
        answer=completed_state["answer"],
        evidence=completed_state["evidence"],
        requires_human_review=completed_state["requires_human_review"],
    )


def _run_store(request: Request) -> InMemoryAgentRunStore:
    return request.app.state.agent_run_store
