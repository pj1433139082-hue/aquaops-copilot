from collections import OrderedDict
from threading import Lock
from typing import TYPE_CHECKING, Literal, TypedDict
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    from aquaops.agent.coverage import CoverageGateResult
    from aquaops.agent.policy import ToolBudget, ToolTrace
else:
    # LangGraph resolves TypedDict hints at runtime.  These fields are
    # constructed only by ``initialize_safe_agent_state``; importing their
    # concrete operation-only classes here would break the research graph's
    # existing import boundary.
    ToolBudget = object
    ToolTrace = object
    CoverageGateResult = object

AgentMode = Literal["operations", "research"]


class AgentRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: AgentMode
    question: str = Field(min_length=3, max_length=1000)

    @field_validator("question", mode="before")
    @classmethod
    def question_must_contain_non_whitespace(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        question = value.strip()
        if not question:
            raise ValueError("question must contain non-whitespace content")
        return question


class AgentRunCompleted(BaseModel):
    run_id: UUID
    request_id: str
    status: Literal["completed"]
    answer: str
    evidence: list[dict[str, str]]
    requires_human_review: bool


class InMemoryAgentRunStore:
    """Thread-safe, bounded process-local results for single-process local development.

    The oldest completed result is evicted when capacity is exceeded. This is not
    durable storage or a task queue: process restarts and multiple workers do not
    share results. A later backend implementation must replace this store.
    """

    def __init__(self, max_entries: int = 100) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1")
        self._max_entries = max_entries
        self._results: OrderedDict[UUID, AgentRunCompleted] = OrderedDict()
        self._lock = Lock()

    def save(self, result: AgentRunCompleted) -> bool:
        with self._lock:
            if result.run_id in self._results:
                return False
            self._results[result.run_id] = result
            while len(self._results) > self._max_entries:
                self._results.popitem(last=False)
            return True

    def get(self, run_id: UUID) -> AgentRunCompleted | None:
        with self._lock:
            return self._results.get(run_id)


class AgentState(TypedDict, total=False):
    request_id: str
    mode: AgentMode
    question: str
    evidence: list[dict[str, str]]
    answer: str
    requires_human_review: bool
    tool_budget: "ToolBudget"
    tool_traces: tuple["ToolTrace", ...]
    coverage: "CoverageGateResult | None"


def initialize_safe_agent_state(state: object) -> AgentState:
    """Drop untrusted model state and issue fresh execution-only security fields.

    The public operations graph is a single bounded run.  Its budget and trace
    are therefore created by the host at entry rather than accepted from the
    model/request payload.  Imports stay local so the research graph keeps its
    existing no-operations-tool import boundary.
    """

    from aquaops.agent.policy import ToolBudget

    source = state if type(state) is dict else {}
    initialized: AgentState = {
        "tool_budget": ToolBudget.initial(),
        "tool_traces": (),
        "coverage": None,
    }
    for key in ("request_id", "mode", "question"):
        value = source.get(key)
        if isinstance(value, str):
            initialized[key] = value
    return initialized
