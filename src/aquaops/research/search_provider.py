from typing import Protocol

from pydantic import BaseModel


class SearchHit(BaseModel):
    title: str
    url: str
    snippet: str


class SearchProvider(Protocol):
    def search(self, query: str, limit: int = 5) -> list[SearchHit]: ...


class MockSearchProvider:
    def search(self, query: str, limit: int = 5) -> list[SearchHit]:
        return [
            SearchHit(
                title="mock source",
                url="https://example.invalid/mock",
                snippet=query,
            )
        ][:limit]
