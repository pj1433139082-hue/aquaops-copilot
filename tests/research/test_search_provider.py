from aquaops.research.search_provider import MockSearchProvider, SearchHit


def test_mock_search_provider_returns_sourced_search_hit() -> None:
    hits = MockSearchProvider().search("ammonia nitrogen public evidence")

    assert hits == [
        SearchHit(
            title="mock source",
            url="https://example.invalid/mock",
            snippet="ammonia nitrogen public evidence",
        )
    ]


def test_mock_search_provider_respects_limit() -> None:
    hits = MockSearchProvider().search("public evidence", limit=0)

    assert hits == []
