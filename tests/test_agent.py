import pytest
from fakes import ScriptedClient, final, response, tool_use

from marginalia.agent import build_agent, run_agent, tool_calls
from marginalia.answer import Answer

# The fixtures book_collection (the 4-chunk test book) and embedder come from conftest.py;
# the fake Claude client comes from fakes.py.

FINAL = Answer(answer="Не больше, чем можете себе позволить.", sources=["b:2"], not_in_books=False)


def final_answer():
    return final(FINAL)


def test_agent_searches_then_answers(book_collection, embedder):
    client = ScriptedClient(
        [response([tool_use("t1", "search_docs", {"query": "risk"})], "tool_use"), final_answer()]
    )

    state = run_agent(build_agent(client, book_collection, embedder), "сколько рисковать?")

    assert state["answer"] == FINAL
    assert tool_calls(state) == [("search_docs", {"query": "risk"})]
    assert len(state["usage"]) == 2
    # The second request carries the search result back to the model, tied to the tool call id.
    result = client.requests[1]["messages"][-1]["content"][0]
    assert result["tool_use_id"] == "t1"
    assert '[b:2] book_id=b heading_path=["Risk"]' in result["content"]
    assert result["is_error"] is False


def test_get_section_tool_returns_the_whole_section(book_collection, embedder):
    client = ScriptedClient(
        [
            response([tool_use("t1", "get_section", {"book_id": "b", "heading_path": ["Risk"]})], "tool_use"),
            final_answer(),
        ]
    )

    run_agent(build_agent(client, book_collection, embedder), "q")

    result = client.requests[1]["messages"][-1]["content"][0]
    assert result["content"] == "[b:2] Risk\nNever risk more than you can afford to risk."


def test_unknown_section_is_reported_to_the_model_as_a_tool_error(book_collection, embedder):
    client = ScriptedClient(
        [
            response([tool_use("t1", "get_section", {"book_id": "b", "heading_path": ["Nope"]})], "tool_use"),
            final_answer(),
        ]
    )

    run_agent(build_agent(client, book_collection, embedder), "q")

    result = client.requests[1]["messages"][-1]["content"][0]
    assert result["is_error"] is True
    assert "no section" in result["content"]


def test_two_tool_calls_in_one_turn_are_answered_in_one_message(book_collection, embedder):
    client = ScriptedClient(
        [
            response(
                [tool_use("t1", "search_docs", {"query": "risk"}), tool_use("t2", "search_docs", {"query": "trend"})],
                "tool_use",
            ),
            final_answer(),
        ]
    )

    run_agent(build_agent(client, book_collection, embedder), "q")

    results = client.requests[1]["messages"][-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["t1", "t2"]


def test_agent_gives_up_after_max_model_calls(book_collection, embedder):
    endless = [response([tool_use(f"t{i}", "search_docs", {"query": "risk"})], "tool_use") for i in range(10)]
    client = ScriptedClient(endless)

    with pytest.raises(RuntimeError, match="within 3 model calls"):
        run_agent(build_agent(client, book_collection, embedder), "q", max_model_calls=3)

    assert len(client.requests) == 3
