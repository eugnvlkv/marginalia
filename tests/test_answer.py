from types import SimpleNamespace

import pytest

from marginalia.answer import Answer, answer_question, format_excerpts
from marginalia.index import Hit

HITS = [
    Hit(chunk_id="b:2", book_id="b", heading_path=("Risk",), text="Never risk more than 2%.", score=0.9),
    Hit(chunk_id="b:0", book_id="b", heading_path=(), text="A short book.", score=0.5),
]


class FakeClient:
    """Stands in for anthropic.Anthropic: records the request, returns a canned answer. No network, no cost."""

    def __init__(self, parsed_output):
        self.request = None
        self.messages = self  # so that client.messages.parse(...) lands in self.parse

        self._response = SimpleNamespace(
            parsed_output=parsed_output,
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=100, output_tokens=20),
        )

    def parse(self, **request):
        self.request = request
        return self._response


def test_excerpts_carry_ids_and_paths():
    text = format_excerpts(HITS)

    assert "[b:2] Risk\nNever risk more than 2%." in text
    assert "[b:0] \nA short book." in text


def test_answer_question_sends_excerpts_and_question_and_returns_the_answer():
    canned = Answer(answer="Не больше 2%.", sources=["b:2"], not_in_books=False)
    client = FakeClient(canned)

    answer, usage = answer_question(client, "сколько рисковать?", HITS)

    assert answer == canned
    assert usage.input_tokens == 100
    content = client.request["messages"][0]["content"]
    assert "[b:2] Risk" in content
    assert content.endswith("Question: сколько рисковать?")
    assert client.request["output_format"] is Answer


def test_missing_structured_answer_raises():
    client = FakeClient(parsed_output=None)

    with pytest.raises(RuntimeError, match="stop_reason=end_turn"):
        answer_question(client, "q", HITS)
