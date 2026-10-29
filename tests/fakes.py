"""Fake Claude responses for tests: no network, no cost."""

from types import SimpleNamespace


def tool_use(id, name, input):
    return SimpleNamespace(type="tool_use", id=id, name=name, input=input)


def response(content, stop_reason, parsed_output=None):
    return SimpleNamespace(
        content=content,
        stop_reason=stop_reason,
        parsed_output=parsed_output,
        usage=SimpleNamespace(input_tokens=100, output_tokens=10),
    )


def final(answer):
    """The model's last turn: a structured answer, no more tool calls."""
    return response([SimpleNamespace(type="text", text="{...}")], "end_turn", parsed_output=answer)


class ScriptedClient:
    """Plays a fixed list of model responses, one per call, and records every request."""

    def __init__(self, responses):
        self.messages = self  # so that client.messages.parse(...) lands in self.parse
        self._responses = iter(responses)
        self.requests = []

    def parse(self, **request):
        self.requests.append({**request, "messages": list(request["messages"])})
        return next(self._responses)
