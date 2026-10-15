"""A stand-in LLM provider for tests: records what it was sent, returns a canned reply.

It makes no network calls and logs nothing. Tests use ``received_text`` to prove
that something (a restricted document, a secret) never reached the provider.
"""
from __future__ import annotations

from collections.abc import Iterable


class FakeProvider:
    """Callable ``provider_fn`` for ``call_with_corpus``.

    ``reply`` is returned for every call, unless ``replies`` is given: then each
    call consumes the next scripted reply, and running out is an AssertionError
    (a test bug must not be hidden by a silent default). ``error`` is raised
    after the request is recorded, to simulate a provider outage.
    """

    def __init__(
        self,
        reply: object = "fake reply",
        *,
        replies: Iterable[object] | None = None,
        error: BaseException | None = None,
    ) -> None:
        self.reply = reply
        self.error = error
        self.calls: list = []
        self.replies_given: list = []
        self._script = list(replies) if replies is not None else None

    def __call__(self, request) -> object:
        self.calls.append(request)
        if self.error is not None:
            raise self.error
        if self._script is None:
            return self.reply
        if not self._script:
            raise AssertionError("FakeProvider was called more often than it has scripted replies")
        reply = self._script.pop(0)
        self.replies_given.append(reply)
        return reply

    @property
    def received_text(self) -> str:
        """Everything the provider was ever sent: system prompts and every message."""
        pieces: list[str] = []
        for request in self.calls:
            pieces.append(request.system)
            pieces.extend(message["content"] for message in request.messages)
        return "\n".join(pieces)
