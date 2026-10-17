"""The only way the orchestrator calls a provider.

``LLMClient.complete`` always goes through ``guard.call_with_corpus``: every call, with or
without corpus text, is budgeted (3,000 ``cl100k_base`` tokens) and logs one ``llm_call``
line with its ``corpus_tokens``. A call with no corpus text passes an empty
``CorpusContext`` and so logs ``corpus_tokens=0``.
"""
from __future__ import annotations

from app.llm import deadline
from app.llm.budget import CorpusContext
from app.llm.guard import GuardedCall, LLMRequest, call_with_corpus
from app.llm.providers.base import LLMUnavailable, MIN_CALL_SECONDS, Provider


class LLMClient:
    def __init__(self, provider: Provider) -> None:
        self.provider = provider

    def complete(
        self,
        *,
        purpose: str,
        request_id: str,
        system: str,
        messages: list[dict],
        corpus: CorpusContext | None = None,
        json_schema: dict | None = None,
    ) -> GuardedCall:
        """One guarded call. ``response`` of the result is the model's text.

        Raises ``LLMUnavailable`` (provider down or timed out after its one retry) and
        the guard's errors (a budget or system-prompt violation is a bug, never sent)."""

        left = deadline.remaining()
        if left is not None and left < MIN_CALL_SECONDS:
            raise LLMUnavailable("deadline")

        def send(request: LLMRequest) -> str:
            return self.provider.complete(request.system, list(request.messages), json_schema)

        return call_with_corpus(
            system,
            messages,
            corpus if corpus is not None else CorpusContext(),
            send,
            request_id=request_id,
            purpose=purpose,
        )
