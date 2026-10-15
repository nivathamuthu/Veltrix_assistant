"""The chat orchestrator: deterministic guards, dates, confirmation and routing around the
(Phase 7b) LLM. Everything here that touches permissions, writes or precedence is plain
Python; the LLM only ever understands a message or phrases an answer."""
