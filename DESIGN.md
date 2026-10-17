# Design

## Ingestion (tables and the scanned memo OCR)

Startup (`app/ingestion/pipeline.py` `run_ingestion`) parses every manifest file, chunks it, rebuilds the hybrid index, then extracts holidays. A missing or unreadable file is skipped (`app/ingestion/corpus.py`); `/health` still becomes 200 if the rest succeed.

PDF pages (`app/ingestion/parsers/pdf_parser.py`) use pdfplumber. A page with fewer than 20 non-space text-layer characters is OCR’d: `ocr_pdf_page` in `app/ingestion/parsers/ocr.py` rasters it at 300 DPI (pdf2image / Poppler) and reads English with Tesseract. The decision is per page, never by filename. DOCX, HTML and Markdown have their own parsers. Tables become one line per data row with column names repeated (`render_table` in `app/ingestion/parsers/tables.py`) so a retrieved row still names its cells. Calendar extractors read the structured `(header, value)` pairs, not the rendered string.

## Chunking and retrieval

`chunk_document` (`app/ingestion/chunker.py`) targets 200–350 `cl100k_base` tokens (hard max 450). The section heading is part of the chunk text. Tables are never split mid-row; oversized prose splits on sentences with ~30-token overlap inside the same section. Tiny neighbours merge unless that would cross a table.

`HybridIndex` (`app/ingestion/index.py`) holds L2-normalised NumPy vectors (fastembed `BAAI/bge-small-en-v1.5`) and a `rank_bm25.BM25Okapi` model. Search requires `SearchFilters`. `_eligible_positions` drops unread / out-of-region / out-of-date chunks using metadata only; BM25 IDF and the vector slice are built from those rows, so a restricted chunk is never scored. Eligible BM25 and cosine ranks are fused with RRF (`k=60`). `retrieve_for_question` (`app/retrieval/pipeline.py`) asks for eight current and three upcoming hits; `select_diverse` keeps each document’s best chunk first.

Every LLM call goes through `LLMClient.complete` → `call_with_corpus` → `enforce_budget` (3,000 corpus tokens, `app/llm/budget.py` / `app/llm/guard.py`). The system prompt is checked for leaked corpus text. Citations are the intersection of used, sent, and readable `doc_id`s (`enforce_invariants`).

## Document precedence

Section 5 / B4 is Python, not model judgment.

1. `resolve_reference_date` (`app/retrieval/reference_date.py`) picks today, or a single past day / month / year named in the question (a past year means 31 December). Unclear dates `clarify`.
2. Eligibility (`is_eligible`, `can_read`, `covers_region`, `readable_access_levels` in `app/retrieval/policy_resolver.py`): role may read `access_level`; regions include the requester’s region or `GLOBAL`; `effective_date <=` reference. Ineligible claims cannot win, tie, or be reported.
3. Extraction may invent nothing: `build_claims` (`app/orchestrator/evidence.py`) keeps values that appear in the sent chunk and attaches manifest metadata from the chunk.
4. `resolve_topic` then applies authority (`policy, addendum, memo, handbook, calendar` beat `faq, wiki`), then scope (regional beats `GLOBAL`), then recency. A remaining tie with the same value is decided; different values are a conflict and both are shown.
5. `upcoming_change` may add at most three newer, readable, in-region winners, labelled upcoming, never current. `resolve_facts` calls both functions per topic.

## Access control

Documents: `readable_access_levels` + `HybridIndex._eligible_positions` / `search` before retrieval; `retrieve_for_question` passes role and region from the DB `Requester`. Identity is `get_employee` in `app/api/chat.py` `_authenticate` from `X-Employee-Id`. Message claims are ignored.

Leave data: `may_view_employee` and `is_manager` in `app/tools/access.py` (self; manager + `is_direct_report`; HR any one id). Called from `get_leave_balance`, `list_leave_requests`, `get_team_leave`. A forbidden id and a missing id take the same path; `refusal()` echoes no target id. Apply/cancel operate only on the requester. Bulk phrases are refused by `check_guards` (`app/orchestrator/guards.py`) before the analyzer.

## Personal data protection

Logs use employee ids, `request_id`, route, tool name/status, `doc_id`, token counts. `register_pii` loads names, emails and phones from the DB; every log line passes `redact`. `mask_email` is first character + `***` + domain. Phones are never shown. `scrub_visible_text` runs on every answer. `safe_args` rejects free text in tool args. Analyzer prompts send role, region, pending flags and the message — not names, emails or phones.

## Failure modes found and fixed

- **Mid-sentence page split:** pdfplumber ended a page mid-sentence. `_join_page_breaks` glues only when the next page starts lowercase prose (not a heading, table or bullet).
- **Paragraph packing:** joining units by raw token counts overflowed the hard max once separators were added. `_pack` / `_PACK_SAFETY_TOKENS` recompute the target from what is left.
- **`(cid:)` fonts:** unmapped glyphs became `(cid:N)`. A leading one is treated as a bullet; others are left and counted, not deleted.
- **Calendar kind:** a table under a “holidays” heading is not automatically public. `classify_kind` uses the table heading, then caption; conflict or silence is `unknown` and does not count as a working-day holiday.
- **Confirmation race:** two “yes” messages could both see the same proposal. `claim_proposal` deletes the row inside `BEGIN IMMEDIATE`; only one caller executes. Concurrent apply/cancel tests lock the same way.

## What breaks first at 10,000 documents

The index is rebuilt from scratch every startup: every chunk is re-embedded and a full `BM25Okapi` is built. Embed time and RSS for the in-memory vector matrix grow linearly; BM25 tokenisation and the per-filter BM25 cache (`_BM25_CACHE_SIZE = 64`) do too. `select_diverse` is cheap (one pass over the ranked pool). Ten thousand documents would make startup the first thing to miss the 10-minute `/health` budget, then memory.

Fix: persist embeddings keyed by content hash and ingest incrementally; put vectors in a real vector DB (or memory-map `vectors.npy`); keep BM25 per collection or refresh it lazily; drop the full rebuild. Diversity can stay in process — it is not the bottleneck.
