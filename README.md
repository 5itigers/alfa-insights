# alfa-insights
insight analysis ai-service

## Index local documents

Install the project requirements in your virtual environment:

```bash
python -m pip install -r requirements.txt
```

The repository defaults to the existing PostgreSQL database
`AI-Service-vectordb` at `localhost:5432`, using role `postgres`. Chunks are
written to `public.document_chunks`; document version metadata is stored in
`public.rag_documents`. The database must already exist and the pgvector
extension must be installed on its server.

Supply the database password through `PGPASSWORD` in your shell or PyCharm run
configuration. No password is included in the default connection URL:

```bash
export PGPASSWORD='your-database-password'
```

To use another connection, set `DATABASE_URL`. A value passed explicitly to
`PostgresStore(database_url)` overrides both the environment and the default.
Python reads the environment rather than loading `.env` automatically.

For an optional, separate Docker Compose database instead:

```bash
docker compose up -d db
export DATABASE_URL=postgresql://rag:rag@localhost:55432/alfa_insights
```

The Compose credentials are local development defaults. To customize them,
copy `.env.example` to `.env`, update its Docker values, and export the matching
`DATABASE_URL` in your shell. Its default port `55432` leaves the existing local
PostgreSQL server on `5432` available.

Run from the repository root:

```bash
python -m src.rag.ingestion.indexing \
  "data/raw/Бонд_Бриф_14092026.pdf" \
  --source-type bank_report --published-at 2026-09-14 --init-db
```

Use `--init-db` on the first run to create the schema. The database role needs
permission to create the vector extension and tables. The same option is safe
to repeat. A source can be a single PDF, UTF-8 text/Markdown file, or a folder
containing one source type. Use `--source-type news` for articles. Keep the
hackathon brief outside the folders you index as bank research.

Indexing uses the existing loaders and chunker, then normalizes BGE-M3 vectors
and stores them in PostgreSQL. The first real run downloads the pinned BGE-M3
model weights (about 2.3 GB). The model is loaded once per run. Scanned PDFs
require Tesseract with Russian and English language data.

Each command prints a JSON result containing the source path, version ID,
chunk count, and whether the document was skipped. Unchanged files with the
same processing settings are skipped before OCR or model inference. Changes
to document contents, model revision, chunk settings, or source metadata create
a new document version. Existing versions and their citations remain stored.
Retrieval selects `rag_documents.is_current` and the query encoder's model name
and revision; historical versions can be addressed by `version_id` and `chunk_id`.

The existing `document_chunks` columns are used directly: `source_id` holds the
immutable document `version_id`, `content` holds the embedded text, and
`chunk_index` is a zero-based position across the entire document. `metadata`
preserves `chunk_id`, `page_number`, the original `page_chunk_index`, character
offsets, token count, and extraction method. Join `document_chunks.source_id`
to `rag_documents.version_id` to retrieve source details and model revisions.

Publication dates and source URLs are supplied explicitly with `--published-at`
and `--source-url`. Omitted publication dates remain unknown. Date-only CLI
values are stored at midnight UTC. `indexed_at` records when indexing ran.

The schema includes a Russian full-text GIN index. Dense vector search starts
with exact search; an approximate HNSW index can be added after measuring a
need. Changes to extraction/chunking behavior or OCR model files require a bump
to `PIPELINE_VERSION` so cached documents are reprocessed. Missing local files
are not automatically removed from the database.

## Retrieve evidence

After indexing documents, run a question against the same database:

```bash
python -m src.rag.retrieval "Как ключевая ставка влияет на стоимость кредитования?" \
  --source-type bank_report --top-k 5
```

For a publication window, add `--published-from 2026-09-01
--published-before 2026-10-01`. The lower bound is inclusive; the upper bound
is exclusive. Date-only CLI values and timestamps without an offset use UTC.
Documents with unknown publication dates are excluded when a date filter is
set. These filters apply to publication dates, not the date indexing ran, and
do not reconstruct which document versions existed at a historical instant.

Reuse a retriever in application code so the embedding model stays loaded:

```python
from datetime import datetime, timezone
from src.rag.retrieval import Retriever
from src.rag.storage import PostgresStore

retriever = Retriever(PostgresStore())
passages = retriever.retrieve(
    "Как ключевая ставка влияет на стоимость кредитования?",
    source_type="bank_report",
    published_before=datetime(2026, 10, 1, tzinfo=timezone.utc),
    top_k=5,
)
for passage in passages:
    print(passage.citation_id, passage.title, passage.page_number, passage.text)
```

The model loads on the first query. Importing `retrieval` or constructing a
`Retriever` does not load model weights or connect to PostgreSQL. The query is
embedded with the same pinned BGE-M3 model revision as indexing; overlong inputs
are rejected by the existing embedder rather than silently truncated. The
query requires the schema created by indexing with `--init-db`.

Search combines exact cosine similarity with PostgreSQL Russian full-text
search using reciprocal rank fusion: each matching rank contributes
`1 / (rrf_k + rank)`, with `rrf_k=60` by default. Each branch fetches up to
`candidate_limit=30` eligible chunks, and the combined ranking returns up to
`top_k=5`. Set the candidate limit to at least `top_k`. Search settings can be
overridden through the Python method or CLI flags.

Both branches filter current document versions, embedding model and revision,
source type, and publication window before selecting candidates. Set
`source_type="news"` for news evidence and `source_type="bank_report"` for bank
views; make separate calls when the answer needs both. Omitting this filter
searches both types. Chunks that have no matching document metadata are excluded.

Each result includes the text, immutable `version_id` and `chunk_id`, document
position, page, source path/URL/title/type/date, original character offsets,
token count, extraction method, fused `score`, and each branch's rank and score.
`citation_id` combines the version and chunk IDs, so citations can retain the
exact historical evidence after a document changes. The CLI prints a JSON
array, including citation IDs and ISO publication timestamps.

An empty eligible corpus returns `[]`. With eligible documents, vector search
can return weak matches even when no passage answers the question. The fused
score measures ranking agreement, not answer confidence. Check retrieved
passages against a few known questions before connecting generation.

Search operators follow the [pgvector documentation](https://github.com/pgvector/pgvector)
and [PostgreSQL full-text search documentation](https://www.postgresql.org/docs/17/textsearch-controls.html).
Use a UTF-8 database locale such as `C.UTF-8` or `ru_RU.UTF-8` for Russian
case handling. A plain `C`/`POSIX` locale can leave Cyrillic capital letters
unchanged, causing keyword searches to miss otherwise matching text.

## Generate an answer

`generation.py` takes a question and the `RetrievedChunk` objects returned by
retrieval. It sends one request to the organizer's `/chat/completions` endpoint
and returns a structured `GenerationResult` with a rendered `answer` property.
No additional Python dependencies are required.

Set the team's key in your local shell or PyCharm environment:

```bash
export RAG_LLM_API_KEY='your-team-api-key'
```

The default base URL is `https://changellenge.up.railway.app/v1` and the default
model is `qwen/qwen3.6-35b-a3b`. Optional environment overrides are
`RAG_LLM_BASE_URL` and `RAG_LLM_MODEL`. The backup URL is
`https://api.alfahack-gateway26.ru/v1`; select it explicitly when needed.
`.env.example` lists the variables, but Python does not load `.env` automatically.

Only these exact organizer-provided model IDs are accepted:

```text
google/gemma-4-26b-a4b-it
z-ai/glm-5.3-flash
qwen/qwen3.6-35b-a3b
deepseek/deepseek-v4.1-flash
qwen/qwen3-vl-30b-a3b-instruct
moonshotai/kimi-k2.6
```

Connect retrieval and generation from the repository root:

```python
from src.rag.generation import Generator
from src.rag.retrieval import Retriever

question = "Как изменение ключевой ставки влияет на стоимость кредитования?"
retriever = Retriever()
news = retriever.retrieve(question, source_type="news", top_k=3)
reports = retriever.retrieve(question, source_type="bank_report", top_k=3)
result = Generator().generate(question, [*news, *reports])

print(result.answer)
for source in result.sources:
    print(source.source_id, source.title, source.published_at,
          source.page_number, source.citation_id)
print(result.usage)
```

The response separates `facts` from news, `bank_views` from bank research, and
`implications` derived by the model. Every claim has `source_ids`, such as `S1`.
`result.sources` maps those IDs to the original text, immutable citation ID,
document version, chunk, title, URL/path, publication date, and page. Only cited
sources appear in the result. `missing_information` records evidence gaps;
`status` is `answered` or `insufficient_evidence`.

The prompt tells the model to use the provided evidence, distinguish dated bank
analysis from its own inferences, acknowledge unknown dates and conflicts, and
treat instructions inside source text as data. Source IDs and source types are
validated locally. This validates citation references, not whether the cited
text actually entails a claim; real-answer evaluation is still required.

Empty evidence returns `insufficient_evidence` without reading an API key or
making a paid request. Duplicate citation IDs are removed, whole passages are
packed in retrieval order, and conflicting copies of the same citation are
rejected. Default limits are 20,000 characters for the JSON user message,
4,000 characters for a question, 1,500 output tokens, and a 60-second network
timeout. The character limit is not a model-specific token count.
`GenerationConfig` accepts overrides for `max_context_chars`,
`max_output_tokens`, and `timeout_seconds`; `GenerationConfig.from_env()` reads
the key, model, and base URL. It accepts only the two organizer-provided gateways.

Invalid JSON, missing/unknown citations, incompatible source types, truncated
answers, and gateway errors raise `GenerationError`. Invalid inputs or settings
raise `ValueError`. There are no automatic retries, model changes, or gateway
fallbacks. `result.usage` retains token usage reported by the gateway; it does
not calculate ruble costs or enforce the team's shared daily budget.

## Tests

```bash
python -m unittest discover -s tests -v
```

Database tests run only when `RAG_TEST_DATABASE_URL` points to a disposable
PostgreSQL database with pgvector available and a UTF-8 locale. They use a deterministic fixture
encoder to test persistence, versioning, and hybrid retrieval without downloading
model weights. Retrieval tests cover ranking fusion, citations, source/date
filters, model compatibility, and exclusion of historical document versions.
Generation tests use mocked gateway responses: they check the exact model IDs,
request format, citation validation, context limits, abstention, and failure
handling without sending paid requests.
