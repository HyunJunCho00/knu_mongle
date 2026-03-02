### school_international_student_agent_service

#### Crawl Targets (school split)

Crawler loads targets from `src/crawl/schools/*.jsonl` first.  
If no school file exists, it falls back to `src/crawl/knu_master.jsonl`.

Recommended line schema:

```json
{"school_id":"knu","school_name":"Kyungpook National University","dept_id":"korean","dept_name":"국어국문학과","program_level":"undergrad","url":"https://home.knu.ac.kr/HOME/korean/sub.htm?nav_code=kor1657071357"}
```

#### Ingestion (local)

Run ingestion directly from crawled jsonl files:

```bash
python -m src.etl.ingestion --input data --collection school_notice --batch-size 16
```

Optional Groq metadata enrichment:

```bash
python -m src.etl.ingestion --input data --collection school_notice --enable-groq-metadata
```

#### Required environment variables

- `QDRANT_URL`
- `QDRANT_API_KEY`
- `CF_ACCOUNT_ID` (or `CLOUDFLARE_ACCOUNT_ID`)
- `CF_API_TOKEN` (or `CLOUDFLARE_API_TOKEN`)
- `GROQ_API_KEY` (optional)

#### GitHub Actions secrets

Set these repository secrets for `.github/workflows/ingest.yml`:

- `QDRANT_URL`
- `QDRANT_API_KEY`
- `CF_ACCOUNT_ID`
- `CF_API_TOKEN`
- `GROQ_API_KEY` (optional)

#### Retrieval pipeline (updated)

- Query decomposition for multi-intent questions
- Hybrid retrieval (dense + sparse)
- Rule-based reranking
- Query-time context packing (neighbor chunks)
- Evidence fields in response (`point_id`, `source_url`, `section_path`, `chunk_index`)

#### Attachment parsing policy (Korean docs)

- `.hwpx`: `python-hwpx` -> XML fallback (`lxml`)
- `.hwp`: `pyhwp(hwp5txt)` -> `hwp-extract` -> `olefile` fallback
- Image attachments: VLM fallback if text parser is not applicable
- Each attachment stores parser metadata: `parser_name`, `parser_version`, `parse_confidence`, `parse_error`, `extraction_method`
