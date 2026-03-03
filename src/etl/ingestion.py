import argparse
import os
import random
import re
import time
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from qdrant_client import QdrantClient, models

from src.core.config import settings
from src.etl.encoders import ConditionalDenseEncoder, ConditionalMetadataEnricher
from src.etl.utils import (
    build_bm25_text,
    chunk_text,
    classify_block_type,
    deterministic_uuid,
    iter_jsonl,
    iter_metadata_files,
    normalize_whitespace,
)


@dataclass
class Chunk:
    text: str
    level: str
    block_type: str
    chunk_type: str
    section_header: str
    node_path: str
    parent_path: str
    node_type: str
    parent_index: int
    chunk_index: int
    total_chunks: int
    parent_text: str


class KoreanNoticeChunker:
    def __init__(self, chunk_size: int = 900):
        self.chunk_size = chunk_size

    @staticmethod
    def _split_sections(text: str) -> List[Tuple[int, str, str]]:
        lines = text.split("\n")
        sections: List[Tuple[str, List[str]]] = []
        current_header = "본문"
        current_lines: List[str] = []
        header_pattern = re.compile(
            r"^\s*(\d+[.)]|[가-힣A-Za-z]{1,20}\s*[:：]|[■◆▶●]+|[0-9]+\s*-\s*)"
        )

        for line in lines:
            ln = line.strip()
            if not ln:
                continue
            is_header = len(ln) <= 60 and bool(header_pattern.match(ln))
            if is_header:
                if current_lines:
                    sections.append((current_header, current_lines))
                current_header = ln[:60]
                current_lines = []
            else:
                current_lines.append(ln)

        if current_lines:
            sections.append((current_header, current_lines))

        if not sections:
            sections = [("본문", [text])]

        return [
            (s_idx, header, "\n".join(body_lines))
            for s_idx, (header, body_lines) in enumerate(sections)
        ]

    @staticmethod
    def _detect_node_type(text: str) -> str:
        s = text.strip()
        if re.match(r"^(\-|\*|\d+[.)])\s+", s):
            return "list_item"
        if "|" in s and len(s) < 500:
            return "table_row"
        return "paragraph"

    def chunk(self, text: str, title: str = "") -> List[Chunk]:
        sections = self._split_sections(text)
        chunks: List[Chunk] = []

        for s_idx, header, section_text in sections:
            section_header = header or (title[:60] if title else "본문")
            section_text_for_parent = section_text.strip() or section_header
            pieces = chunk_text(section_text_for_parent, chunk_size=self.chunk_size, overlap=120)
            if not pieces:
                pieces = [section_text_for_parent]

            parent_block_type = classify_block_type(section_header, section_text_for_parent)
            chunks.append(
                Chunk(
                    text=section_text_for_parent,
                    level="parent",
                    block_type=parent_block_type,
                    chunk_type="parent",
                    section_header=section_header,
                    node_path=f"s{s_idx}",
                    parent_path=f"s{s_idx}",
                    node_type="section",
                    parent_index=s_idx,
                    chunk_index=0,
                    total_chunks=len(pieces),
                    parent_text=section_text_for_parent,
                )
            )

            for p_idx, piece in enumerate(pieces):
                node_type = self._detect_node_type(piece)
                block_type = classify_block_type(section_header, piece)
                chunks.append(
                    Chunk(
                        text=piece,
                        level="child",
                        block_type=block_type,
                        chunk_type=node_type,
                        section_header=section_header,
                        node_path=f"s{s_idx}.n{p_idx}",
                        parent_path=f"s{s_idx}",
                        node_type=node_type,
                        parent_index=s_idx,
                        chunk_index=p_idx,
                        total_chunks=len(pieces),
                        parent_text=section_text_for_parent,
                    )
                )

        return chunks


class QdrantIngestor:
    def __init__(
        self,
        input_dir: str,
        collection_name: str,
        batch_size: int,
        enable_metadata: bool,
        qdrant_timeout: float,
        upsert_max_retries: int,
    ):
        self.input_dir = input_dir
        self.collection_name = collection_name
        self.batch_size = batch_size
        self.qdrant_timeout = qdrant_timeout
        self.upsert_max_retries = upsert_max_retries
        self.upsert_base_delay_seconds = max(settings.QDRANT_UPSERT_BASE_DELAY_SECONDS, 0.1)
        self.chunker = KoreanNoticeChunker(chunk_size=settings.CHUNK_SIZE)

        cf_account_id = os.getenv("CF_ACCOUNT_ID") or settings.CLOUDFLARE_ACCOUNT_ID
        cf_api_token = os.getenv("CF_API_TOKEN") or settings.CLOUDFLARE_API_TOKEN
        self.dense_encoder = ConditionalDenseEncoder(
            cf_account_id=cf_account_id,
            cf_api_token=cf_api_token,
        )
        print(f"[INFO] Dense encoder backend={self.dense_encoder.backend}")

        self.metadata_enricher: Optional[ConditionalMetadataEnricher] = None
        if enable_metadata:
            self.metadata_enricher = ConditionalMetadataEnricher(groq_api_key=settings.GROQ_API_KEY)
            print(f"[INFO] Metadata enricher backend={self.metadata_enricher.backend}")

        self.client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            timeout=self.qdrant_timeout,
            cloud_inference=True,
        )
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        quantization = models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8,
                quantile=0.99,
                always_ram=True,
            )
        )

        if self.client.collection_exists(self.collection_name):
            try:
                self.client.update_collection(
                    collection_name=self.collection_name,
                    quantization_config=quantization,
                )
                print("[INFO] Applied Qdrant scalar int8 quantization to existing collection")
            except Exception as exc:
                print(f"[WARN] Could not update quantization for existing collection: {exc}")
            return
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense": models.VectorParams(size=1024, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={"sparse": models.SparseVectorParams()},
            quantization_config=quantization,
        )
        print("[INFO] Created collection with Qdrant scalar int8 quantization")

    @staticmethod
    def _first_non_empty(row: Dict, keys: List[str], default: str = "") -> str:
        for key in keys:
            value = row.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        return default

    @staticmethod
    def _safe_int(value: object, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _resolve_content(self, row: Dict) -> str:
        normalized = row.get("normalized", {})
        if isinstance(normalized, dict):
            content = normalize_whitespace(normalized.get("content", ""))
            if content:
                return content
        return normalize_whitespace(row.get("content", ""))

    def _resolve_title(self, row: Dict) -> str:
        normalized = row.get("normalized", {})
        if isinstance(normalized, dict):
            title = normalize_whitespace(normalized.get("title", ""))
            if title:
                return title
        return normalize_whitespace(row.get("title", ""))

    def _resolve_date(self, row: Dict) -> str:
        normalized = row.get("normalized", {})
        if isinstance(normalized, dict):
            published_at = str(normalized.get("published_at", "")).strip()
            if published_at:
                return published_at
        return self._first_non_empty(row, ["published_at", "date"], "")

    @staticmethod
    def _resolve_attachments(row: Dict) -> List[Dict]:
        attachments = row.get("attachments", [])
        assets = row.get("assets", {})
        if isinstance(assets, dict) and isinstance(assets.get("attachments"), list):
            attachments = assets.get("attachments", [])
        return attachments if isinstance(attachments, list) else []

    def _base_payload(self, row: Dict) -> Dict:
        school_id = self._first_non_empty(row, ["school_id", "school"], "")
        dept_id = self._first_non_empty(row, ["dept_id", "dept"], "")
        title = self._resolve_title(row)
        date_value = self._resolve_date(row)
        attachments = self._resolve_attachments(row)

        return {
            "doc_id": self._first_non_empty(row, ["doc_id"], ""),
            "domain": self._first_non_empty(row, ["domain"], "notice"),
            "source_type": self._first_non_empty(row, ["source_type"], ""),
            "version": self._safe_int(row.get("version", 1), default=1),
            "is_current": bool(row.get("is_current", True)),
            "school_id": school_id,
            "dept_id": dept_id,
            "program_level": self._first_non_empty(row, ["program_level"], "all"),
            "url": self._first_non_empty(row, ["canonical_url", "url"], ""),
            "title": title,
            "date": date_value,
            "valid_until": self._first_non_empty(row, ["valid_until"], ""),
            "is_expired": bool(row.get("is_expired", False)),
            "requires_action": bool(row.get("requires_action", False)),
            "deadline_confidence": float(row.get("deadline_confidence", 0.0) or 0.0),
            "has_attachments": bool(attachments),
            "attachment_count": len(attachments),
            # Backward compatibility key for retriever filter
            "dept": dept_id,
            "school": school_id,
        }

    @staticmethod
    def _normalize_deadline_datetime(raw_value: str) -> Optional[str]:
        value = str(raw_value or "").strip()
        if not value:
            return None
        value = value.replace(".", "-").replace("/", "-")
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"

        try:
            dt = datetime.fromisoformat(value)
            return dt.strftime("%Y-%m-%dT%H:%M")
        except ValueError:
            pass

        match = re.search(
            r"(20\d{2})-(0[1-9]|1[0-2])-([0-2]\d|3[01])(?:\s+|T)?([0-2]\d):([0-5]\d)",
            value,
        )
        if match:
            y, m, d, hh, mm = match.groups()
            return f"{y}-{m}-{d}T{hh}:{mm}"

        date_only = re.search(r"(20\d{2})-(0[1-9]|1[0-2])-([0-2]\d|3[01])", value)
        if date_only:
            y, m, d = date_only.groups()
            return f"{y}-{m}-{d}T23:59"
        return None

    @staticmethod
    def _deadline_evidence_from_content(content: str) -> str:
        text = str(content or "")
        if not text:
            return ""
        keyword_match = re.search(r"(마감|기한|제출|접수|deadline|due)", text, flags=re.IGNORECASE)
        date_match = re.search(
            r"(20\d{2}[./-](0[1-9]|1[0-2])[./-]([0-2]\d|3[01])(?:\s*[T ]\s*[0-2]\d:[0-5]\d)?)",
            text,
            flags=re.IGNORECASE,
        )
        anchor = None
        if keyword_match:
            anchor = keyword_match.start()
        elif date_match:
            anchor = date_match.start()
        if anchor is None:
            return ""
        left = max(anchor - 70, 0)
        right = min(anchor + 90, len(text))
        return text[left:right].strip()

    def _derive_deadline_fields(self, row: Dict) -> Dict:
        deadlines_raw = row.get("deadlines", [])
        if not isinstance(deadlines_raw, list):
            deadlines_raw = []

        normalized_deadlines: List[Dict[str, str]] = []
        valid_until_dt: Optional[datetime] = None

        for item in deadlines_raw:
            label = ""
            raw_dt = ""
            if isinstance(item, dict):
                label = str(item.get("label", "")).strip()
                raw_dt = str(item.get("datetime", "")).strip()
            elif isinstance(item, str):
                raw_dt = item.strip()
            normalized_dt = self._normalize_deadline_datetime(raw_dt)
            if not normalized_dt:
                continue
            normalized_deadlines.append({"label": label[:30], "datetime": normalized_dt})
            try:
                dt_obj = datetime.strptime(normalized_dt, "%Y-%m-%dT%H:%M")
                if valid_until_dt is None or dt_obj > valid_until_dt:
                    valid_until_dt = dt_obj
            except ValueError:
                continue

        existing_valid_until = self._normalize_deadline_datetime(str(row.get("valid_until", "")))
        if existing_valid_until:
            try:
                existing_dt = datetime.strptime(existing_valid_until, "%Y-%m-%dT%H:%M")
                if valid_until_dt is None or existing_dt > valid_until_dt:
                    valid_until_dt = existing_dt
            except ValueError:
                pass

        valid_until = valid_until_dt.strftime("%Y-%m-%dT%H:%M") if valid_until_dt else ""
        is_expired = bool(valid_until_dt and valid_until_dt < datetime.now())

        existing_confidence = row.get("deadline_confidence")
        if existing_confidence is not None and str(existing_confidence).strip() != "":
            try:
                confidence = float(existing_confidence)
            except (TypeError, ValueError):
                confidence = 0.0
        else:
            if normalized_deadlines:
                confidence = 0.9 if len(normalized_deadlines) >= 2 else 0.8
            elif row.get("requires_action"):
                confidence = 0.25
            else:
                confidence = 0.0

        evidence_text = str(row.get("evidence_text", "") or "").strip()
        if not evidence_text:
            if normalized_deadlines:
                evidence_text = " | ".join(
                    [
                        f"{dl.get('label', '')} {dl.get('datetime', '')}".strip()
                        for dl in normalized_deadlines[:3]
                    ]
                ).strip()
            if not evidence_text:
                evidence_text = self._deadline_evidence_from_content(self._resolve_content(row))

        return {
            "deadlines": normalized_deadlines if normalized_deadlines else deadlines_raw,
            "valid_until": valid_until,
            "is_expired": is_expired,
            "deadline_confidence": max(0.0, min(1.0, confidence)),
            "evidence_text": evidence_text[:500],
        }

    def _build_points(self, rows: List[Tuple[Dict, Chunk]]) -> List[models.PointStruct]:
        texts = [chunk.text for _, chunk in rows]
        dense_vectors = self.dense_encoder.encode(texts)
        points: List[models.PointStruct] = []

        for idx, (row, chunk) in enumerate(rows):
            payload = self._base_payload(row)
            doc_id = payload.get("doc_id", "") or deterministic_uuid(
                [payload["url"], payload["date"], payload["dept_id"], payload["title"]],
                separator="|",
            )
            parent_id = deterministic_uuid(
                [doc_id, chunk.section_header, str(chunk.parent_index)],
                separator="|",
            )
            block_id = deterministic_uuid(
                [parent_id, chunk.block_type, str(chunk.chunk_index), chunk.level],
                separator="|",
            )

            payload.update(
                {
                    "doc_id": doc_id,
                    "block_id": block_id,
                    "level": chunk.level,
                    "parent_id": parent_id if chunk.level == "child" else "",
                    "content": chunk.text,
                    "block_type": chunk.block_type,
                    "chunk_type": chunk.chunk_type,
                    "node_type": chunk.node_type,
                    "node_path": chunk.node_path,
                    "parent_path": chunk.parent_path,
                    "section_header": chunk.section_header,
                    "chunk_index": chunk.chunk_index,
                }
            )

            bm25_text = build_bm25_text(
                title=payload["title"],
                section_header=chunk.section_header,
                chunk_text=chunk.text,
            )
            payload["bm25_text"] = bm25_text

            dense_vec = dense_vectors[idx] if idx < len(dense_vectors) else [0.0] * 1024

            points.append(
                models.PointStruct(
                    id=block_id,
                    vector={
                        "dense": dense_vec,
                        "sparse": models.Document(text=bm25_text, model="qdrant/bm25"),
                    },
                    payload=payload,
                )
            )
        return points

    @staticmethod
    def _iter_error_chain(exc: Exception):
        seen = set()
        current = exc
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            yield current
            current = getattr(current, "__cause__", None) or getattr(current, "__context__", None)

    def _status_code_from_error(self, exc: Exception) -> Optional[int]:
        for err in self._iter_error_chain(exc):
            status_code = getattr(err, "status_code", None)
            if isinstance(status_code, int):
                return status_code

            response = getattr(err, "response", None)
            if response is not None:
                response_code = getattr(response, "status_code", None)
                if isinstance(response_code, int):
                    return response_code

        message = str(exc)
        if "429" in message:
            return 429
        for code in [500, 502, 503, 504]:
            if str(code) in message:
                return code
        return None

    def _is_retryable_error(self, exc: Exception) -> bool:
        status_code = self._status_code_from_error(exc)
        if status_code == 429 or (status_code is not None and 500 <= status_code < 600):
            return True

        timeout_markers = ("timeout", "timed out", "temporarily unavailable", "connection reset")
        for err in self._iter_error_chain(exc):
            name = err.__class__.__name__.lower()
            msg = str(err).lower()
            if isinstance(err, (ConnectionError, TimeoutError, OSError)):
                return True
            if any(marker in name or marker in msg for marker in timeout_markers):
                return True
        return False

    def _upsert_with_retry(self, points: List[models.PointStruct]) -> None:
        max_retries = self.upsert_max_retries
        for attempt in range(max_retries + 1):
            try:
                self.client.upsert(collection_name=self.collection_name, points=points, wait=False)
                return
            except Exception as exc:
                status_code = self._status_code_from_error(exc)
                is_retryable = self._is_retryable_error(exc)
                if not is_retryable or attempt == max_retries:
                    raise

                sleep_seconds = min((2 ** attempt) * self.upsert_base_delay_seconds, 12.0)
                sleep_seconds += random.uniform(0.0, 0.25)
                print(
                    f"[WARN] Upsert retry attempt={attempt + 1}/{max_retries} "
                    f"status={status_code} sleep={sleep_seconds:.2f}s "
                    f"error={exc.__class__.__name__}"
                )
                time.sleep(sleep_seconds)

    def _row_doc_id(self, row: Dict) -> str:
        existing = str(row.get("doc_id", "")).strip()
        if existing:
            return existing
        return deterministic_uuid(
            [
                self._first_non_empty(row, ["canonical_url", "url"], ""),
                self._resolve_date(row),
                self._first_non_empty(row, ["dept_id", "dept"], ""),
                self._resolve_title(row),
            ],
            separator="|",
        )

    def _iter_latest_rows(self, file_path) -> List[Dict]:
        latest: Dict[str, Dict[str, object]] = {}
        for order, row in enumerate(iter_jsonl(file_path)):
            doc_id = self._row_doc_id(row)
            version = self._safe_int(row.get("version", 1), default=1)
            prev = latest.get(doc_id)
            if (not prev) or version > int(prev["version"]) or (
                version == int(prev["version"]) and order > int(prev["order"])
            ):
                latest[doc_id] = {"version": version, "order": order, "row": row}

        selected = sorted(latest.values(), key=lambda item: int(item["order"]))
        return [item["row"] for item in selected]

    def run(self) -> None:
        files = sorted(list(iter_metadata_files(self.input_dir)))
        if not files:
            print(f"[INFO] No metadata files found under: {self.input_dir}")
            return

        total_docs = 0
        total_chunks = 0

        for file_path in files:
            print(f"[INFO] Processing: {file_path}")
            batch_rows: List[Tuple[Dict, Chunk]] = []

            for row in self._iter_latest_rows(file_path):
                title = self._resolve_title(row)
                content = self._resolve_content(row) or title
                date = self._resolve_date(row)
                row["content"] = content

                if self.metadata_enricher and not row.get("summary"):
                    enriched = self.metadata_enricher.enrich(title=title, content=content, date=date)
                    if enriched:
                        row["summary"] = row.get("summary") or enriched.get("summary", "")
                        row["deadlines"] = row.get("deadlines") or enriched.get("deadlines", [])
                        row["requires_action"] = row.get(
                            "requires_action", enriched.get("requires_action", False)
                        )
                        row["contact"] = row.get("contact") or enriched.get("contact", "")
                        row["category"] = row.get("category") or enriched.get("category", "")
                        row["target_group"] = row.get("target_group") or enriched.get("target_group", "")
                        row["valid_until"] = row.get("valid_until") or enriched.get("valid_until", "")
                        row["deadline_confidence"] = (
                            row.get("deadline_confidence")
                            if row.get("deadline_confidence") not in (None, "")
                            else enriched.get("deadline_confidence", 0.0)
                        )
                        row["evidence_text"] = row.get("evidence_text") or enriched.get("evidence_text", "")

                row.update(self._derive_deadline_fields(row))

                chunks = self.chunker.chunk(content, title=title)
                if not chunks:
                    continue

                total_docs += 1
                for chunk in chunks:
                    batch_rows.append((row, chunk))
                    if len(batch_rows) >= self.batch_size:
                        points = self._build_points(batch_rows)
                        self._upsert_with_retry(points)
                        total_chunks += len(points)
                        print(f"[INFO] Upserted batch chunks: {len(points)} (total={total_chunks})")
                        batch_rows = []

            if batch_rows:
                points = self._build_points(batch_rows)
                self._upsert_with_retry(points)
                total_chunks += len(points)
                print(f"[INFO] Upserted final file batch: {len(points)} (total={total_chunks})")

        print(f"[DONE] docs={total_docs}, chunks={total_chunks}, collection={self.collection_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Metadata ingestion to Qdrant")
    parser.add_argument("--input", default="data", help="Input directory with crawled .jsonl files")
    parser.add_argument(
        "--collection",
        default=settings.COLLECTION_NAME,
        help="Qdrant collection name",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Upsert batch size")
    parser.add_argument(
        "--qdrant-timeout",
        type=float,
        default=settings.QDRANT_TIMEOUT_SECONDS,
        help="Qdrant HTTP timeout in seconds",
    )
    parser.add_argument(
        "--upsert-max-retries",
        type=int,
        default=settings.QDRANT_UPSERT_MAX_RETRIES,
        help="Maximum retry count for upsert on retryable errors",
    )
    parser.add_argument(
        "--enable-metadata",
        dest="enable_metadata",
        action="store_true",
        default=None,
        help="Enable LLM metadata enrichment (GPU: local Qwen, non-GPU: Groq)",
    )
    parser.add_argument(
        "--disable-metadata",
        dest="enable_metadata",
        action="store_false",
        help="Disable LLM metadata enrichment",
    )
    parser.add_argument(
        "--enable-groq-metadata",
        dest="enable_metadata",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def _is_gpu_server() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def main() -> None:
    args = parse_args()
    gpu_available = _is_gpu_server()
    enable_metadata = args.enable_metadata if args.enable_metadata is not None else gpu_available
    print(f"[INFO] Metadata enrichment enabled={enable_metadata} (gpu={gpu_available})")
    ingestor = QdrantIngestor(
        input_dir=args.input,
        collection_name=args.collection,
        batch_size=args.batch_size,
        enable_metadata=enable_metadata,
        qdrant_timeout=args.qdrant_timeout,
        upsert_max_retries=max(0, args.upsert_max_retries),
    )
    ingestor.run()


if __name__ == "__main__":
    main()
