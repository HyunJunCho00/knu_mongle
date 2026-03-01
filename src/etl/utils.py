import hashlib
import json
import re
import uuid
from pathlib import Path
from typing import Dict, Iterable, Iterator, List


def normalize_whitespace(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def build_bm25_text(title: str, section_header: str, chunk_text: str) -> str:
    parts = [
        normalize_whitespace(title),
        normalize_whitespace(section_header),
        normalize_whitespace(chunk_text),
    ]
    return " ".join([p for p in parts if p]).strip()


def deterministic_point_id(fields: List[str]) -> str:
    seed = "||".join([str(v or "").strip() for v in fields])
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()
    return str(uuid.UUID(digest[:32]))


def deterministic_uuid(fields: List[str], separator: str = "|") -> str:
    seed = separator.join([str(v or "").strip() for v in fields])
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()
    return str(uuid.UUID(digest[:32]))


def classify_block_type(section_header: str, text: str) -> str:
    source = f"{section_header or ''} {text or ''}".lower()
    rules = [
        ("deadline", [r"deadline", r"due\s*date", r"\ub9c8\uac10", r"\uae30\ud55c", r"\uae4c\uc9c0"]),
        (
            "eligibility",
            [r"eligib", r"\ub300\uc0c1", r"\uc790\uaca9", r"\uc9c0\uc6d0\s*\uc790\uaca9", r"\uc2e0\uccad\s*\uc790\uaca9"],
        ),
        (
            "required_docs",
            [
                r"required\s*doc",
                r"\uc81c\ucd9c\s*\uc11c\ub958",
                r"\uad6c\ube44\s*\uc11c\ub958",
                r"\ud544\uc694\s*\uc11c\ub958",
            ],
        ),
        ("procedure", [r"procedure", r"step", r"\uc2e0\uccad\s*\ubc29\ubc95", r"\uc808\ucc28", r"\ubc29\ubc95"]),
        (
            "contact",
            [r"contact", r"email", r"\ubb38\uc758", r"\uc5f0\ub77d\ucc98", r"\ub2f4\ub2f9\uc790", r"\uc804\ud654"],
        ),
        ("fee", [r"fee", r"\uc218\uc218\ub8cc", r"\ube44\uc6a9", r"\ub4f1\ub85d\uae08", r"\ub0a9\ubd80"]),
        ("policy", [r"policy", r"\uc720\uc758\uc0ac\ud56d", r"\uc548\ub0b4\uc0ac\ud56d", r"\uaddc\uc815", r"\uc815\ucc45"]),
        ("attachment_summary", [r"attachment", r"\ucca8\ubd80", r"\ubd99\uc784"]),
    ]

    for block_type, patterns in rules:
        for pattern in patterns:
            if re.search(pattern, source):
                return block_type
    return "general"


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 120) -> List[str]:
    if not text:
        return []

    cleaned = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not cleaned:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", cleaned) if p.strip()]
    if not paragraphs:
        paragraphs = [cleaned]

    chunks: List[str] = []
    current = ""
    for para in paragraphs:
        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current)
            tail = current[-overlap:] if overlap > 0 else ""
            current = f"{tail}\n\n{para}".strip() if tail else para
        else:
            for i in range(0, len(para), chunk_size):
                piece = para[i : i + chunk_size]
                if piece:
                    chunks.append(piece)
            current = ""

    if current:
        chunks.append(current)

    return [c for c in chunks if c and len(c.strip()) >= 10]


def iter_metadata_files(input_dir: str) -> Iterator[Path]:
    root = Path(input_dir)
    if not root.exists():
        return iter(())
    return (p for p in root.rglob("*_metadata.jsonl") if p.is_file())


def iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue
