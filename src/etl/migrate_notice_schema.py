import argparse
import datetime
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from urllib.parse import urlsplit, urlunsplit


def utc_now_iso() -> str:
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def canonicalize_url(url: str) -> str:
    raw = str(url or "").strip()
    if not raw:
        return ""
    parts = urlsplit(raw)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, ""))


def normalize_date(value: str) -> str:
    raw = str(value or "").strip().replace(".", "-").replace("/", "-")
    # Keep as-is when exact yyyy-mm-dd is not present.
    for i in range(0, max(len(raw) - 9, 1)):
        token = raw[i : i + 10]
        if len(token) == 10 and token[4] == "-" and token[7] == "-":
            yyyy = token[:4]
            mm = token[5:7]
            dd = token[8:10]
            if yyyy.isdigit() and mm.isdigit() and dd.isdigit():
                return token
    return raw


def build_doc_id(school_id: str, dept_id: str, canonical_url: str) -> str:
    seed = f"{school_id}|{dept_id}|{canonical_url}"
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()


def build_content_hash(
    title: str,
    published_at: str,
    canonical_url: str,
    content: str,
    attachments: List[Dict],
    images: List[Dict],
) -> str:
    compact_attachments = [
        {
            "name": a.get("name", ""),
            "url": a.get("url", ""),
            "sha256": a.get("sha256", ""),
            "size": a.get("size", 0),
            "status": a.get("status", ""),
        }
        for a in attachments
    ]
    compact_images = [
        {
            "url": i.get("url", ""),
            "sha256": i.get("sha256", ""),
            "size": i.get("size", 0),
            "status": i.get("status", ""),
        }
        for i in images
    ]
    payload = {
        "title": title,
        "published_at": published_at,
        "canonical_url": canonical_url,
        "content": content,
        "attachments": compact_attachments,
        "images": compact_images,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


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


def to_notice_envelope(row: Dict) -> Dict:
    title = str(row.get("title", "") or "").strip()
    content = str(row.get("content", "") or "").strip()
    date_raw = str(row.get("date", "") or "").strip()
    published_at = normalize_date(str(row.get("published_at", "") or date_raw))
    url = str(row.get("url", "") or "").strip()
    canonical_url = canonicalize_url(str(row.get("canonical_url", "") or url))

    school_id = str(row.get("school_id", row.get("school", "")) or "").strip()
    dept_id = str(row.get("dept_id", row.get("dept", "")) or "").strip()
    attachments = row.get("attachments", [])
    if not isinstance(attachments, list):
        attachments = []
    images = row.get("images", [])
    if not isinstance(images, list):
        images = []

    doc_id = str(row.get("doc_id", "") or "").strip()
    if not doc_id:
        doc_id = build_doc_id(school_id, dept_id, canonical_url)

    content_hash = str(row.get("content_hash", "") or "").strip()
    if not content_hash:
        content_hash = build_content_hash(
            title=title,
            published_at=published_at,
            canonical_url=canonical_url,
            content=content,
            attachments=attachments,
            images=images,
        )

    collected_at = str(row.get("collected_at", "") or "").strip() or utc_now_iso()
    updated_at = str(row.get("updated_at", "") or "").strip() or collected_at

    out = dict(row)
    out["doc_id"] = doc_id
    out["domain"] = str(row.get("domain", "") or "notice")
    out["source_type"] = str(row.get("source_type", "") or "board_notice")
    out["url"] = url
    out["canonical_url"] = canonical_url
    out["date"] = published_at
    out["published_at"] = published_at
    out["content_hash"] = content_hash
    out["collected_at"] = collected_at
    out["updated_at"] = updated_at

    if not isinstance(row.get("raw"), dict):
        out["raw"] = {
            "title_raw": title,
            "date_raw": date_raw,
            "url_raw": url,
            "content_raw": content,
        }
    if not isinstance(row.get("normalized"), dict):
        out["normalized"] = {
            "title": title,
            "published_at": published_at,
            "canonical_url": canonical_url,
            "content": content,
        }
    if not isinstance(row.get("assets"), dict):
        out["assets"] = {
            "images": images,
            "attachments": attachments,
        }
    return out


def migrate_rows(rows: List[Dict]) -> List[Dict]:
    migrated = [to_notice_envelope(row) for row in rows]
    doc_versions: Dict[str, Tuple[int, str]] = {}

    for row in migrated:
        doc_id = str(row.get("doc_id", "")).strip()
        row_hash = str(row.get("content_hash", "")).strip()
        prev = doc_versions.get(doc_id)
        if prev is None:
            version = 1
        else:
            prev_version, prev_hash = prev
            version = prev_version if row_hash == prev_hash else prev_version + 1
        row["version"] = version
        row["is_current"] = False
        doc_versions[doc_id] = (version, row_hash)

    last_index: Dict[str, int] = {}
    for idx, row in enumerate(migrated):
        last_index[str(row.get("doc_id", "")).strip()] = idx
    for idx in last_index.values():
        migrated[idx]["is_current"] = True
    return migrated


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate notice jsonl rows to envelope schema")
    parser.add_argument("--input", default="data", help="Input root directory")
    parser.add_argument(
        "--glob",
        default="*.jsonl",
        help="File glob relative to input root (default: *.jsonl, recursive)",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite source files. If false, write *.migrated.jsonl",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze only, do not write files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.input)
    if not root.exists():
        print(f"[ERROR] Input path not found: {root}")
        return

    files = sorted(
        [
            p
            for p in root.rglob(args.glob)
            if p.is_file() and not p.name.endswith(".migrated.jsonl")
        ]
    )
    if not files:
        print(f"[INFO] No files matched under {root} with glob={args.glob}")
        return

    total_rows = 0
    touched_files = 0
    for path in files:
        rows = list(iter_jsonl(path))
        if not rows:
            continue
        migrated = migrate_rows(rows)
        total_rows += len(migrated)
        touched_files += 1

        if args.dry_run:
            print(f"[DRY] {path} rows={len(migrated)}")
            continue

        if args.in_place:
            out_path = path
        else:
            out_path = path.with_suffix(".migrated.jsonl")
        write_jsonl(out_path, migrated)
        print(f"[OK] {path} -> {out_path} rows={len(migrated)}")

    print(f"[DONE] files={touched_files}, rows={total_rows}")


if __name__ == "__main__":
    main()
