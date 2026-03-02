import json
import os
import time
from typing import Dict, List, Optional

import requests


class CloudflareDenseEncoder:
    def __init__(
        self,
        account_id: str,
        api_token: str,
        model: str = "@cf/baai/bge-m3",
        timeout: int = 30,
        max_retries: int = 3,
    ):
        self.account_id = account_id
        self.api_token = api_token
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.dim = 1024
        self.url = f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/ai/run/{self.model}"
        self.headers = {"Authorization": f"Bearer {self.api_token}"}
        self.session = requests.Session()

    def _post_with_retry(self, payload: Dict) -> Optional[requests.Response]:
        for attempt in range(self.max_retries):
            try:
                resp = self.session.post(
                    self.url,
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout,
                )
                if resp.status_code < 400:
                    return resp
                if resp.status_code == 429 or 500 <= resp.status_code < 600:
                    sleep_s = min(2 ** attempt, 8)
                    time.sleep(sleep_s)
                    continue
                return resp
            except requests.RequestException:
                if attempt == self.max_retries - 1:
                    return None
                sleep_s = min(2 ** attempt, 8)
                time.sleep(sleep_s)
        return None

    def encode(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        payload = {"text": texts}
        resp = self._post_with_retry(payload)
        if resp is None:
            return [[0.0] * self.dim for _ in texts]

        try:
            data = resp.json()
            if not data.get("success"):
                return [[0.0] * self.dim for _ in texts]
            vectors = data.get("result", {}).get("data", [])
            if not vectors:
                return [[0.0] * self.dim for _ in texts]
            if isinstance(vectors[0], list):
                fixed = []
                for v in vectors:
                    fixed.append(v if len(v) == self.dim else [0.0] * self.dim)
                if len(fixed) < len(texts):
                    fixed.extend([[0.0] * self.dim for _ in range(len(texts) - len(fixed))])
                return fixed[: len(texts)]
            if isinstance(vectors, list) and len(vectors) == self.dim:
                return [vectors for _ in texts]
            return [[0.0] * self.dim for _ in texts]
        except (ValueError, KeyError, TypeError):
            return [[0.0] * self.dim for _ in texts]


class LocalBGEM3DenseEncoder:
    def __init__(
        self,
        model_id: str = "BAAI/bge-m3",
        batch_size: int = 8,
        max_length: int = 512,
    ):
        self.model_id = model_id
        self.batch_size = batch_size
        self.max_length = max_length
        self.dim = 1024
        self._ready = False
        self._init_attempted = False
        self._torch = None
        self._tokenizer = None
        self._model = None

    def _init_once(self) -> None:
        if self._init_attempted:
            return
        self._init_attempted = True
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except Exception:
            return

        if not torch.cuda.is_available():
            return

        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
            model = AutoModel.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            model.eval()
            self._torch = torch
            self._tokenizer = tokenizer
            self._model = model
            self._ready = True
        except Exception:
            self._ready = False

    @staticmethod
    def _mean_pool(token_embeddings, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        masked = token_embeddings * mask
        summed = masked.sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-9)
        return summed / denom

    def encode(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        self._init_once()
        if not self._ready:
            return [[0.0] * self.dim for _ in texts]

        out_vectors: List[List[float]] = []
        bs = self.batch_size
        i = 0
        while i < len(texts):
            batch = [str(x or "") for x in texts[i : i + bs]]
            try:
                tokenized = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                tokenized = {k: v.to(self._model.device) for k, v in tokenized.items()}
                with self._torch.no_grad():
                    outputs = self._model(**tokenized)
                    pooled = self._mean_pool(outputs.last_hidden_state, tokenized["attention_mask"])
                    pooled = self._torch.nn.functional.normalize(pooled, p=2, dim=1)
                out_vectors.extend(pooled.detach().float().cpu().tolist())
                i += bs
            except Exception as exc:
                if self._torch is not None and self._torch.cuda.is_available():
                    self._torch.cuda.empty_cache()
                if ("out of memory" in str(exc).lower() or "cuda" in str(exc).lower()) and bs > 1:
                    bs = max(1, bs // 2)
                    time.sleep(0.2)
                    continue
                out_vectors.extend([[0.0] * self.dim for _ in batch])
                i += bs

        if len(out_vectors) < len(texts):
            out_vectors.extend([[0.0] * self.dim for _ in range(len(texts) - len(out_vectors))])
        return out_vectors[: len(texts)]


def _extract_json(text: str) -> Dict:
    if not text:
        return {}
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}


def _is_gpu_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


class LocalQwenMetadataEnricher:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-14B-Instruct",
        max_new_tokens: int = 400,
        max_input_chars: int = 12000,
    ):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.max_input_chars = max_input_chars

        self._ready = False
        self._init_attempted = False
        self._torch = None
        self._tokenizer = None
        self._model = None

    def _init_once(self) -> None:
        if self._init_attempted:
            return
        self._init_attempted = True

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception:
            return

        if not torch.cuda.is_available():
            return

        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            self._torch = torch
            self._tokenizer = tokenizer
            self._model = model
            self._ready = True
        except Exception:
            self._ready = False

    @staticmethod
    def _prompt(title: str, content: str, date: str) -> str:
        return (
            "Extract JSON only. Do not include markdown.\\n"
            "Required keys: summary, category, deadlines, target_group, requires_action, contact, "
            "valid_until, deadline_confidence, evidence_text.\\n"
            "Rules:\\n"
            "- summary: short Korean sentence <=150 chars\\n"
            "- deadlines: list of {label, datetime} with datetime in YYYY-MM-DDTHH:MM if possible\\n"
            "- requires_action: boolean\\n"
            "- valid_until: final deadline datetime in YYYY-MM-DDTHH:MM, empty if unknown\\n"
            "- deadline_confidence: float 0.0~1.0\\n"
            "- evidence_text: short quote-like evidence from the notice\\n"
            "- Unknown values should be empty string/list/false\\n\\n"
            f"title: {title}\\n"
            f"date: {date}\\n"
            f"content: {content}"
        )

    @staticmethod
    def _split_windows(text: str, window_size: int, overlap: int = 1200) -> List[str]:
        if not text:
            return [""]
        if len(text) <= window_size:
            return [text]
        step = max(1, window_size - max(0, overlap))
        windows = []
        for start in range(0, len(text), step):
            piece = text[start : start + window_size]
            if not piece:
                continue
            windows.append(piece)
            if start + window_size >= len(text):
                break
        return windows

    @staticmethod
    def _merge_metadata(base: Dict, new: Dict) -> Dict:
        if not isinstance(base, dict):
            base = {}
        if not isinstance(new, dict) or not new:
            return base

        merged = dict(base)

        for key in ["summary", "category", "target_group", "contact", "valid_until"]:
            if not str(merged.get(key, "")).strip() and str(new.get(key, "")).strip():
                merged[key] = new.get(key)

        merged["requires_action"] = bool(merged.get("requires_action", False)) or bool(
            new.get("requires_action", False)
        )

        try:
            base_conf = float(merged.get("deadline_confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            base_conf = 0.0
        try:
            new_conf = float(new.get("deadline_confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            new_conf = 0.0
        merged["deadline_confidence"] = max(base_conf, new_conf)

        merged_deadlines = merged.get("deadlines", [])
        if not isinstance(merged_deadlines, list):
            merged_deadlines = []
        new_deadlines = new.get("deadlines", [])
        if not isinstance(new_deadlines, list):
            new_deadlines = []
        seen = set()
        dedup = []
        for item in merged_deadlines + new_deadlines:
            if isinstance(item, dict):
                label = str(item.get("label", "")).strip()
                dt = str(item.get("datetime", "")).strip()
                key = (label, dt)
                if key in seen:
                    continue
                seen.add(key)
                dedup.append({"label": label, "datetime": dt})
            elif isinstance(item, str):
                dt = item.strip()
                key = ("", dt)
                if key in seen:
                    continue
                seen.add(key)
                dedup.append({"label": "", "datetime": dt})
        merged["deadlines"] = dedup

        merged_evidence = str(merged.get("evidence_text", "") or "").strip()
        new_evidence = str(new.get("evidence_text", "") or "").strip()
        if not merged_evidence and new_evidence:
            merged["evidence_text"] = new_evidence
        elif merged_evidence and new_evidence and new_evidence not in merged_evidence:
            merged["evidence_text"] = f"{merged_evidence} | {new_evidence}"[:1000]

        return merged

    def _infer_once(self, title: str, content: str, date: str) -> Dict:
        content_text = content or ""
        input_sizes = [self.max_input_chars, min(8000, self.max_input_chars), min(5000, self.max_input_chars)]

        for input_size in input_sizes:
            input_text = self._prompt(title=title, content=content_text[:input_size], date=date)
            try:
                tokenized = self._tokenizer(input_text, return_tensors="pt", truncation=True, max_length=6144)
                tokenized = {k: v.to(self._model.device) for k, v in tokenized.items()}

                with self._torch.no_grad():
                    outputs = self._model.generate(
                        **tokenized,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        temperature=0.0,
                        eos_token_id=self._tokenizer.eos_token_id,
                        pad_token_id=self._tokenizer.eos_token_id,
                    )

                generated_ids = outputs[0][tokenized["input_ids"].shape[1] :]
                text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
                parsed = _extract_json(text)
                if isinstance(parsed, dict) and parsed:
                    return parsed
            except Exception as exc:
                if self._torch is not None and self._torch.cuda.is_available():
                    self._torch.cuda.empty_cache()
                if "out of memory" in str(exc).lower() or "cuda" in str(exc).lower():
                    time.sleep(0.2)
                    continue
                return {}
        return {}

    def enrich(self, title: str, content: str, date: str) -> Dict:
        self._init_once()
        if not self._ready:
            return {}

        content_text = content or ""
        windows = self._split_windows(content_text, window_size=self.max_input_chars, overlap=1200)
        merged: Dict = {}

        for window in windows:
            parsed = self._infer_once(title=title, content=window, date=date)
            merged = self._merge_metadata(merged, parsed)

        return merged


class GroqMetadataEnricher:
    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.3-70b-versatile",
        timeout: int = 30,
        max_retries: int = 3,
        min_interval: float = 0.65,
    ):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.min_interval = min_interval
        self.last_call = 0.0
        self.url = "https://api.groq.com/openai/v1/chat/completions"
        self.session = requests.Session()
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _rate_limit_wait(self) -> None:
        now = time.time()
        wait_s = self.min_interval - (now - self.last_call)
        if wait_s > 0:
            time.sleep(wait_s)
        self.last_call = time.time()

    def enrich(self, title: str, content: str, date: str) -> Dict:
        if not self.api_key:
            return {}

        prompt = (
            "Extract JSON only with keys: "
            "summary, category, deadlines, target_group, requires_action, contact, "
            "valid_until, deadline_confidence, evidence_text. "
            "Use Korean where possible. summary max 150 chars.\\n\\n"
            f"title: {title}\\n"
            f"date: {date}\\n"
            f"content: {content[:4000]}"
        )
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 500,
        }

        for attempt in range(self.max_retries):
            self._rate_limit_wait()
            try:
                resp = self.session.post(
                    self.url,
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout,
                )
                if resp.status_code < 400:
                    out = resp.json()
                    text = out.get("choices", [{}])[0].get("message", {}).get("content", "")
                    return _extract_json(text)
                if resp.status_code == 429 or 500 <= resp.status_code < 600:
                    time.sleep(min(2 ** attempt, 8))
                    continue
                return {}
            except requests.RequestException:
                if attempt == self.max_retries - 1:
                    return {}
                time.sleep(min(2 ** attempt, 8))
        return {}


class ConditionalMetadataEnricher:
    """GPU server: local Qwen, non-GPU server: Groq."""

    def __init__(self, groq_api_key: Optional[str] = None, qwen_model_id: Optional[str] = None):
        self.is_gpu = _is_gpu_available()
        self.backend = "qwen_local" if self.is_gpu else "groq"

        if self.is_gpu:
            model_id = qwen_model_id or os.getenv("LOCAL_LLM_ID", "Qwen/Qwen2.5-14B-Instruct")
            self.impl = LocalQwenMetadataEnricher(model_id=model_id)
        else:
            self.impl = GroqMetadataEnricher(api_key=groq_api_key or "")

    def enrich(self, title: str, content: str, date: str) -> Dict:
        return self.impl.enrich(title=title, content=content, date=date)


class ConditionalDenseEncoder:
    """GPU server: local BGE-M3 (FP16), non-GPU server: Cloudflare BGE-M3."""

    def __init__(self, cf_account_id: Optional[str] = None, cf_api_token: Optional[str] = None):
        self.is_gpu = _is_gpu_available()
        self.backend = "bge_m3_local_fp16" if self.is_gpu else "cloudflare_bge_m3"
        self.dim = 1024

        if self.is_gpu:
            model_id = os.getenv("LOCAL_DENSE_MODEL_ID", "BAAI/bge-m3")
            self.impl = LocalBGEM3DenseEncoder(model_id=model_id)
            return

        if not cf_account_id or not cf_api_token:
            raise ValueError("Cloudflare credentials are required on non-GPU server.")
        self.impl = CloudflareDenseEncoder(account_id=cf_account_id, api_token=cf_api_token)

    def encode(self, texts: List[str]) -> List[List[float]]:
        return self.impl.encode(texts)
