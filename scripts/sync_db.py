#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import json
import random
import hashlib
import inspect
import subprocess
from typing import Any, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain_google_vertexai import (
    VertexAIEmbeddings,
    VectorSearchVectorStoreDatastore,
)
from langchain.indexes import SQLRecordManager, index
from langchain_core.documents import Document

from google.cloud import aiplatform
from google.cloud import aiplatform_v1
from google.cloud.aiplatform_v1.types import IndexDatapoint, UpsertDatapointsRequest
import google.auth

# =========================== ENV ===========================
POSTGRES_CONNECTION_STRING = os.environ["POSTGRES_CONNECTION_STRING"]
PROJECT_ID = os.environ["GCP_PROJECT_ID"]
INDEX_ID = os.environ["VERTEX_AI_INDEX_ID"]
ENDPOINT_ID = os.environ["VERTEX_AI_ENDPOINT_ID"]
REGION = os.environ["VERTEX_AI_REGION"]
_raw_bucket = os.environ.get("VERTEX_AI_GCS_BUCKET", "")

GCS_BUCKET = _raw_bucket.replace("gs://", "").strip("/")
REPO_ROOT = os.path.abspath(os.getenv("REPO_ROOT", os.getcwd()))
USE_STREAM = os.getenv("VECTOR_UPDATE_MODE", "").lower() in ("stream", "streaming", "true", "1")
SAFE_STORE_LIMIT = int(os.getenv("SAFE_STORE_LIMIT", "1400"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
# 변경 파일 수가 이 값 미만이어도 이제 풀스캔하지 않음(그냥 증분)
DIFF_MIN_FILES_FOR_INCREMENTAL = int(os.getenv("DIFF_MIN_FILES_FOR_INCREMENTAL", "5"))
RECORD_MANAGER_NAMESPACE = os.getenv("RECORD_NAMESPACE", f"vertexai/{INDEX_ID}")

# ---- Stream throttles (12MB 제한 대비 여유치로 11.5MB) ----
STREAM_BYTES_BUDGET_PER_MIN_KB = int(os.getenv("STREAM_BYTES_BUDGET_PER_MIN", "11500"))
STREAM_BATCHES_PER_MIN = int(os.getenv("STREAM_BATCHES_PER_MIN", "20"))
STREAM_UPSERT_BATCH = int(os.getenv("STREAM_UPSERT_BATCH", "160"))  # 속도 ↑
SLEEP_BETWEEN_BATCH = float(os.getenv("SLEEP_BETWEEN_BATCH", "1.0"))

# ---- Embedding throttles & token guard ----
EMB_API_BATCH_SIZE = int(os.getenv("EMB_API_BATCH_SIZE", "8"))     # 속도 ↑
EMB_MAX_TOKENS_PER_REQ = int(os.getenv("EMB_MAX_TOKENS_PER_REQ", "18000"))
EMB_SAFETY_MARGIN_TOKENS = int(os.getenv("EMB_SAFETY_MARGIN_TOKENS", "1500"))
EMB_PER_DOC_MAX_TOKENS = int(os.getenv("EMB_PER_DOC_MAX_TOKENS", "1200"))  # 속도 ↑
EMB_WORKERS = int(os.getenv("EMB_WORKERS", "6"))                   # 속도 ↑

# ---- Retry/backoff ----
EMB_RETRY_MAX = int(os.getenv("EMB_RETRY_MAX", "6"))
EMB_RETRY_BASE = float(os.getenv("EMB_RETRY_BASE", "2.0"))
EMB_RETRY_MAX_SLEEP = float(os.getenv("EMB_RETRY_MAX_SLEEP", "22.0"))

UPSERT_RETRY_MAX = int(os.getenv("UPSERT_RETRY_MAX", "6"))
UPSERT_RETRY_BASE = float(os.getenv("UPSERT_RETRY_BASE", "2.0"))
UPSERT_RETRY_MAX_SLEEP = float(os.getenv("UPSERT_RETRY_MAX_SLEEP", "22.0"))

def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, int(len(text) / 3.6))

os.chdir(REPO_ROOT)
CODE_DIRS = [os.path.join(REPO_ROOT, p) for p in ["src", "include", "third_party", "test"]]

# ============================ INIT ============================
aiplatform.init(project=PROJECT_ID, location=REGION)
try:
    creds, detected_project = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    print("Auth project:", detected_project)
    print("Service Account:", getattr(creds, "service_account_email", None))
except Exception:
    pass

print("Initializing clients and managers...")
embeddings = VertexAIEmbeddings(model_name="text-embedding-004")

vectorstore = None
if not USE_STREAM:
    if not GCS_BUCKET:
        print("[fatal] batch 모드인데 staging bucket이 지정되지 않았습니다.")
        sys.exit(1)
    print(f"[env] Using GCS staging bucket: {GCS_BUCKET}")
    sig = inspect.signature(VectorSearchVectorStoreDatastore.from_components)
    params = sig.parameters
    bucket_kwargs: Dict[str, Any] = {}
    if "staging_bucket_name" in params:
        bucket_kwargs = {"staging_bucket_name": GCS_BUCKET}
    elif "index_staging_bucket_name" in params:
        bucket_kwargs = {"index_staging_bucket_name": GCS_BUCKET}
    elif "staging_bucket" in params:
        bucket_kwargs = {"staging_bucket": {"bucket_name": GCS_BUCKET}}
    elif "gcs_bucket_name" in params:
        bucket_kwargs = {"gcs_bucket_name": GCS_BUCKET}
    else:
        print("[fatal] No compatible staging-bucket parameter.")
        sys.exit(1)

    vectorstore = VectorSearchVectorStoreDatastore.from_components(
        project_id=PROJECT_ID,
        region=REGION,
        index_id=INDEX_ID,
        endpoint_id=ENDPOINT_ID,
        embedding=embeddings,
        **bucket_kwargs,
    )

record_manager = SQLRecordManager(RECORD_MANAGER_NAMESPACE, db_url=POSTGRES_CONNECTION_STRING)
record_manager.create_schema()
print("Initialization complete.")

# ===================== DIFF / LOAD =====================
def _git_stdout(cmd: List[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout

def _within_targets(p: str) -> bool:
    ap = os.path.abspath(p)
    return any(ap == d or ap.startswith(d + os.sep) for d in CODE_DIRS)

def _walk_all_files() -> List[str]:
    paths: List[str] = []
    exclude_dirs = {".git", "out", "build", "bazel-bin", "bazel-out", ".cipd", "third_party/llvm-build"}
    for base in CODE_DIRS:
        if not os.path.isdir(base):
            continue
        for root, subdirs, files in os.walk(base):
            subdirs[:] = [s for s in subdirs if s not in exclude_dirs]
            for fn in files:
                full = os.path.join(root, fn)
                rel = os.path.relpath(full, REPO_ROOT).replace("\\", "/")
                paths.append(rel)
    print(f"[full-scan] collected files={len(paths)} (first 5) -> {paths[:5]}")
    return paths

def get_changed_files(force_full: bool) -> Dict[str, List[str]]:
    if force_full:
        return {"added_modified": _walk_all_files(), "deleted": []}

    print("Checking for changed files using git diff...")
    out = ""
    tried: List[str] = []

    before = os.getenv("DIFF_BEFORE")
    after = os.getenv("DIFF_AFTER")
    if before and after:
        tried.append(f"{before}..{after}")
        out = _git_stdout(["git", "diff", "--name-status", before, after])

    if not out.strip():
        tried.append("HEAD~1..HEAD")
        out = _git_stdout(["git", "diff", "--name-status", "HEAD~1", "HEAD"])

    if not out.strip():
        tried.append("origin/main...HEAD")
        out = _git_stdout(["git", "diff", "--name-status", "origin/main...HEAD"])

    files = {"added_modified": [], "deleted": []}
    if not out.strip():
        print(f"[diff] no changes (tried: {', '.join(tried)}) → EXIT")
        return files  # ★ 변경사항 없음 → 그대로 종료

    for line in out.strip().split("\n"):
        if not line:
            continue
        parts = line.split("\t")
        status = parts[0]
        file_path = parts[-1]
        abs_path = os.path.join(REPO_ROOT, file_path)
        if not _within_targets(abs_path):
            continue
        if status.startswith(("A", "M", "R", "C")):
            files["added_modified"].append(file_path)
        elif status.startswith("D"):
            files["deleted"].append(file_path)

    # ★ 변경 파일이 적더라도 풀스캔 강제 금지
    print(
        f"Found files: {len(files['added_modified'])} added/modified, "
        f"{len(files['deleted'])} deleted"
    )
    return files

def load_documents_from_files(file_paths: List[str]) -> List[Document]:
    print(f"Loading {len(file_paths)} documents from files...")
    docs: List[Document] = []
    for rel in file_paths:
        if os.path.isabs(rel):
            abs_path = rel
            rel_path = os.path.relpath(rel, REPO_ROOT).replace("\\", "/")
        else:
            rel_path = rel.replace("\\", "/")
            abs_path = os.path.join(REPO_ROOT, rel_path)
        try:
            with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            if not content:
                continue
            docs.append(Document(page_content=content, metadata={"source": rel_path}))
        except FileNotFoundError:
            print(f"Warning: File not found, skip: {rel_path}")
        except Exception as e:
            print(f"Error loading file {rel_path}: {e}")
    return docs

def trim_to_safe_bytes(text: str, byte_limit: int = SAFE_STORE_LIMIT) -> str:
    data = text.encode("utf-8", errors="ignore")
    if len(data) <= byte_limit:
        return text
    data = data[:byte_limit]
    while True:
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            data = data[:-1]

# ================= EMBEDDING (token-guard + backoff) =================
def _truncate_by_tokens(s: str, max_toks: int) -> str:
    if estimate_tokens(s) <= max_toks:
        return s
    ratio = max(0.1, max_toks / max(1, estimate_tokens(s)))
    cut = int(len(s) * ratio)
    return s[:cut]

def _pack_requests_by_tokens(texts: List[str]) -> List[List[str]]:
    hard_cap = max(1000, EMB_MAX_TOKENS_PER_REQ - EMB_SAFETY_MARGIN_TOKENS)
    packed: List[List[str]] = []
    cur: List[str] = []
    cur_tokens = 0
    for t in texts:
        t2 = _truncate_by_tokens(t, EMB_PER_DOC_MAX_TOKENS)
        tok = estimate_tokens(t2)
        if tok > hard_cap:
            t2 = _truncate_by_tokens(t2, hard_cap)
            tok = estimate_tokens(t2)
        if cur and (cur_tokens + tok > hard_cap or len(cur) >= EMB_API_BATCH_SIZE):
            packed.append(cur); cur, cur_tokens = [], 0
        cur.append(t2); cur_tokens += tok
    if cur: packed.append(cur)
    return packed

def _do_req_with_backoff(batch: List[str]) -> List[List[float]]:
    delay = EMB_RETRY_BASE
    for attempt in range(EMB_RETRY_MAX):
        try:
            return embeddings.embed_documents(batch)
        except Exception as e:
            msg = str(e)
            retriable = (
                "InternalServerError" in msg or "500" in msg or "429" in msg
                or "ResourceExhausted" in msg or "Deadline" in msg
                or "temporarily" in msg.lower() or "unavailable" in msg.lower()
            )
            if not retriable or attempt == EMB_RETRY_MAX - 1:
                print(f"[emb][fail] {msg}")
                raise
            sleep = min(EMB_RETRY_MAX_SLEEP, delay + random.uniform(0, 1.0))
            print(f"[emb][retry {attempt+1}/{EMB_RETRY_MAX}] {msg} -> sleep {sleep:.1f}s")
            time.sleep(sleep); delay *= 1.7
    return embeddings.embed_documents(batch)

def embed_all(texts: List[str]) -> List[List[float]]:
    print(f"[emb] total={len(texts)} bs={EMB_API_BATCH_SIZE} thr={EMB_WORKERS}")
    vecs: List[List[float]] = []
    pkgs = _pack_requests_by_tokens(texts)
    done = 0; total = len(texts)
    with ThreadPoolExecutor(max_workers=EMB_WORKERS) as ex:
        futs = [ex.submit(_do_req_with_backoff, b) for b in pkgs]
        for fu in as_completed(futs):
            res = fu.result()
            vecs.extend(res); done += len(res)
            if done % 512 == 0 or done == total:
                print(f"[emb] {done}/{total} ({done*100//max(1,total)}%)")
    return vecs

# ================= STREAM UPSERT (throttle + backoff) =================
def _dp_id(source: str, start: int, end: int) -> str:
    return hashlib.sha256(f"{source}|{start}|{end}".encode()).hexdigest()[:40]

def approx_point_kb(vec_dim: int = 768, meta_overhead: int = 300) -> int:
    return int((vec_dim * 4 + meta_overhead) / 1024)

def stream_upsert_docs(docs: List[Document]) -> None:
    texts = [d.page_content for d in docs]
    if not texts:
        print("[stream] nothing to upsert"); return

    vecs = embed_all(texts)

    datapoints: List[IndexDatapoint] = []
    for vec, d in zip(vecs, docs):
        src = d.metadata.get("source", "unknown")
        start = int(d.metadata.get("chunk_start", 0))
        end = int(d.metadata.get("chunk_end", start + len(d.page_content)))
        datapoints.append(IndexDatapoint(datapoint_id=_dp_id(src, start, end), feature_vector=vec))

    index_client = aiplatform_v1.IndexServiceClient(
        client_options={"api_endpoint": f"{REGION}-aiplatform.googleapis.com"}
    )
    index_name = f"projects/{PROJECT_ID}/locations/{REGION}/indexes/{INDEX_ID}"

    per_point_kb = approx_point_kb(768)
    minute_budget_kb = STREAM_BYTES_BUDGET_PER_MIN_KB
    max_batches_per_min = STREAM_BATCHES_PER_MIN
    batch_size = STREAM_UPSERT_BATCH
    sleep_between = SLEEP_BETWEEN_BATCH

    sent_in_min_kb = 0; batches_in_min = 0; minute_start = time.time()
    total = len(datapoints); done = 0; i = 0

    while i < total:
        now = time.time()
        if now - minute_start >= 60:
            minute_start = now; sent_in_min_kb = 0; batches_in_min = 0

        remaining = max(0, minute_budget_kb - sent_in_min_kb)
        if remaining <= 0 or batches_in_min >= max_batches_per_min:
            wait = 60 - (now - minute_start)
            if wait > 0:
                print(f"[upsert] minute cap reached → sleep {wait:.1f}s"); time.sleep(wait)
            continue

        budget_pts = max(1, int(remaining // per_point_kb))
        cur_batch_size = max(1, min(batch_size, budget_pts, total - i))
        batch = datapoints[i:i+cur_batch_size]
        req = UpsertDatapointsRequest(index=index_name, datapoints=batch)

        delay = UPSERT_RETRY_BASE
        for attempt in range(UPSERT_RETRY_MAX):
            try:
                index_client.upsert_datapoints(request=req)
                i += cur_batch_size; done += cur_batch_size
                sent_in_min_kb += cur_batch_size * per_point_kb
                batches_in_min += 1
                if done % 200 == 0 or done == total:
                    print(f"[upsert] {done}/{total} ({done*100//total}%)  "
                          f"minute_used={sent_in_min_kb}/{minute_budget_kb} KB  "
                          f"batches={batches_in_min}/{max_batches_per_min}")
                time.sleep(sleep_between)
                break
            except Exception as e:
                msg = str(e)
                retriable = (
                    "ResourceExhausted" in msg or "Quota" in msg or "429" in msg
                    or "Internal" in msg or "500" in msg or "unavailable" in msg.lower()
                )
                if not retriable or attempt == UPSERT_RETRY_MAX - 1:
                    print(f"[upsert][fail] {msg}"); raise
                batch_size = max(20, int(batch_size * 0.7))
                wait = min(UPSERT_RETRY_MAX_SLEEP, max(1.0, 60 - (time.time() - minute_start)))
                extra = min(UPSERT_RETRY_MAX_SLEEP, delay + random.uniform(0, 1.0))
                sleep = max(wait, extra)
                print(f"[upsert][retry {attempt+1}/{UPSERT_RETRY_MAX}] {msg} "
                      f"-> shrink batch={batch_size}, sleep {sleep:.1f}s")
                time.sleep(sleep)
                minute_start = time.time(); sent_in_min_kb = 0; batches_in_min = 0; delay *= 1.7

    print(f"[stream] upserted {done} datapoints")

# =========================== MAIN ===========================
if __name__ == "__main__":
    force_full = ("--full" in sys.argv) or (os.getenv("FORCE_FULL_SCAN", "").lower() in ("1", "true", "yes"))
    print(f"[mode] REPO_ROOT={REPO_ROOT}")
    print(f"[mode] force_full={force_full}, use_stream={USE_STREAM}")

    changed = get_changed_files(force_full=force_full)
    add_mod = changed["added_modified"]
    deleted = changed["deleted"]

    # ★ 변경사항 없으면 즉시 종료
    if not add_mod and not deleted:
        print("No changes detected. Exit.")
        sys.exit(0)

    docs_raw = load_documents_from_files(add_mod)
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.CPP, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True
    )
    split_docs = splitter.split_documents(docs_raw)

    safe_docs: List[Document] = []
    for d in split_docs:
        pc = d.page_content
        if len(pc.encode("utf-8", errors="ignore")) > SAFE_STORE_LIMIT:
            pc = trim_to_safe_bytes(pc, SAFE_STORE_LIMIT)
        md = dict(d.metadata or {})
        start_idx = int(md.get("start_index", 0))
        end_idx = start_idx + len(pc)
        md["chunk_start"] = start_idx
        md["chunk_end"] = end_idx
        safe_docs.append(Document(page_content=pc, metadata=md))

    print(f"[docs] chunks={len(safe_docs)} from {len(add_mod)} files; deletes={len(deleted)}")

    if USE_STREAM:
        if deleted:
            # 스트림 즉시 삭제는 로컬 manifest 없으면 불가 → 경고만
            show = min(20, len(deleted))
            for f in deleted[:show]:
                print(f"[warn][stream] delete '{f}' skipped (no manifest for datapoint ids)")
            if len(deleted) > show:
                print(f"[warn][stream] ... and {len(deleted)-show} more deleted paths")
        if safe_docs:
            stream_upsert_docs(safe_docs)
        else:
            print("No new/modified docs to upsert.")
    else:
        kwargs = {"record_manager": record_manager, "cleanup": "incremental"}
        sig = inspect.signature(index).parameters
        if "vectorstore" in sig: kwargs["vectorstore"] = vectorstore
        elif "vector_store" in sig: kwargs["vector_store"] = vectorstore
        if "source_id_key" in sig: kwargs["source_id_key"] = "source"
        if "source_id" in sig: kwargs["source_id"] = os.getenv("SOURCE_ID", "v8-repo")
        index(safe_docs, **kwargs)

    print("Done.")