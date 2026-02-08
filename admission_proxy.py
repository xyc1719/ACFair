# admission_proxy.py
'''
AI generated content without human review.
'''
import asyncio
import json
import time
import numpy as np
from typing import Any, Dict
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

VLLM_BASE = "http://127.0.0.1:8000"
MAX_INFLIGHT = 8
DEFAULT_TTFT_SLO_MS = 500  # 默认 TTFT SLO（ms），可按需调整
DEFAULT_TPOT_SLO_MS = 100  # 默认 TPOT SLO（ms），可按需调整
KV_CACHE_BYTES_PER_TOKEN = 100000  # 每个 token 的 KV cache 估算字节数（粗略值，实际可根据模型调整）

# in-memory per-user token counters (simple, not persisted)
user_token_stats = {}
user_KV_cache_stats = {}

def now_ms() -> int:
    return int(time.time() * 1000)

import math
from transformers import AutoConfig

DTYPE_BYTES = {
    "fp16": 2, "float16": 2,
    "bf16": 2, "bfloat16": 2,
    "fp32": 4, "float32": 4,
    "fp8": 1,  # 如果你确实在用 fp8 KV（并非所有配置都支持）
    "int8": 1,
}

def estimate_kv_bytes_per_block(
    model_name_or_path: str,
    block_size: int,
    dtype: str = "fp16",
    tensor_parallel_size: int = 1,
) -> dict:
    """
    返回每个 block 的 KV bytes 估算值（按单 GPU 视角）。
    """
    cfg = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)

    # 1) 层数
    num_layers = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer", None)
    if num_layers is None:
        raise ValueError("无法从 config 推断 num_hidden_layers / n_layer")

    # 2) KV heads
    # LLaMA/Qwen2 等通常是 num_key_value_heads；有些模型叫 n_head_kv
    num_kv_heads = getattr(cfg, "num_key_value_heads", None) or getattr(cfg, "n_head_kv", None)
    if num_kv_heads is None:
        # 若没有明确 KV head 字段，就退化成 attention heads（非 GQA/MQA）
        num_kv_heads = getattr(cfg, "num_attention_heads", None) or getattr(cfg, "n_head", None)
        if num_kv_heads is None:
            raise ValueError("无法从 config 推断 num_key_value_heads / num_attention_heads")

    # 3) head_dim
    head_dim = getattr(cfg, "head_dim", None)
    if head_dim is None:
        hidden_size = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None)
        num_attn_heads = getattr(cfg, "num_attention_heads", None) or getattr(cfg, "n_head", None)
        if hidden_size is None or num_attn_heads is None:
            raise ValueError("无法从 config 推断 head_dim（缺 hidden_size 或 num_attention_heads）")
        head_dim = hidden_size // num_attn_heads

    # 4) dtype bytes
    dtype_key = dtype.lower()
    if dtype_key not in DTYPE_BYTES:
        raise ValueError(f"未知 dtype={dtype}，可选：{sorted(DTYPE_BYTES.keys())}")
    dtype_bytes = DTYPE_BYTES[dtype_key]

    # 5) bytes/block（先算全量，再换算到单 GPU）
    bytes_per_block_total = (
        num_layers
        * num_kv_heads
        * 2  # K + V
        * block_size
        * head_dim
        * dtype_bytes
    )

    # 常见实现中，TP 会把 heads 分到各卡，KV cache 也随之分片
    bytes_per_block_per_gpu = bytes_per_block_total // max(1, tensor_parallel_size)

    return {
        "model": model_name_or_path,
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "block_size": block_size,
        "dtype": dtype_key,
        "dtype_bytes": dtype_bytes,
        "tensor_parallel_size": tensor_parallel_size,
        "bytes_per_block_total_all_gpus": int(bytes_per_block_total),
        "bytes_per_block_per_gpu_est": int(bytes_per_block_per_gpu),
        "mb_per_block_per_gpu_est": bytes_per_block_per_gpu / (1024 * 1024),
    }

def estimate_tokens(text: str) -> int:
    """非常粗略的 token 估算：按空白分词计数。"""
    if not text:
        return 0
    return len(text.split())

def record_user_tokens(user_id: str, input_tokens: int = 0, output_tokens: int = 0):
    s = user_token_stats.setdefault(user_id, {"input": 0, "output": 0})
    s["input"] += input_tokens
    s["output"] += output_tokens

def record_user_KV_cache(user_id: str, kv_cache_bytes: int):
    s = user_KV_cache_stats.setdefault(user_id, {"kv_cache_bytes": 0})
    s["kv_cache_bytes"] += kv_cache_bytes

historical_KV_cache = []

def historical_KVcache_estimator(user_id: str, input_tokens: int, max_output_tokens: int = 128):
    """基于历史 KV cache 数据计算 P90/P95 峰值估算"""
    if len(historical_KV_cache)<5:
        # 缺少历史数据，返回保守估计
        return int((input_tokens + max_output_tokens) * KV_CACHE_BYTES_PER_TOKEN)

    # 保留最近 100 条记录以避免内存溢出
    if len(historical_KV_cache) > 100:
        historical_KV_cache.pop(0)

    # 计算 P90 作为准入的保守估计
    p90_kv_bytes = percentile(historical_KV_cache, 90)
    return p90_kv_bytes 

'''
Main entry point
'''
info = estimate_kv_bytes_per_block(
model_name_or_path="Qwen/Qwen2.5-3B-Instruct", # model name
block_size=16,
dtype="fp16",
tensor_parallel_size=1,
)
KV_CACHE_BYTES_PER_TOKEN = info["bytes_per_block_total_all_gpus"]//16

app = FastAPI()
sema = asyncio.Semaphore(MAX_INFLIGHT)

@app.post("/v1/chat/completions")
async def chat_completions(req: Request):
    body: Dict[str, Any] = await req.json()
    stream = bool(body.get("stream", False))

    # no pending queue; reject immediately if at capacity
    if sema.locked() and sema._value == 0:
        return JSONResponse(
            status_code=429,
            content={"error": {"message": "Rejected by admission control (over capacity)."}}
        )

    await sema.acquire()
    admitted_at = now_ms()

    async def _release():
        sema.release()

    # --- Forward to vLLM ---
    client = httpx.AsyncClient(timeout=None, trust_env=False)

    start = time.perf_counter()
    first_chunk_time = None
    last_chunk_time = None
    chunk_intervals_ms = []

    async def stream_generator():
        nonlocal first_chunk_time, last_chunk_time
        # initialize variables to avoid UnboundLocalError in finally block
        user_id = None
        kv_cache_bytes_est = 0
        input_tokens_est = 0
        
        try:
            # extract user id
            user_id = body.get("user")
            #TODO:基于历史的KV cache 占用估算 P90/P95
            peak_kv_cache_est = historical_KVcache_estimator(user_id, input_tokens_est,128)
            # estimate input tokens from message prompt (rough)
            msgs = body.get("messages") or []
            for m in msgs:
                if isinstance(m, dict):
                    input_tokens_est += estimate_tokens(m.get("content", ""))
            kv_cache_bytes_est = int(input_tokens_est * KV_CACHE_BYTES_PER_TOKEN)
            # record input tokens immediately
            record_user_tokens(user_id, input_tokens=input_tokens_est)
            record_user_KV_cache(user_id, int(input_tokens_est * KV_CACHE_BYTES_PER_TOKEN))

            async with client.stream(
                "POST",
                f"{VLLM_BASE}/v1/chat/completions",
                json=body,
                headers={"Content-Type": "application/json"},
            ) as r:
                # report HTTPError if status code from vllm is not 2xx
                r.raise_for_status()

                async for line in r.aiter_lines():
                    if not line:
                        continue
                    # OpenAI SSE: "data: {...}" or "data: [DONE]"
                    if line.startswith("data: "):
                        payload = line[len("data: "):]

                        t = time.perf_counter()
                        if first_chunk_time is None:
                            first_chunk_time = t
                            # TTFT SLO check
                            ttft_ms = (first_chunk_time - start) * 1000
                            slo_violated = ttft_ms > DEFAULT_TTFT_SLO_MS
                        
                        # attempt to count output tokens from chunk payload
                        try:
                            if payload != "[DONE]":
                                obj = json.loads(payload)
                                # OpenAI-style delta content
                                for ch in obj.get("choices", []):
                                    delta = ch.get("delta", {})
                                    if isinstance(delta, dict):
                                        content = delta.get("content")
                                        if content:
                                            out_toks = estimate_tokens(content)
                                            kv_cache_bytes_est += out_toks * KV_CACHE_BYTES_PER_TOKEN
                                            record_user_tokens(user_id, output_tokens=out_toks)
                                            record_user_KV_cache(user_id, out_toks * KV_CACHE_BYTES_PER_TOKEN)
                        except Exception:
                            # ignore parse errors
                            pass
                        if last_chunk_time is not None:
                            chunk_intervals_ms.append((t - last_chunk_time) * 1000)
                            # TPOT SLO check (after first chunk)
                            if sum(chunk_intervals_ms) > len(chunk_intervals_ms) * DEFAULT_TPOT_SLO_MS:
                                slo_violated = True
                        last_chunk_time = t

                        yield (line + "\n\n").encode("utf-8")
                    else:
                        # passthrough (rare)
                        yield (line + "\n").encode("utf-8")
        finally:
            await client.aclose()
            await _release()
            record_user_KV_cache(user_id, -kv_cache_bytes_est)
            historical_KV_cache.append(kv_cache_bytes_est)

            end = time.perf_counter()
            ttft_ms = None if first_chunk_time is None else (first_chunk_time - start) * 1000
            e2e_ms = (end - start) * 1000

            # assemble per-request metrics including KV cache estimate and SLO result
            metrics = {
                "admitted_at_ms": admitted_at,
                "stream": stream,
                "ttft_ms": ttft_ms,
                "e2e_ms": e2e_ms,
                "chunk_intervals_ms_p50": percentile(chunk_intervals_ms, 50),
                "chunk_intervals_ms_p95": percentile(chunk_intervals_ms, 95),
                "kv_cache_bytes_est": kv_cache_bytes_est,
                "user_id": user_id,
                "rejected": False,
            }
            if ttft_ms is not None:
                metrics["slo_TTFT"] = DEFAULT_TTFT_SLO_MS
                metrics["slo_TPOT"] = DEFAULT_TPOT_SLO_MS
                metrics["slo_violated"] = slo_violated

            print(json.dumps(metrics, ensure_ascii=False))

            # also print aggregated per-user counters (lightweight)
            try:
                uts = user_token_stats.get(user_id, {})
                print(json.dumps({"user_id": user_id, "user_token_stats": uts}, ensure_ascii=False))
            except Exception:
                pass

    if stream:
        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    # non-stream: simpler but weaker.
    try:
        r = await client.post(f"{VLLM_BASE}/v1/chat/completions", json=body)
        r.raise_for_status()
        data = r.json()
        end = time.perf_counter()
        e2e_ms = (end - start) * 1000
        # 
        print(json.dumps({"admitted_at_ms": admitted_at, "stream": False, "e2e_ms": e2e_ms, "rejected": False}, ensure_ascii=False))
        return JSONResponse(content=data)
    finally:
        await client.aclose()
        await _release()

def percentile(arr:list, p):
    if arr is None or len(arr) == 0:
        return 0
    x = np.array(arr)
    return np.percentile(x, p)
