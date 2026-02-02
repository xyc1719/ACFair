# admission_proxy.py
'''
AI generated content without human review.
'''
import asyncio
import json
import time
from typing import Any, Dict

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

VLLM_BASE = "http://127.0.0.1:8000"
MAX_INFLIGHT = 8

app = FastAPI()
sema = asyncio.Semaphore(MAX_INFLIGHT)

def now_ms() -> int:
    return int(time.time() * 1000)

@app.post("/v1/chat/completions")
async def chat_completions(req: Request):
    body: Dict[str, Any] = await req.json()
    stream = bool(body.get("stream", False))

    # --- Admission decision (demo: concurrency cap) ---
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
        try:
            async with client.stream(
                "POST",
                f"{VLLM_BASE}/v1/chat/completions",
                json=body,
                headers={"Content-Type": "application/json"},
            ) as r:
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
                        if last_chunk_time is not None:
                            chunk_intervals_ms.append((t - last_chunk_time) * 1000)
                        last_chunk_time = t

                        yield (line + "\n\n").encode("utf-8")
                    else:
                        # passthrough (rare)
                        yield (line + "\n").encode("utf-8")
        finally:
            await client.aclose()
            await _release()

            end = time.perf_counter()
            ttft_ms = None if first_chunk_time is None else (first_chunk_time - start) * 1000
            e2e_ms = (end - start) * 1000

            # 这里 demo 先直接 print；后面换成写 CSV / Prometheus
            print(json.dumps({
                "admitted_at_ms": admitted_at,
                "stream": stream,
                "ttft_ms": ttft_ms,
                "e2e_ms": e2e_ms,
                "chunk_intervals_ms_p50": percentile(chunk_intervals_ms, 50),
                "chunk_intervals_ms_p95": percentile(chunk_intervals_ms, 95),
                "rejected": False,
            }, ensure_ascii=False))

    if stream:
        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    # non-stream: simpler
    try:
        r = await client.post(f"{VLLM_BASE}/v1/chat/completions", json=body)
        r.raise_for_status()
        data = r.json()
        end = time.perf_counter()
        e2e_ms = (end - start) * 1000
        # 非流式 TTFT 不好定义；可以先跳过或用“响应到达”代替
        print(json.dumps({"admitted_at_ms": admitted_at, "stream": False, "e2e_ms": e2e_ms, "rejected": False}, ensure_ascii=False))
        return JSONResponse(content=data)
    finally:
        await client.aclose()
        await _release()

def percentile(arr, p):
    if not arr:
        return None
    s = sorted(arr)
    k = int(round((p/100.0) * (len(s)-1)))
    return s[max(0, min(k, len(s)-1))]
