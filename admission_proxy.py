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
MAX_INFLIGHT = 32
DEFAULT_TTFT_SLO_MS = 500  # 默认 TTFT SLO（ms），可按需调整
DEFAULT_TPOT_SLO_MS = 50  # 默认 TPOT SLO（ms），可按需调整
DEFAULT_QUEUE_WAIT_MS = 100 # TODO: 超参数控制最大排队等待时间，超过则直接拒绝（ms），后续使用TTFT estimator优化
KV_CACHE_BYTES_PER_TOKEN = 100000  # 每个 token 的 KV cache 估算字节数（粗略值，实际可根据模型调整）
KV_CACHE_CAPACITY = 5 * 1024 * 1024 * 1024  # KV cache 总容量上限（字节），当前环境
MAX_PENDING_QUEUE_SIZE = 100  # TODO: 超参数控制队列长度上限，超过则直接拒绝

DEFAULT_TPM_QUOTA = 1000*60  # 每分钟最大 token 数配额（全局），可按需调整
user_TPM_stats = {}  # in-memory per-user TPM

# in-memory per-user token counters (simple, not persisted)
user_token_stats = {}
user_KV_cache_stats = {}

IO_TOKEN_WEIGHT_RATIO = 2

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
    # TODO: tokenizer 级别的准确估算，但性能太差
    if not text:
        return 0
    return len(text.split())

def get_user_TPM(user_id: str) -> float:
    """获取用户的当前 TPM（每分钟 token 数），简单滑动窗口计数。"""
    now = time.time()
    window_start = now - 60  # 1 分钟窗口
    user_TPM_stats.setdefault(user_id, {"total":0, "history":[]})
    stats = user_TPM_stats[user_id]["history"]
    while stats and stats[0][0] < window_start:
        user_TPM_stats[user_id]["total"] -= stats[0][1]
        stats.pop(0)
    return user_TPM_stats[user_id]["total"]

def record_user_TPM(user_id: str, token_count: int):
    now = time.time()
    stats = user_TPM_stats.setdefault(user_id, {"total":0, "history":[]})
    stats["history"].append((now, token_count))
    stats["total"] += token_count

user_VTC_stats={}
def get_user_VTC(user_id: str):
    user_VTC_stats.setdefault(user_id,{"active":0, "VTC": 0})
    return user_VTC_stats[user_id]['VTC']

def record_user_VTC(user_id: str, token_count: int):
    user_VTC_stats.setdefault(user_id,{"active":0, "VTC": 0})
    user_VTC_stats[user_id]["VTC"]+=token_count

def get_min_user_VTC():
    flag = False
    min_VTC = 0
    for i, (key,value) in enumerate(user_VTC_stats.items()):
        if value["active"] >0:
            if not flag:
                min_VTC = value["VTC"]
                flag = True
            else:
                min_VTC = min(min_VTC,value["VTC"])
    return min_VTC

def set_user_VTC(user_id:str,token_count: int):
    user_VTC_stats.setdefault(user_id,{"active":0, "VTC": 0})
    user_VTC_stats[user_id]["VTC"]= token_count


def record_user_tokens(user_id: str, input_tokens: int = 0, output_tokens: int = 0):
    s = user_token_stats.setdefault(user_id, {"input": 0, "output": 0})
    s["input"] += input_tokens
    s["output"] += output_tokens

def record_user_KV_cache(user_id: str, kv_cache_bytes: int = 0):
    s = user_KV_cache_stats.setdefault(user_id, {"kv_cache_bytes": 0})
    s["kv_cache_bytes"] += kv_cache_bytes

historical_KV_cache = []

def historical_KVcache_estimator(input_tokens: int, max_output_tokens: int = 2048):
    """基于历史 KV cache 数据计算 P90/P95 峰值估算"""
    if len(historical_KV_cache)<10:
        # 缺少历史数据，返回保守估计
        return int((input_tokens + max_output_tokens) * KV_CACHE_BYTES_PER_TOKEN)

    # 保留最近 100 条记录以避免内存溢出
    if len(historical_KV_cache) > 100:
        historical_KV_cache.pop(0)

    # 计算 P90 作为准入的保守估计
    #TODO: 针对不同prompt长度，分桶p90。或者线性估计
    p90_kv_bytes = percentile(historical_KV_cache, 90)
    return p90_kv_bytes 

import re
METRICS_URL = f"{VLLM_BASE}/metrics"
HEADROOM_RATIO = 0.10

reserved_kv_bytes = 0
reserved_lock = asyncio.Lock()

_metric_line_re = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{.*\})?\s+([0-9\.eE\+\-]+)\s*$')
async def get_kv_cache_usage_ratio() -> float | None:
    # kv_cache_usage_ratio \in [0.0, 1.0]
    async with httpx.AsyncClient(timeout=1.0, trust_env=False) as c:
        r = await c.get(METRICS_URL)
        r.raise_for_status()
        text = r.text

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _metric_line_re.match(line)
        if not m:
            continue
        name, _, value_str = m.group(1), m.group(2), m.group(3)

        if name != "vllm:kv_cache_usage_perc":
            continue

        val = float(value_str)
        # clamp
        return max(0.0, min(1.0, val))

    return None

async def admit_or_reject(body: dict, prompt_tokens: int,pred: int = None) -> tuple[bool, int, dict]:
    """
    return (admit?, predicted_bytes, debug_info)
    """
    global reserved_kv_bytes

    if pred is None:
        pred = historical_KVcache_estimator(prompt_tokens, body.get("max_tokens", 2048))

    usage = await get_kv_cache_usage_ratio()
    if usage is None:
        return (False, pred, {"reason": "no_metrics"})

    used = int(usage * KV_CACHE_CAPACITY)
    free = int(KV_CACHE_CAPACITY - used)
    headroom = int(KV_CACHE_CAPACITY * HEADROOM_RATIO)

    async with reserved_lock:
        available = free - headroom - reserved_kv_bytes
        if pred <= available:
            reserved_kv_bytes += pred
            return (True, pred, {
                "usage": usage, "used": used, "free": free,
                "headroom": headroom, "reserved_after": reserved_kv_bytes,
                "pred": pred, "available_before": available
            })
        return (False, pred, {
            "usage": usage, "used": used, "free": free,
            "headroom": headroom, "reserved": reserved_kv_bytes,
            "pred": pred, "available": available,
            "reason": "kv_insufficient"
        })
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

@dataclass
class QueueItem:
    body: Dict[str, Any]
    arrival_ms: int
    input_tokens_est: int
    predicted_kv_bytes: int

    start_event: asyncio.Event = field(default_factory=asyncio.Event)
    reject_reason: Optional[str] = None
    admitted_dbg: Optional[Dict[str, Any]] = None

pending_queue = []  # elements: ( arrival_ms, ddl_ms, QueueItem )
queue_lock = asyncio.Lock()
queue_cv = asyncio.Condition(queue_lock)
seq_counter = 0

async def dispatcher_loop():
    global seq_counter
    while True:
        async with queue_cv:
            # after notify, check if queue is non-empty
            while not pending_queue:
                await queue_cv.wait()

            # FCFS version
            arrival_ms, ddl_ms, item = pending_queue[0]
            now = now_ms()
            if now > ddl_ms:
                pending_queue.pop(0)
                item.reject_reason = "waiting_time_exceeded"
                item.start_event.set()
                continue

            # check admit or reject
            admitted, pred, dbg = await admit_or_reject(item.body, item.input_tokens_est,item.predicted_kv_bytes)
            if not admitted:
                # lack of resources, try again later
                await queue_cv.wait()
                continue
  
            # admitted 
            pending_queue.pop(0)
            item.admitted_dbg = {"predicted_kv_cache_bytes": pred, **dbg}

        #setup stream generator
        await sema.acquire()
        item.start_event.set()

async def dispatcher_loop_TPM():
    global seq_counter
    while True:
        async with queue_cv:
            # after notify, check if queue is non-empty
            while not pending_queue:
                await queue_cv.wait()

            # clean up expired requests.
            now = now_ms()
            i = 0
            while i < len(pending_queue):
                arrival_ms, ddl_ms, item = pending_queue[i]
                if now > ddl_ms:
                    pending_queue.pop(i)
                    item.reject_reason = "waiting_time_exceeded(TPM)"
                    item.start_event.set()
                    continue
                i += 1
            if not pending_queue:
                continue

            # check admit or reject
            selected_item = None
            i = 0
            while i < len(pending_queue):
                arrival_ms, ddl_ms, item = pending_queue[i]
                user_id = item.body.get("user")
                if not user_id:
                    pending_queue.pop(i)
                    item.reject_reason = "missing_user_id"
                    item.start_event.set()
                    continue
                user_TPM = get_user_TPM(user_id)
                #TODO: 限制超过TPM用户，直到他的请求超时，或者TPM下降
                if user_TPM + item.input_tokens_est <= DEFAULT_TPM_QUOTA:
                    admitted, pred, dbg = await admit_or_reject(item.body, item.input_tokens_est,item.predicted_kv_bytes)
                    # if admitted 
                    if admitted:
                        pending_queue.pop(i)
                        item.admitted_dbg = {"predicted_kv_cache_bytes": pred, **dbg}
                        selected_item = item
                        break
                i += 1

        #setup stream generator
        if selected_item is not None:
            await sema.acquire()
            selected_item.start_event.set()
        else:
            # lack of TPM or resources, waiting for new request/ request exit/ timer tick.
            await queue_cv.wait()
            continue

async def dispatcher_loop_VTC():
    global seq_counter
    while True:
        print("alive tick...")
        async with queue_cv:
            # after notify, check if queue is non-empty
            while not pending_queue:
                await queue_cv.wait()

            # clean up expired requests.
            now = now_ms()
            i = 0
            while i < len(pending_queue):
                arrival_ms, ddl_ms, item = pending_queue[i]
                if now > ddl_ms:
                    pending_queue.pop(i)
                    item.reject_reason = "waiting_time_exceeded(VTC)"
                    item.start_event.set()
                    continue
                i += 1
            if not pending_queue:
                continue

            # check admit or reject
            selected_idx = 0
            min_VTC = 1<<32
            for i in range(len(pending_queue)):
                arrival_ms, ddl_ms, item = pending_queue[i]
                user_id = item.body.get("user")
                if not user_id:
                    pending_queue.pop(i)
                    item.reject_reason = "missing_user_id"
                    item.start_event.set()
                    continue

                if(get_user_VTC(user_id)<min_VTC):
                    selected_idx=i
                    min_VTC = get_user_VTC(user_id)
            arrival_ms, ddl_ms,item = pending_queue[selected_idx]
            # check admit or reject
            admitted, pred, dbg = await admit_or_reject(item.body, item.input_tokens_est,item.predicted_kv_bytes)
            if not admitted:
                # lack of resources, try again later
                await queue_cv.wait()
                continue
  
            # admitted 
            pending_queue.pop(selected_idx)
            item.admitted_dbg = {"predicted_kv_cache_bytes": pred, **dbg}

        #setup stream generator
        await sema.acquire()
        item.start_event.set()
                 

async def deadline_tick():
    # timer for waiting time exceeded rejection
    while True:
        await asyncio.sleep(0.05)  # 50ms
        async with queue_cv:
            queue_cv.notify_all()

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
@app.on_event("startup")
async def _startup():
    asyncio.create_task(dispatcher_loop_VTC())
    asyncio.create_task(deadline_tick())

@app.post("/v1/chat/completions")
async def chat_completions(req: Request):
    body: Dict[str, Any] = await req.json()
    stream = bool(body.get("stream", False))

    # no pending queue; reject immediately if at capacity
    # if sema.locked() and sema._value == 0:
    #     return JSONResponse(
    #         status_code=429,
    #         content={"error": {"message": "Rejected by admission control (over capacity)."}}
    #     )

    # await sema.acquire()
    # admitted_at = now_ms()
    # admitted, pred, dbg = await admit_or_reject(body, estimate_tokens(body.get("messages", [{}])[0].get("content", "")))
    # if not admitted:
    #     sema.release()
    #     print(json.dumps({"admitted": False, **dbg}, ensure_ascii=False))
    #     return JSONResponse(
    #         status_code=429,
    #         content={"error": {"message": "Rejected by admission control (KV cache constraints)."}}
    #     )
    # else:
    #     print(json.dumps({"admitted_at_ms": admitted_at, "predicted_kv_cache_bytes": pred, **dbg}, ensure_ascii=False))

    # async def _release():
    #     sema.release()

    item:QueueItem = QueueItem(
        body=body,
        arrival_ms=now_ms(),
        input_tokens_est=estimate_tokens(body.get("messages", [{}])[0].get("content", "")),
        predicted_kv_bytes=historical_KVcache_estimator(estimate_tokens(body.get("messages", [{}])[0].get("content", "")), body.get("max_tokens", 2048))
    )
    # new request enqueue and notify dispatcher
    global seq_counter
    async with queue_cv:
        if len(pending_queue) >= MAX_PENDING_QUEUE_SIZE: # TODO: 超参数控制队列长度上限，超过则直接拒绝
            return JSONResponse(
                status_code=429,
                content={"error": {"message": "Rejected by admission control (queue full)."}}
            )
        # else:
        #     print(len(pending_queue), seq_counter)
        user_id = body.get("user")
        #new user, align the VTC to minimal active user.
        get_user_VTC(user_id)
        if user_VTC_stats[body.get("user")]["active"]== 0:
            set_user_VTC(user_id,get_min_user_VTC())
        pending_queue.append((now_ms(), now_ms() + DEFAULT_QUEUE_WAIT_MS, item))
        seq_counter += 1
        queue_cv.notify_all()

    async def stream_generator(item: QueueItem):
        client = httpx.AsyncClient(timeout=None, trust_env=False)

        global reserved_kv_bytes
        start = time.perf_counter()
        first_chunk_time = None
        last_chunk_time = None
        chunk_intervals_ms = []
        user_id = None
        kv_cache_bytes_est = 0
        input_tokens_est = 0
        
        try:
            # extract user id
            user_id = body.get("user")
            #基于历史的KV cache 估算 P90/P95
            peak_kv_cache_est = item.predicted_kv_bytes
            input_tokens_est = item.input_tokens_est
            kv_cache_bytes_est = int(input_tokens_est * KV_CACHE_BYTES_PER_TOKEN)
            # record input tokens immediately
            record_user_tokens(user_id, input_tokens=input_tokens_est)
            record_user_KV_cache(user_id, kv_cache_bytes=kv_cache_bytes_est)
            record_user_TPM(user_id, input_tokens_est)
            record_user_VTC(user_id, input_tokens_est)
            user_VTC_stats[user_id]['active']+=1

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
                                            # update reserved KV cache
                                            if kv_cache_bytes_est < peak_kv_cache_est:
                                                async with reserved_lock:
                                                    reserved_kv_bytes = max(0,reserved_kv_bytes - min(out_toks * KV_CACHE_BYTES_PER_TOKEN, peak_kv_cache_est - kv_cache_bytes_est))
                                            kv_cache_bytes_est += out_toks * KV_CACHE_BYTES_PER_TOKEN
                                            record_user_tokens(user_id, output_tokens=out_toks)
                                            record_user_KV_cache(user_id, out_toks * KV_CACHE_BYTES_PER_TOKEN)
                                            record_user_TPM(user_id, out_toks* IO_TOKEN_WEIGHT_RATIO)
                                            record_user_VTC(user_id, out_toks* IO_TOKEN_WEIGHT_RATIO)
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
            sema.release()
            user_VTC_stats[user_id]['active']=max(0,user_VTC_stats[user_id]['active']-1)
            # Request terminated, notify dispatcher and check the feasibility of new request.
            async with queue_cv:
                queue_cv.notify_all()
            record_user_KV_cache(user_id, -kv_cache_bytes_est)
            historical_KV_cache.append(kv_cache_bytes_est)
            # release reserved KV cache
            if kv_cache_bytes_est < peak_kv_cache_est:
                async with reserved_lock:
                    reserved_kv_bytes = max(0,reserved_kv_bytes - ( peak_kv_cache_est - kv_cache_bytes_est))

            end = time.perf_counter()
            ttft_ms = None if first_chunk_time is None else (first_chunk_time - start) * 1000
            e2e_ms = (end - start) * 1000

            # assemble per-request metrics including KV cache estimate and SLO result
            metrics = {
                "admitted_at_ms": item.arrival_ms,
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
        # 1) 等 dispatcher 放行或拒绝
        await item.start_event.wait()

        # 2) 如果是被拒绝唤醒：直接返回 SSE 错误并结束
        if item.reject_reason:
            return JSONResponse(
        status_code=429,
        content={"error": {"message": item.reject_reason}})
            
        # --- Forward to vLLM ---
        return StreamingResponse(stream_generator(item=item), media_type="text/event-stream")
    else:
        pass

def percentile(arr:list, p):
    if arr is None or len(arr) == 0:
        return 0
    x = np.array(arr)
    return np.percentile(x, p)
