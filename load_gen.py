# load_gen.py
'''
AI generated content without human review.
'''
import asyncio, time, json
import httpx
from datasets import load_dataset

PROXY = "http://127.0.0.1:9000/v1/chat/completions"
NUM_REQUESTS = 32

async def one(i: int, stream=True):
    body = {
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "messages": [{"role":"user","content": f"Give me a short tip about systems research. (req={i})"}],
        "max_tokens": 128,
        "stream": stream,
    }
    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
        if stream:
            ttft = None
            async with client.stream("POST", PROXY, json=body) as r:
                if r.status_code != 200:
                    print("reject", i, r.status_code, await r.aread())
                    return
                async for line in r.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        payload = line[len("data: "):]
                        # print(payload, flush=True)
                        if ttft is None:
                            ttft = (time.perf_counter() - t0) * 1000
                        if payload.strip() == "[DONE]":
                            break

            print(json.dumps({"i": i, "ttft_ms": ttft}, ensure_ascii=False))
        else:
            r = await client.post(PROXY, json=body)
            dt = (time.perf_counter() - t0) * 1000
            print("status", r.status_code, "e2e_ms", dt)

async def main():
    # 并发 请求
    await asyncio.gather(*[one(i, stream=True) for i in range(NUM_REQUESTS)])

asyncio.run(main())
