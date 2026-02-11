# load_gen.py
'''
AI generated content without human review.
'''
import asyncio, time, json
import httpx
from datasets import load_dataset
import pandas as pd

PROXY = "http://127.0.0.1:9000/v1/chat/completions"
CSV_PATH = "../data/BurstGPT_simple.csv" # only 3k lines 
MAX_INFLIGHT = 200

async def one(client: httpx.AsyncClient, i: int, stream: bool = True):
    #message body,including prompt, model, token length and streaming flag.
    body = {
        "user": f"user_{i%10}",  # 模拟 10 个用户轮流发请求 TODO: 可根据实际 分布 结构调整 user_id 的提取方式
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "messages": [{"role":"user","content": f"Give me a short tip about systems research. (req={i})"}],
        "max_tokens": 2048,
        "stream": stream,
    }
    t0 = time.perf_counter()
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
                    if ttft is None:
                        ttft = (time.perf_counter() - t0) * 1000
                    if payload.strip() == "[DONE]":
                        dt = (time.perf_counter() - t0) * 1000
                        break

        print(json.dumps({"i": i, "ttft_ms": ttft, "e2e_ms": dt}, ensure_ascii=False))
    else:
        r = await client.post(PROXY, json=body)
        dt = (time.perf_counter() - t0) * 1000
        print("status", r.status_code, "e2e_ms", dt)

async def schedule_one(client: httpx.AsyncClient, sem: asyncio.Semaphore, i: int, delay_s: float):
    if delay_s > 0:
        await asyncio.sleep(delay_s)

    # 并发控制
    async with sem:
        await one(client, i, stream=True)


async def main():
    df = pd.read_csv(CSV_PATH, nrows=3000)

    if "Timestamp" not in df.columns:
        raise ValueError(f"Can't find the column named 'Timestamp' in CSV。当前列: {list(df.columns)}")

    ts = df["Timestamp"]
    t0 = ts[0]

    sem = asyncio.Semaphore(MAX_INFLIGHT)

    async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
        tasks = []
        for idx, t in enumerate(ts):
            # default (t - t0) / 100
            delay_s = (t - t0) / 1000.0
            tasks.append(asyncio.create_task(schedule_one(client, sem, idx, delay_s)))

        await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
