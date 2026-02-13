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

BASE_PROMPT = "You are a helpful assistant.\nTask: echo nothing from padding.\n"
PAD_UNIT = " a"
def synth_prompt(target_tokens: int) -> str:

    s = BASE_PROMPT + "\nPADDING:\n"
    # cur = len(tok.encode(s, add_special_tokens=False))
    cur = 16  # already measured, to accelerate the process.
    if cur > target_tokens:
        # backbone too long, 16 tokens.
        return s[:target_tokens]
        # raise ValueError("Base prompt already exceeds target_tokens")

    need = target_tokens - cur
    # 朴素填充：重复 PAD_UNIT，然后微调（最后用 while 精确对齐）
    s += PAD_UNIT * need

    # # fine-tune
    # while len(tok.encode(s, add_special_tokens=False)) > target_tokens:
    #     s = s[:-1]
    # while len(tok.encode(s, add_special_tokens=False)) < target_tokens:
    #     s += PAD_UNIT

    return s

async def one(client: httpx.AsyncClient, i: int, item: dict, stream: bool = True):
    #message body,including prompt, model, token length and streaming flag.
    body = {
        "user": f"user_{i%10}",  # 模拟 10 个用户轮流发请求 TODO: 可根据实际 分布 结构调整 user_id 的提取方式
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "messages": [{"role":"user","content": synth_prompt(item["input_len"])}],
        "max_tokens": item["output_len"],
        "ignore_eos": True,
        "min_tokens": item["output_len"],
        "stream": stream
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

async def schedule_one(client: httpx.AsyncClient, sem: asyncio.Semaphore, i: int,item:dict,delay_s: float):
    if delay_s > 0:
        await asyncio.sleep(delay_s)

    # 并发控制
    async with sem:
        await one(client, i,item, stream=True)


async def main():
    df = pd.read_csv(CSV_PATH, nrows=200)

    if "Timestamp" not in df.columns:
        raise ValueError(f"Can't find the column named 'Timestamp' in CSV。当前列: {list(df.columns)}")

    ts = df["Timestamp"]
    t0 = ts[0]

    sem = asyncio.Semaphore(MAX_INFLIGHT)

    async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
        tasks = []
        for idx, item in enumerate(df.itertuples()):
            # print(item[3],item[4],item[5]) # Request_tokens, Response_tokens, Total_tokens
            # default (t - t0) / 100
            delay_s = (item.Timestamp - t0) / 1000
            if item [3]==0 and item[4]==0:
                continue
            tasks.append(asyncio.create_task(schedule_one(client, sem, idx,{"input_len": item[3], "output_len": item[4]}, delay_s)))

        await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
