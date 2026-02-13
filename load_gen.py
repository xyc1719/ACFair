# load_gen.py
'''
AI generated content without human review.
'''
import asyncio, time, json
import httpx
from datasets import load_dataset
import pandas as pd
import numpy as np

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
        # input tokens < base prompt(16 tokens). rare and can be ignored.
        return s
        # raise ValueError("Base prompt already exceeds target_tokens")

    need = target_tokens - cur
    # repeat PAD_UNIT until reaching target_tokens
    s += PAD_UNIT * need
    # TODO: high prefix hit rate can cause performance issues, consider adding some randomness to the padding.

    # # fine-tune
    # while len(tok.encode(s, add_special_tokens=False)) > target_tokens:
    #     s = s[:-1]
    # while len(tok.encode(s, add_special_tokens=False)) < target_tokens:
    #     s += PAD_UNIT

    return s

def make_zipf_probs(n_users: int, s: float) -> np.ndarray:
    ranks = np.arange(1, n_users + 1, dtype=np.float64)
    w = 1.0 / np.power(ranks, s)
    return w / w.sum()

def assign_users_zipf(n_reqs: int, n_users: int = 1000, s: float = 1.2, seed: int = 1):
    rng = np.random.default_rng(seed)
    p = make_zipf_probs(n_users, s)
    # 返回 0..n_users-1 的 user index
    return rng.choice(n_users, size=n_reqs, p=p)

async def one(client: httpx.AsyncClient, i: int, item: dict, stream: bool = True):
    #message body,including prompt, model, token length and streaming flag.
    # #old-version
    # body = {
    #     "user": f"user_{i%10}",  # 模拟 10 个用户轮流发请求
    #     "model": "Qwen/Qwen2.5-3B-Instruct",
    #     "messages": [{"role":"user","content": f"Give me a short tip about systems research. (req={i})"}],
    #     "max_tokens": 2048,
    #     "stream": stream,
    # }
    body = {
        "user": f"user_{item['user']}",  # 模拟用户zipf分布
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
                print("user", item["user"], "reject", i, r.status_code, await r.aread())
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

        print(json.dumps({"i": i, "user": item["user"], "ttft_ms": ttft, "e2e_ms": dt}, ensure_ascii=False))
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
    
    #environment parameters for #user and skewness of user distribution
    N_USERS = 20
    ZIPF_S = 1.2
    user_idx = assign_users_zipf(len(df), n_users=N_USERS, s=ZIPF_S, seed=42)
    print(user_idx)

    t0 = df["Timestamp"][0]

    sem = asyncio.Semaphore(MAX_INFLIGHT)

    async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
        tasks = []
        for idx, row in df.iterrows():
            in_len = int(row["Request tokens"])
            out_len = int(row["Response tokens"])
            # default (t - t0) / 100
            delay_s = float((row["Timestamp"] - t0) / 1000.0)
            if in_len == 0 and out_len == 0:
                continue
            tasks.append(asyncio.create_task(schedule_one(client, sem, idx, {"input_len": in_len, "output_len": out_len, "user": int(user_idx[idx])}, delay_s)))

        await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
