# ACFair

# Manual

## 启动 OpenAI-compatible server

```bash
./vllm.sh
```

或

```bash
conda activate vllm
export VLLM_ATTENTION_BACKEND=TRITON_ATTN
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-3B-Instruct \
  --port 8000 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096 \
  --max-num-seqs 32 \
  --kv-cache-memory-bytes 5g

```

通过最后四个参数，以调整占用。(--kv-cache-memory-bytes存在时，忽略--gpu-memory-utilization)

## Admission Proxy

```bash
pip install fastapi uvicorn httpx
uvicorn admission_proxy:app --host 0.0.0.0 --port 9000
```

此时你的客户端应该打到 `http://127.0.0.1:9000/v1/chat/completions`，proxy 再转发到 `:8000` 的 vLLM。

## Load Generator

cli:

```bash
python load_gen.py
```

**Succeed!**

---

## KV cache figuration

QWen2.5-3B-Instructor on RTX5070Ti

max Available KV cache memory: 7.89 GiB

max GPU KV cache size: 229,888 tokens

max num of GPU blocks: 14368

上下文长度和最大并行请求，同时影响KV capacity需求和计算图等其他开销。
