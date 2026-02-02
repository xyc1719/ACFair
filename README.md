# ACFair

# Manual

## 启动 OpenAI-compatible server

```bash
export VLLM_ATTENTION_BACKEND=TRITON_ATTN
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-3B-Instruct \
  --port 8000 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096 \
  --max-num-seqs 16

```

通过最后三个参数，以调整占用

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
