conda activate vllm
export VLLM_ATTENTION_BACKEND=TRITON_ATTN
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-3B-Instruct \
  --port 8000 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096 \
  --max-num-seqs 32 \
  --kv-cache-memory-bytes 5g