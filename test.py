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

if __name__ == "__main__":
    # 例子：把这里改成你正在运行的模型名
    info = estimate_kv_bytes_per_block(
        model_name_or_path="Qwen/Qwen2.5-3B-Instruct",
        block_size=16,
        dtype="fp16",
        tensor_parallel_size=1,
    )
    for k, v in info.items():
        print(f"{k}: {v}")