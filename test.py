from transformers import AutoTokenizer

# tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

BASE = "You are a helpful assistant.\nTask: echo nothing from padding.\n"
PAD_UNIT = " a"

def synth_prompt(target_tokens: int) -> str:

    s = BASE + "\nPADDING:\n"
    # cur = len(tok.encode(s, add_special_tokens=False))
    cur = 16  # already measured, to accelerate the process.
    if cur > target_tokens:
        # backbone too long, 16 tokens.
        raise ValueError("Base prompt already exceeds target_tokens")

    need = target_tokens - cur
    # 朴素填充：重复 PAD_UNIT，然后微调（最后用 while 精确对齐）
    s += PAD_UNIT * need

    # # fine-tune
    # while len(tok.encode(s, add_special_tokens=False)) > target_tokens:
    #     s = s[:-1]
    # while len(tok.encode(s, add_special_tokens=False)) < target_tokens:
    #     s += PAD_UNIT

    return s

if __name__ == "__main__":
    for target in [512, 1024, 1536, 2048]:
        prompt = synth_prompt(target)
        # print(f"Target: {target}, Actual: {len(tok.encode(prompt, add_special_tokens=False))}")
        print(f"Target: {target}, Actual: {len(prompt.split())}")