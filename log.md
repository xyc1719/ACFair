# TODOLIST:

1. Collection of Time-Request Distribution Dataset
2. rewrite load_gen.py (no user/prompt length constraint)
3. github version control
4. measure and check component.
5. predict output/total token and decision logic of whether the system is underload/ nearly-overloaded/ overloaded.
6. simple admission control policy.

## Collection of Time-Request Distribution Dataset

- WildChat
- Azure
- BurstGPT
- mooncake

None of them has user label, and none of them has conversation content except WildChat.

## rewrite load_gen.py

generate request follow Timestamp/100.0 ms, all request prompts are barely same.

### future work

- systhesis user label (random/ specific workload user)
- a knob to control fixed req/s
  - The density of req/s changes over time. We obtain intuitive results by running the following script.split_data.py and plot.py
- generate request prompt with corresponding length and promise the output length is not less than it.

## Github upload

succeed!

---

**0208**

## measure and check component

status monitor of input/output token usage and KV cache occupation at user level.

## Output token prediction

According to historical distribution, make the prediction at P90/P95.

---

simple admission control policy
