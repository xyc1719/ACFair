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

## measure and check component

draft: at admission_proxy, monitoring vLLM performance(KV cache, input/output token and cost). check if it's satisfy the SLO.

have to skim the vLLM document and query AI for help.
