# component

- Collection of Time-Request Distribution Dataset
- Rewrite load_gen.py as a data-driven program
- measure and check part of running result(SLO/KV cache/statics storage).

# SOLA-like KVcache-aware control

- predict future KV cache occupation
- The logic for determining feasibility according to future occupation.

# Admission Control policy

- simple pending queue
- TPM and VTC admission control
- Measurement-Driven Fair Admission Control
- several baseline: FCFS(vLLM/Sarathi)/TPM/VTC/*HTB*/...

# User label synthesis

no label due to privacy protection.
