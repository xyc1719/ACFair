import re

_metric_line_re = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{.*\})?\s+([0-9\.eE\+\-]+)\s*$')

line = 'vllm:kv_cache_usage_perc{engine="0",model_name="Qwen/Qwen2.5-3B-Instruct"} 0.0'

m = _metric_line_re.match(line)
if not m:
    exit(429)
name, _, value_str = m.group(1), m.group(2), m.group(3)
print (name,value_str)