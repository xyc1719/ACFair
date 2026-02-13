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

# 0211

## TODOLIST

- 矫正KV cache和token usage的实时性监测
- 固定vLLM KV cache memory bytes，能正确的监测KVcache占用比，并进行可行性准入。
- 准入队列，简单的等待超时(429)
  - 超参数
  - future work可使用TTFT estimator，根据历史开销进行优化（EMA平滑）。
- VTC策略实现
- TPM策略实现

## KV cache和token usage实时性监测

asyncio/await 为轮询实现，伪多线程，不存在写入竞争，over。

## 固定KV cache，实现可行性准入

通过 `--kv-cache-memory-bytes 5g`限制KV cache占用，warm up时会提示对应的max_num_token参数（135,632 tokens）。

设置固定KV cache大小，和max_model_len和max_num_seqs的乘积有关，但是该乘积也会增加其他开销（如计算图）。大致为memory = model + KV cache(context, reqs) + other(context, reqs)。

`kv_cache_usage_perc` 指标表示KV cache整体占用。

非防止临时大量请求准入，导致KV cache占用偏差，引入reserved_kv_cache占位并随request处理而更新。**隐含风险**：reserved_lock同步速度较慢。

可行性准入条件为`free - headroom - reserved >= pred`。

### 测试

在不改变prompt输入时，需要调整部分代码使得，KV constraint变得明显。

1. vllm: `--no-enable-prefix-caching` KV cache 5g ->0.5g
2. admission_proxy: KV cache 5g -> 0.5g
3. load_gen: time /100 -> time /1000

但在开始一些prompt之后，KV cache不再成为约束点，可能的原因是total token length太短（使用估计器后跑不满占用）

## 准入队列

QueueItem + dispatcher(Item in / finish) + deadline_tick(50 ms recall dispatcher)

`dispatcher_loop`仅需在新请求进队和请求完成时，检查准入可行性。

`deadline_tick`定期清除等待超时的请求。

对于新任务，先进队再敲钟，若超时则直接返回，否则正常执行 `stream_generator(item)`

**TODO**:*超时等待时间可以用TTFT estimator优化*

---

*remain:

- +变长prompt
  - 对应数据集
  - 早停策略或WildChat
  - 合成Prompt
- -用户标签合成
  - 好的分布
  - 分析论证的文档（英文两段），附带引用文献
  - load_gen.py调整
- +动态TPM (优化策略的实现)，测试
- 针对4个指标，记录实验结果并优化
- outline流程示意图

# 0212

## TODOLIST

- 准入队列 3 over
- VTC策略实现 2 over
- TPM策略实现 1 over
- 合成prompt
- *用户标签合成 3
- 动态TPM策略实现 3

## 准入队列

见上文

## TPM策略实现

超出quota时等待，否则判断KVcache是否允许准入。超额用户的请求可能被超时清退。

## VTC策略实现

每次选取VTC最小值，判断是否允许准入，否则继续等待。不公平用户请求可能被超时清退。

对于新用户，和最小的活跃用户对齐。

**风险**：公平性未进行充分测试。

**已知bug**： *load_gen.py启动两次，第一次绝大部分请求被接受，第二次绝大部分请求都被拒绝了。*

    非sema问题

---

## 0213

## TODOLIST

- 合成Prompt 1.5
- 用户标签合成 3
- 1400
- log&周报
  - 实验进度
  - 用户标签合成-分析报告
  - 推辞
  - 未来工作方向

## 合成Prompt

对于这类缺少对应Prompt的数据，通用的做法是合成。并限制对单个request限制最大output len

load_gen调整,合成input len等长的Prompt，拒绝EOS早停，并限制对单个request限制max_tokens和min_tokens.

## 用户标签合成

arxiv data storage fair -> twitter/... 分布分析，学习分析思路

=>查找分析大模型服务的类似用户请求分布 -)**{分布z， 分布分析文档（英文两段）， 引用}**

=> rewrite load_gen.py

类似LLM serving 用户请求分布 --> BurstGPT 3.2 user pattern figure.7

分布类型 zipf，无明确参数。

## DEBUG

- admission_proxy 中 reserved_kv_cache存在泄漏。
