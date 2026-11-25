---
layout: post
title: ".Here 怎么让 GLM-4 的一致性测试误报？"
date: 2026-11-25
lang: zh
permalink: /zh/2026/11/25/glm-4-token-prefix.html
translation_url: /2026/11/25/glm-4-token-prefix.html
---

*五个字符 `.Here` 怎么让 GLM-4 在两张 GPU 上都栽了跟头，以及怎么靠保留 token ID 前缀把同一引擎从 59/64 变成 64/64（[PR #53692](https://github.com/vllm-project/vllm/pull/53692)）。*

本来打算把 GLM-4 加进 vLLM batch invariance 的已验证模型名单，结果跑完官方测试，我又把它从 PR 里撤了下来。用 `FLASH_ATTN` 对比 decode 和 prefill 时，RTX 4090D 和 H20 都栽在同一条 prompt 上：位置 0 到 10 全过，位置 11 选出了不同的 token，位置 12 到 15 选中的 token 相同，但对应的 logprob 最大差了约 0.16。当时我直接在[两卡测试记录](https://github.com/vllm-project/vllm/pull/53692#issuecomment-5472201949)里留了句 “GLM has a real failure”。

后来查下去才发现是误判，坑就埋在输出里的 `.Here` 这五个字符上。

## 位置 11 到底在比什么？

这项测试先让 GLM-4-9B-Chat 做一次贪心生成，记下每一步选中的 token ID 和它的 logprob，再针对每个位置单独发一个请求——把此前的前缀当作 prompt 交给模型，只让它生成一个 token（`max_tokens=1`）。测试先核对这个 token 的 ID，只有 ID 一致时才检查对应的 FP32 logprob 是否精确相等（见[合入版本的测试源码](https://github.com/vllm-project/vllm/blob/89439db7276cf165377c088496e20098c8137727/tests/v1/determinism/test_batch_invariance.py#L779-L867)）。

比如查到位置 11（从 0 开始算），就拿原始 prompt 拼上前 11 个已生成的 token 当作 prefill 输入，让模型重新预测第 12 个 token。这个对比成立的前提，是两次预测看到的 token ID 序列完全一样。

旧测试在这一步绕了个弯。为了拿到位置 11 之前的文本，它让模型从原始 prompt 再生成 11 步，把输出的字符串拼回原始 prompt，再交给 vLLM。这样一来，整个前缀都又过了一遍 tokenizer。

## 同一个 `.Here`，两套 token ID

出问题的那条回答里刚好有一段 `…tree.Here`。把句点附近的 token 单独拎出来，区别就看得见了：

| 前缀来源 | 文本 | 对应的 token ID |
| --- | --- | --- |
| 直接沿用 decode 生成的 token | `.Here` | `[13, 8419]` |
| 转成文本拼回 prompt，再用 GLM 的 tokenizer 重编码 | `.Here` | `[89329]` |

decode 时，GLM 先生成了句点 `.`（13），下一步才生成 `Here`（8419）。句点一旦选中，就已经占掉序列里的一个位置。可重编码时，BPE tokenizer 一次看到整段字符串，正好能把 `.` 和后面的四个字母合并成词表里的单个 token（89329）。文本打出来都是 `.Here`，底层的切分早就变了。

验证位置 11 时，重建的前缀第一次把 `.Here` 整段包了进去。decode 是在 `[…, 13, 8419]` 后面接着生成，prefill 却是在 `[…, 89329]` 后面接着生成。计算再确定，喂进去的前缀都不一样了，同一个模型也可能选出不同的下一个 token；后面几步即便碰巧选中了同一个 token，它的 logprob 对应的依然是两段不同的历史。

## 只换前缀，重跑 64 个位置

为了验证这次合并能不能解释那 5 处失败，我在同一张 RTX 4090D、同一个 vLLM 引擎实例上做了组对照：两边都用编译模式、`FLASH_ATTN` 和 `VLLM_BATCH_INVARIANT=1`，固定 4 条 prompt，每条查 16 个生成位置，只改 prefill 前缀的构造方式（见[逐位置 A/B 记录](https://github.com/vllm-project/vllm/pull/53692#issuecomment-5486601496)）。

| prefill 输入 | 选中同一 token 且其 FP32 logprob 相等的位置数 |
| --- | --- |
| 把生成文本拼回 prompt 后重新编码 | 59 / 64 |
| 原始 prompt 的 token ID + 实际生成的 token ID 前缀 | **64 / 64** |

第一组挂掉的 5 个位置，全都落在第 0 条 prompt 的位置 11–15，正好就是重编码后前缀 ID 错开的那 5 步。保留 token 边界后，这 5 处差异全部消失，另外 59 处仍然通过。

改动的关键是从第一次生成的结果里同时取出 prompt 和输出的 ID。验证位置 `i` 时，切片恰好停在待预测的 token 前面：

```python
prompt_ids = list(decode_output.prompt_token_ids or [])
generated_ids = decode_output.outputs[0].token_ids
prefix_ids = prompt_ids + list(generated_ids[:i])
prefill_prompt = TokensPrompt(prompt_token_ids=prefix_ids)
```

翻了下社区记录，@xRay2016 之前就在 Qwen 的 tokenizer 上碰过同一类问题，当时合在一起的两个字符是 `.` 和 `#`（见[最早的复现评论](https://github.com/vllm-project/vllm/issues/27433#issuecomment-4358595058)）；@Evangade 随后交过直接用 token ID 构造前缀的[原始补丁](https://github.com/vllm-project/vllm/pull/43317)。这次刚好在 GLM 上撞出了一个具体的复现样本，做完对照后，我按维护者的建议保留了原作者署名，把修复带进了[这次 PR](https://github.com/vllm-project/vllm/pull/53692)。

之后在 RTX PRO 6000 上用修复后的测试跑 GLM-4-9B-Chat，整个 `test_batch_invariance.py` 文件的 19 个用例全部通过，GLM 也回到了已验证模型名单里。

以后再碰上两条推理路径的 logprob 对不上，我第一件事就是先把出错位置的前缀按 token ID 打出来。只看字符串的话，两边日志一模一样，都是 `…tree.Here`，根本看不出破绽。
