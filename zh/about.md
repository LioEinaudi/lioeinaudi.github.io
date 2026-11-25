---
layout: page
title: 关于
permalink: /zh/about/
lang: zh
translation_url: /about/
---

<img class="about-avatar" src="{{ '/assets/avatar.jpg' | relative_url }}" alt="赵锦彬" />

我是**赵锦彬**，网名 **Lio Einaudi**。我做 LLM 与多模态推理系统研究——GPU kernel、AI 编译器图优化、确定性推理。开源工作主要在 [vLLM](https://github.com/vllm-project/vllm/pulls?q=author%3ALioEinaudi)。

---

## 实习经历

- **NIO（蔚来）** — 推理系统研究实习生（即将入职）。
- **Dexmal（原力灵机）**，具身智能创业公司 — AI Infra 算法实习生，2026 年 6 月 10 日至 11 月 23 日。

---

## 开源贡献

**[vLLM](https://github.com/vllm-project/vllm/pulls?q=author%3ALioEinaudi)** — batch-invariant 确定性推理性能与正确性（6 个 PR 合入 `main`）：

- **端到端图执行优化**（[#57586](https://github.com/vllm-project/vllm/pull/57586)）：在 RTX PRO 6000 上将 Qwen3-1.7B/4B/8B 的整批生成延迟降低 **54.2%–63.6%**，相对普通推理的额外延迟从 **128%–195% 降至 4%–7%**——为支持的 batch-invariant 配置启用 breakable CUDA Graph，在捕获时按真实 `M` 选择矩阵乘配置，避开 `torch.compile` 追踪阶段用占位 `M` 固化配置的问题。
- **算子调优与多卡扩展**（[#53247](https://github.com/vllm-project/vllm/pull/53247)、[#57456](https://github.com/vllm-project/vllm/pull/57456)、[#58495](https://github.com/vllm-project/vllm/pull/58495)）：在 H20 上将 Qwen3-1.7B 的 batch 8 端到端延迟降低 **29.4%**（吞吐 +3.4%），在 RTX 4090D 上降低 **23.0%**（吞吐 +12.0%），并将双卡 RTX PRO 6000 上 Qwen3-8B 的整批生成延迟从 **3.00s 降至 1.33s**——在跨 batch 位级一致约束下，在 H20、RTX 4090D 和 RTX PRO 6000 上按 M 分桶调优 Triton persistent matmul，并为 Blackwell SM120 补充 TP=2/4/8 的分片形状（[#53247 调优笔记](/zh/2026/08/30/cutting-the-determinism-tax.html)）。
- **正确性与测试修复**（[#56377](https://github.com/vllm-project/vllm/pull/56377)、[#53692](https://github.com/vllm-project/vllm/pull/53692)）：在 RTX PRO 6000 TP=4 复现中将采样 token logprob 的不一致数从 **1522/1536 降至 0**，并在同引擎 A/B 对照中把 GLM-4-9B-Chat 从 59/64 修到 **64/64**（RTX PRO 6000 上 `test_batch_invariance.py` 全部 **19** 项用例通过）——在 batch-invariant 模式下禁用不兼容的 SP/async TP 并用 TP=4 验证修复，集成社区 @Evangade 的 token ID 前缀补丁（[#43317](https://github.com/vllm-project/vllm/pull/43317)），补充 GLM-4/Phi-4 验证与文档（[#53692 排查笔记](/zh/2026/11/25/glm-4-token-prefix.html)）。

---

## 教育经历

- **中央民族大学** — 计算机科学与技术，2028 届。

---

## 获奖

- **蓝桥杯全国软件大赛** — C/C++ 大学 A 组 全国三等奖，2025。

---

## 工作之外

中长跑、钢琴、《进击的巨人》。

---

## 联系方式

- GitHub：[@LioEinaudi](https://github.com/LioEinaudi)
- 邮箱：[Jinbin.Zhao.Dev@gmail.com](mailto:Jinbin.Zhao.Dev@gmail.com)
