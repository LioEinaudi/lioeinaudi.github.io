---
layout: page
title: 关于
permalink: /zh/about/
lang: zh
translation_url: /about/
---

<img class="about-avatar" src="https://github.com/LioEinaudi.png" alt="赵锦彬" />

我是**赵锦彬**,网名 **Lio Einaudi**。我做 LLM 推理性能优化——GPU kernel、MoE 调优、确定性(batch-invariant)推理——只有一条铁律:benchmark 必须跑在最新 release 上、必须通过正确性校验,否则不算数。

---

## 工作成果

- **Batch-invariant GEMM 调优** — [vLLM #53247](https://github.com/vllm-project/vllm/pull/53247),已合并:那个吃掉 batch-invariant 模式 79% kernel 时间的 Triton GEMM,在不变性约束下(`BLOCK_K` 固定)完成调优。**H20 上端到端延迟 −29%(1.42×)**,确定性逐位(bit-for-bit)保持。完整故事见博文:[Cutting the Determinism Tax]({% post_url 2026-08-30-cutting-the-determinism-tax %})(英文)。
- **Fused-MoE tuned configs** — [#48309](https://github.com/vllm-project/vllm/pull/48309) 及后续:为上游只剩通用默认值的形状补 Triton MoE 配置,当前聚焦 H20 上 192 路由专家(E=192)的形状。
- 全部 PR:[vLLM 上的 author:LioEinaudi](https://github.com/vllm-project/vllm/pulls?q=author%3ALioEinaudi)

我刻意在 **NVIDIA H20** 和 **RTX 4090D** 上做测量——避开 H100/B200 主流,更贴近生产集群实际在跑的硬件。

---

## 实习经历

- **Dexmal(原力灵机)**,具身智能创业公司 — AI Infra 算法实习生,2026 年 6 月至今。

---

## 教育经历

- **中央民族大学** — 计算机科学与技术,2028 届。

---

## 获奖

- **蓝桥杯全国软件大赛** — C/C++ 大学 A 组 全国三等奖,2025。

---

## 近期在做

在读 CUTLASS/CuTe kernel 源码,边读边在 H20 上复现它们的 benchmark——目标是从"调 Triton kernel"走到"自己写 Hopper 原生 kernel"。

---

## 工作之外

中长跑、钢琴(笔名就来自钢琴家 Einaudi)、《进击的巨人》。

---

## 联系方式

- GitHub:[@LioEinaudi](https://github.com/LioEinaudi)
- 邮箱:[lioeinaudi@qq.com](mailto:lioeinaudi@qq.com)
