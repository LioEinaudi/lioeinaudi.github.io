---
layout: page
title: About
permalink: /about/
lang: en
translation_url: /zh/about/
---

<img class="about-avatar" src="{{ '/assets/avatar.jpg' | relative_url }}" alt="Lio Einaudi" />

I'm **Jinbin Zhao (赵锦彬)**, online as **Lio Einaudi**. I work on LLM and multimodal inference systems — GPU kernels, compiler graph optimization, and deterministic inference. Most of my public work lives in [vLLM](https://github.com/vllm-project/vllm/pulls?q=author%3ALioEinaudi).

---

## Experience

- **NIO (蔚来)** — Incoming Inference Systems Research Intern.
- **Dexmal (原力灵机)**, embodied-AI startup — AI Infra Algorithm Intern, June 10 – November 23, 2026.

---

## Open Source

**[vLLM](https://github.com/vllm-project/vllm/pulls?q=author%3ALioEinaudi)** — batch-invariant deterministic inference (6 merged PRs):

- **End-to-end graph execution** ([#57586](https://github.com/vllm-project/vllm/pull/57586)): cut Qwen3-1.7B/4B/8B batch-invariant generation latency by **54.2%–63.6%** on RTX PRO 6000 (lowering overhead vs. normal inference from **128%–195% to 4%–7%**) — enabled breakable CUDA graphs for supported batch-invariant configurations so matmul tiles are selected using the actual `M` at capture time.
- **Kernel tuning & multi-GPU shapes** ([#53247](https://github.com/vllm-project/vllm/pull/53247), [#57456](https://github.com/vllm-project/vllm/pull/57456), [#58495](https://github.com/vllm-project/vllm/pull/58495)): cut Qwen3-1.7B batch-8 E2E latency by **29.4%** on H20 (+3.4% throughput) and **23.0%** on RTX 4090D (+12.0% throughput), and cut 2× RTX PRO 6000 Qwen3-8B batch latency from **3.00s to 1.33s** with bitwise consistency intact — tuned M-bucketed Triton persistent matmul configs on H20 (Hopper), RTX 4090D (Ada), and RTX PRO 6000 (Blackwell SM120), plus TP=2/4/8 sharded shapes for SM120 ([#53247 writeup](/2026/08/30/cutting-the-determinism-tax.html)).
- **Correctness & test fixes** ([#56377](https://github.com/vllm-project/vllm/pull/56377), [#53692](https://github.com/vllm-project/vllm/pull/53692)): eliminated **1522/1536 → 0** TP=4 logprob mismatches on RTX PRO 6000 and brought GLM-4-9B-Chat from 59/64 to **64/64** on the A/B prefix check (all **19** `test_batch_invariance.py` cases passing on RTX PRO 6000) — disabled sequence parallelism and async TP in batch-invariant mode (verified at TP=4), integrated @Evangade's community token-ID prefix fix ([#43317](https://github.com/vllm-project/vllm/pull/43317)), and added GLM-4/Phi-4 verification and docs ([#53692 writeup](/2026/11/25/glm-4-token-prefix.html)).

---

## Education

- **Minzu University of China (中央民族大学)** — B.Eng. in Computer Science and Technology, expected 2028.

---

## Awards

- **Lanqiao Cup National Software Competition (蓝桥杯)** — National Third Prize, C/C++ University Group A, 2025.

---

## Beyond work

Middle- and long-distance running, piano, and *Attack on Titan*.

---

## Contact

- GitHub: [@LioEinaudi](https://github.com/LioEinaudi)
- Email: [Jinbin.Zhao.Dev@gmail.com](mailto:Jinbin.Zhao.Dev@gmail.com)
