---
layout: page
title: About
permalink: /about/
lang: en
translation_url: /zh/about/
---

<img class="about-avatar" src="https://github.com/LioEinaudi.png" alt="Lio Einaudi" />

I'm **Jinbin Zhao (赵锦彬)**, online as **Lio Einaudi**. I work on LLM inference performance — GPU kernels, MoE tuning, deterministic (batch-invariant) inference — with one operating rule: benchmarks run on the latest release, gated by a correctness probe, or they don't count.

## Work

- **Batch-invariant GEMM tuning** — [vLLM #53247](https://github.com/vllm-project/vllm/pull/53247), merged: tuned the Triton GEMM that ate 79% of batch-invariant kernel time, within its invariance constraint (`BLOCK_K` pinned). **−29% end-to-end latency (1.42×) on H20**, determinism bit-for-bit intact. Full story: [Cutting the Determinism Tax]({% post_url 2026-08-30-cutting-the-determinism-tax %}).
- **Fused-MoE tuned configs** — [#48309](https://github.com/vllm-project/vllm/pull/48309) and follow-ups: Triton MoE configs for shapes upstream leaves on generic defaults. Current focus: 192-routed-expert shapes on H20.
- All PRs: [author:LioEinaudi on vLLM](https://github.com/vllm-project/vllm/pulls?q=author%3ALioEinaudi)

I measure on **NVIDIA H20** and **RTX 4090D** — deliberately outside the H100/B200 mainstream, and much closer to what production fleets actually run.

## Experience

- **Dexmal (原力灵机)**, embodied-AI startup — AI Infra intern, June 2026 – present.

## Education

- **Minzu University of China (中央民族大学)** — B.Eng. in Computer Science and Technology, expected 2028.

## Awards

- **Lanqiao Cup National Software Competition (蓝桥杯)** — National Third Prize, C/C++ University Group A, 2025.

## Now

Reading CUTLASS/CuTe kernels and reproducing their benchmarks on H20 — moving from tuning Triton kernels to writing Hopper-native ones.

## Beyond work

Middle- and long-distance running, piano (yes, that's where the pen name comes from), and *Attack on Titan*.

## Contact

- GitHub: [@LioEinaudi](https://github.com/LioEinaudi)
- Email: [lioeinaudi@qq.com](mailto:lioeinaudi@qq.com)
