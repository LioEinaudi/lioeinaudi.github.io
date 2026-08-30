---
layout: page
title: About
permalink: /about/
---

I'm **Lio Einaudi**. I work on LLM inference performance: GPU kernels, MoE tuning, and deterministic (batch-invariant) inference. Most of my public work lives in [vLLM](https://github.com/vllm-project/vllm/pulls?q=author%3ALioEinaudi).

If there is one thread through everything below, it's **measurement discipline**: benchmarks run on the latest release or they don't count, every performance claim ships with a correctness gate, and a profile beats reading the code every time.

## Selected work

**Cutting the determinism tax.** vLLM's batch-invariant mode buys reproducibility at a price — on my hardware, 30–35% of throughput. I profiled where the tax was actually paid (79% of batch-invariant kernel time in a single Triton GEMM with hardcoded launch configs), worked out which tuning parameters are legal to touch without breaking the invariance guarantee (`BLOCK_K` is untouchable; scheduling parameters are free), and ran a 19,200-point sweep with a bitwise correctness probe gating every candidate. Merged upstream as [vLLM #53247](https://github.com/vllm-project/vllm/pull/53247): **−29% end-to-end latency (1.42×) at batch 8 on H20, +3–12% throughput**, invariance bit-for-bit intact. Full write-up: [Cutting the Determinism Tax]({% post_url 2026-08-30-cutting-the-determinism-tax %}).

**Fused-MoE tuned configs.** Upstream Triton MoE kernels fall back to generic launch configs for shapes nobody tuned — which in practice means most shapes outside the flagship models. I run `benchmark_moe` sweeps and validate candidates A/B through `VLLM_TUNED_CONFIG_FOLDER` before submitting ([#48309](https://github.com/vllm-project/vllm/pull/48309) and follow-ups); current focus is large routed-expert counts (E=192) on H20, where upstream has zero tuned configs today.

## How I measure

I deliberately benchmark on GPUs outside the H100/B200 mainstream:

- **NVIDIA H20** — Hopper (sm90), 96 GB, huge bandwidth, ~1/6 of H100 compute. The dominant inference card in Chinese production fleets, and a very different tuning target from the cards most defaults are written for.
- **RTX 4090D** — Ada (sm89), consumer silicon. No TMA, no WGMMA — what a lot of real deployments actually run on.

Every result comes from a paired setup: a benchmark harness for the numbers, and a correctness/invariance probe that has to stay green for the numbers to count. Version capture (release, driver, GPU) goes in the checklist — I learned that one the hard way, by "discovering" an upstream bug that had already been fixed in a release newer than the one I was benchmarking.

## Now

Reading CUTLASS/CuTe kernels the way I'd read a textbook — RoPE/norm ops first, then split-K attention decode, then TMA-pipelined fused MoE — and reproducing their benchmarks on H20 as I go. The goal: move from tuning Triton kernels to writing Hopper-native ones.

<!-- ## Education -->
<!-- TODO: 是否公开学校/年级由你决定。这个站点目前是笔名身份,写了学校就基本实名了。 -->
<!-- - B.S. in XXX, XXX University (expected 2028) -->

## Contact

- GitHub: [@LioEinaudi](https://github.com/LioEinaudi)
<!-- - Email: 建议注册一个 lio 身份的邮箱再放上来,别直接挂个人 gmail -->

---

*Lio Einaudi is a pen name — yes, like the composer.*
