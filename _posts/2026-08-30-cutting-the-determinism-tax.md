---
layout: post
title: "Cutting the Determinism Tax: Making vLLM's Batch-Invariant Mode up to 1.4× Faster"
date: 2026-08-30
---

*How a two-GPU measurement project turned into my first merged vLLM PR ([#53247](https://github.com/vllm-project/vllm/pull/53247)) — and what it taught me about measurement discipline.*

## Why batch invariance

Ask an LLM served by any modern inference engine the same question twice — greedy sampling, temperature 0 — and you can get two different answers. Not because of sampling: because your request was batched with different neighbors each time. Reduction order inside the kernels changes with batch composition, floating-point addition is not associative, and logprobs drift by a few ULPs. Usually nobody cares. But if you are doing RL rollouts and want bit-exact replay, or debugging a training-inference mismatch, this non-determinism is poison.

vLLM ships a batch-invariant mode (`VLLM_BATCH_INVARIANT=1`) that swaps the offending kernels for invariant ones. The catch: those kernels are slow. The community calls the slowdown the **determinism tax**, and when I started measuring it, the tax on my hardware was 30–35% of throughput.

This post is about finding out exactly where that tax is paid and tuning most of the biggest line-item away. The work landed upstream as [PR #53247](https://github.com/vllm-project/vllm/pull/53247); the headline numbers on H20 are **−29% end-to-end latency (1.42×) at batch 8** and **+3–12% throughput**, with the invariance guarantee bit-for-bit intact.

## Setup: two GPUs the big labs don't tune for

I rented two kinds of cloud instances deliberately outside the H100/B200 mainstream:

- **NVIDIA H20** — 96 GB, Hopper (sm90), huge bandwidth but ~1/6 of H100's compute. The dominant inference card in Chinese production fleets.
- **RTX 4090D** — 24 GB, Ada (sm89). Consumer silicon, no TMA, no WGMMA.

Model: Qwen3-1.7B. Two harnesses:

1. **A tax benchmark**: serving throughput and latency with `VLLM_BATCH_INVARIANT` on/off, compiled and eager.
2. **An invariance probe**: run a "needle" prompt alone, then mixed into a batch of 32 (greedy, prefix cache off), and compare logprobs **bit-for-bit in fp32**. A tuned kernel only counts if the probe stays green.

The probe matters as much as the benchmark. Every performance claim below passed it.

## First lesson before any tuning: check your version

My first run showed something alarming: batch-invariant mode *failing its own guarantee* in eager mode on both GPUs — prefill logprobs off by ~0.02. A real upstream bug? Almost: it was a real bug that had **already been fixed**. The root cause (a C++ RMSNorm kernel whose reduction block size changes with token count, which batch-invariant mode couldn't intercept) was fixed upstream weeks earlier — but I had benchmarked a stale release that predated the fix. On the current release, eager went green.

Cheap lesson, permanently installed: **benchmarks run on the latest release or they don't count.** Version capture went into the checklist next to GPU model and driver.

Re-measured on the current release, the true tax (compiled, throughput): **29.9% on 4090D, 35.0% on H20**. Latency tax was worse: +73% and +110%.

## Finding where the tax is paid

The batch-invariant module has per-architecture code paths that *look* load-bearing — a cuBLAS workspace configuration for Hopper, aten overrides for sm8x. I profiled instead of trusting the code.

The profile said something simpler: with invariance on, every cuBLASLt GEMM disappears, replaced by a single Triton kernel — `matmul_kernel_persistent` — because vLLM's linear layer routes **all architectures** straight into it. In one H20 profile it was called 14,577 times, accounted for **79.4% of batch-invariant kernel time**, and explained ~97% of the total kernel-time increase. The per-arch branches I'd been reading were dead code for this path. (A workspace-size ablation confirmed it: varying it changed end-to-end time by ≤0.23%.)

One kernel, hardcoded launch configs, all architectures, all shapes. That's not a tax, that's a tuning opportunity.

## The constraint that makes tuning legal

You can't just autotune an invariant kernel — tuning is exactly how you break invariance. The way out is to look at *why* the kernel is invariant: each output row's value depends only on the **K-loop reduction order**. So:

> **`BLOCK_K` must stay constant across all M for a fixed weight shape (N, K).**
> `BLOCK_M`, `BLOCK_N`, `num_warps`, `num_stages` are free to vary per M-bucket — they change scheduling, not reduction order.

That single invariant separates the tunable parameters from the untouchable one, and it's the technical core of the PR. Every candidate config still had to pass the bitwise probe — trust the argument, verify the bits.

A pleasant side-find from the correctness harness: at K=2048 on the 4090D, the batch-invariant Triton kernel *passed* an fp32 reference check that `torch.mm`'s default path failed — PyTorch's bf16 reduced-precision reduction accumulates enough error to lose to the "slow deterministic" kernel on accuracy. Deterministic and *more* accurate.

## The sweep

A 19,200-point sweep across both GPUs — M-buckets × `BLOCK_M`/`BLOCK_N`/`num_warps`/`num_stages`, `BLOCK_K` pinned per shape, two correctness gates per config. Results:

| | decode GEMMs (vs torch) | prefill | E2E throughput tax |
|---|---|---|---|
| RTX 4090D | 0.25× → **0.77×** (3.0×) | 1.16× | 29.9% → **21.9%** |
| H20 | 0.21× → **0.59×** (2.8×) | 1.11× | 35.0% → **31.0%** |

Latency tax dropped harder: 73→36% (4090D), 110→59% (H20).

Two measurement stories from this phase that I'd repeat on any project:

- **The eager "regression" that wasn't.** Tuned eager throughput initially looked 44% *worse* on 4090D. Cause: M-bucketing multiplies Triton JIT compilations (3 → 43 cache entries), and a short benchmark eats the cold-start. Steady-state after warmup: +1.2%. Compiled mode prepays this in graph capture, which is why it never showed there.
- **The phantom H20 number.** One H20 eager baseline was ~40% slower than every other run. Before explaining it, I checked node identity — GPU UUID and serial numbers showed the before/after had landed on *different physical machines*. Re-ran both sides on one node, with the serials logged: the anomaly vanished, and one number I had already posted publicly needed a correction. **Before/after on provably the same node, or it didn't happen.**

## Two ambushes from torch.compile

The config-lookup code sits on a hot path that `torch.compile` traces. Two "clean" implementations died there:

1. `functools.lru_cache` around device-name lookup → Dynamo traces *through* the cache into pynvml's ctypes calls → `Unsupported` → **the compiled engine doesn't start.** Kernel tests and eager pytest were all green; only a real compiled end-to-end run caught it. Shipping it would have been a P0.
2. `bisect_left(..., key=...)` → also `Unsupported` in the current torch.

Final shape: resolve the device to a config table **once at init**, store it in a module-level variable, and look up buckets with a linear scan over a sorted tuple of ≤10 entries. Boring, trace-proof, and just as fast.

New rule in my acceptance checklist: **if code can be traced by Dynamo, the test plan includes a real compiled end-to-end run.** No proxy suffices.

## The review arc

I opened the PR keyed to exact device names, with every one of the 100 config entries mechanically cross-checked against the sweep JSON. First maintainer response, same day:

> "don't want to tune triton for older architecture, not used frequently in production"

Fair concern, wrong premise — the H20 is not an older architecture; it's current-generation Hopper, and it *is* production hardware for a large fraction of real deployments. I replied with that one factual clarification (no argument beyond it) plus an offer to narrow scope. The maintainer's answer: "Ohh make sense for hopper — could you test e2e using vllm bench? Let's see if it's worth the complexity."

So the decision criterion became end-to-end numbers in the exact format of the maintainer's own prior perf PR — command lines and raw before/after output blocks, same-node discipline, upstream's full determinism test suite passing:

- **H20**: latency −29.4% at batch 8 (1.42×), −19.0% at batch 32; throughput +3.4%
- **RTX 4090D**: latency −23.0% at batch 8, −16.5% at batch 32; throughput +12.0%

The review flipped:

> "Wow, very good perf improvement I didn't think about before, nice work"

Three review rounds later (configs moved to their own file, keys widened from exact device names to architecture families — `hopper` backed by H20 measurements, `ada` by 4090D — as the maintainer requested), the PR merged. **Seven days from first measurement to merge.** The maintainer plans follow-up tuning and refactoring on top, and the architecture-family keys leave labeled slots for Blackwell when someone with the hardware shows up.

One more moment worth recording: mid-review, another contributor offered to add H100 configs to the PR — derived numbers, not measured ones. I asked them to hold for a follow-up PR after real measurements instead. The entire credibility of a tuning PR is that *every number has a provenance*; one estimated row poisons the table.

## What generalized

The kernel work is the resume line, but the transferable part is the discipline:

1. **Latest release or it doesn't count.** My scariest "bug" was a stale version.
2. **Profile before believing code structure.** The per-arch branches were dead; one kernel was 79% of the story.
3. **Find the invariant that makes optimization legal.** `BLOCK_K` constancy turned "you can't tune a deterministic kernel" into a 19,200-point sweep.
4. **Same node, logged serials, or re-run.** One cross-node number cost me a public correction.
5. **Compiled E2E is a non-negotiable gate** for anything Dynamo might trace.
6. **In review, one factual clarification beats a debate** — and the reviewer's own preferred evidence format is the fastest path to yes.
7. **Every number needs provenance** — in your tables and everyone else's.

---

*The tax isn't zero yet: decode GEMMs still trail cuBLAS, and the remaining gap lives in shapes and architectures nobody has swept. If you have the hardware, the config table has labeled empty slots.*
