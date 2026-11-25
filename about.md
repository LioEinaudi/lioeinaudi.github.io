---
layout: page
title: About
permalink: /about/
lang: en
translation_url: /zh/about/
---

<img class="about-avatar" src="https://github.com/LioEinaudi.png" alt="Lio Einaudi" />

I'm **Jinbin Zhao (赵锦彬)**, online as **Lio Einaudi**. I work on LLM and multimodal inference systems — GPU kernels, compiler graph optimization, and deterministic inference. Most of my public work lives in [vLLM](https://github.com/vllm-project/vllm/pulls?q=author%3ALioEinaudi).

---

## Experience

- **NIO (蔚来)** — Incoming Inference Systems Research Intern.
- **Dexmal (原力灵机)**, embodied-AI startup — AI Infra Algorithm Intern, June 10 – November 23, 2026.

---

## Open Source

**[vLLM](https://github.com/vllm-project/vllm/pulls?q=author%3ALioEinaudi)** — batch-invariant deterministic inference (6 merged PRs):

- **End-to-end graph execution** ([#57586](https://github.com/vllm-project/vllm/pull/57586)): enabled breakable CUDA graphs for supported batch-invariant configurations so matmul tiles are selected using the actual `M` at capture time.
- **Kernel tuning & multi-GPU shapes** ([#53247](https://github.com/vllm-project/vllm/pull/53247), [#57456](https://github.com/vllm-project/vllm/pull/57456), [#58495](https://github.com/vllm-project/vllm/pull/58495)): M-bucketed Triton persistent matmul configs tuned on H20 (Hopper), RTX 4090D (Ada), and RTX PRO 6000 (Blackwell SM120), plus TP=2/4/8 sharded shapes for SM120 under bitwise batch-invariance constraints.
- **Correctness & test fixes** ([#56377](https://github.com/vllm-project/vllm/pull/56377), [#53692](https://github.com/vllm-project/vllm/pull/53692)): disabled sequence parallelism and async TP in batch-invariant mode (verified at TP=4), integrated the community token-ID prefix fix for determinism tests, and verified GLM-4/Phi-4.

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
