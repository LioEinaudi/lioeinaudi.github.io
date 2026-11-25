---
layout: home
title: Jinbin Zhao (赵锦彬)
lang: en
translation_url: /zh/
list_title: Latest Posts
---

**Inference Systems Engineer**  
Minzu University of China · B.Eng. in Computer Science and Technology, expected 2028

I work on LLM and multimodal inference systems — GPU kernels, compiler graph optimization, and deterministic inference. I tune [vLLM](https://github.com/vllm-project/vllm/pulls?q=author%3ALioEinaudi) kernels and graph execution, and verify bitwise consistency across batch sizes.

---

## Selected Contributions

- Tensor-parallel tuning cut Qwen3-8B batch latency from **3.00 s to 1.33 s**.<br>
  I tuned matmul tiles for the per-rank shapes at TP=2/4/8 on Blackwell SM120. [PR #58495](https://github.com/vllm-project/vllm/pull/58495)
- Choosing matmul tiles at graph capture cut batch latency by **54.2%–63.6%** across Qwen3-1.7B, 4B, and 8B.<br>
  Breakable CUDA graphs let the kernel choose tiles using the actual row count (`M`). [PR #57586](https://github.com/vllm-project/vllm/pull/57586)
- In the GLM-4 prefix test, token IDs and logprobs matched at **64/64 positions**, up from **59/64**.<br>
  I traced the five mismatches to re-tokenizing `.Here` and integrated @Evangade’s fix. [PR #53692](https://github.com/vllm-project/vllm/pull/53692) · [Original patch](https://github.com/vllm-project/vllm/pull/43317) · [Writeup]({{ '/2026/11/25/glm-4-token-prefix.html' | relative_url }})

Both latency benchmarks were decode-heavy with batch invariance enabled (BI=1), 32 input tokens, 128 output tokens, and batch size 8. The TP benchmark used two RTX PRO 6000 GPUs (TP=2). The graph benchmark used one.
{: .benchmark-note}

---
