---
layout: home
lang: en
translation_url: /zh/
list_title: Latest Posts
---

# Jinbin Zhao (赵锦彬)

**Inference Systems Engineer**  
Minzu University of China · B.Eng. in Computer Science and Technology, expected 2028

I work on LLM and multimodal inference systems — GPU kernels, compiler graph optimization, and deterministic inference. I tune [vLLM](https://github.com/vllm-project/vllm/pulls?q=author%3ALioEinaudi) kernels and graph execution, and verify bitwise consistency across batch sizes.

---

## Selected Contributions

- **TP sharded matmul tuning**: cut Qwen3-8B batch-invariant generation latency from **3.00s to 1.33s** on 2× RTX PRO 6000 (Blackwell SM120) with bitwise consistency intact, adding M-bucketed Triton configs for SM120 TP=2/4/8 shapes ([PR #58495](https://github.com/vllm-project/vllm/pull/58495)).
- **Capture-time matmul config selection**: cut Qwen3-1.7B/4B/8B batch-invariant latency by **54.2%–63.6%** on RTX PRO 6000 — lowering determinism overhead from **128%–195% to 4%–7%** — by enabling breakable CUDA graphs so matmul tiles are selected using the actual `M` at capture time ([PR #57586](https://github.com/vllm-project/vllm/pull/57586)).
- **GLM-4 false-failure investigation**: turned a 59/64 determinism failure into **64/64** on the same engine and got all **19** `test_batch_invariance.py` cases passing for GLM-4-9B-Chat on RTX PRO 6000, by tracing `.Here` re-tokenization and integrating @Evangade's community token-ID prefix fix ([PR #53692](https://github.com/vllm-project/vllm/pull/53692), [original patch #43317](https://github.com/vllm-project/vllm/pull/43317), [writeup](/2026/11/25/glm-4-token-prefix.html)).

---
