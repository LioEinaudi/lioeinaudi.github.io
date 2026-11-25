---
layout: home
title: 赵锦彬
permalink: /zh/
lang: zh
translation_url: /
list_title: 最新文章
---

**推理系统工程师**  
中央民族大学 · 计算机科学与技术本科，预计 2028 年毕业

我做 LLM 与多模态推理系统——GPU kernel、AI 编译器图优化与确定性推理。我在 [vLLM](https://github.com/vllm-project/vllm/pulls?q=author%3ALioEinaudi) 上调优算子与图执行，并验证跨 batch 的逐位一致性。

---

## 精选贡献

- 张量并行调优让 Qwen3-8B 的整批生成延迟从 **3.00 秒降到 1.33 秒**。<br>
  我针对 Blackwell SM120 在 TP=2/4/8 下每张卡实际处理的矩阵形状，调优了矩阵乘的分块配置。[PR #58495](https://github.com/vllm-project/vllm/pull/58495)
- 把矩阵乘的分块选择移到图捕获阶段后，Qwen3-1.7B、4B、8B 的整批延迟降低了 **54.2%–63.6%**。<br>
  启用 breakable CUDA Graph 后，算子可以按捕获时矩阵的实际行数（`M`）选择分块。[PR #57586](https://github.com/vllm-project/vllm/pull/57586)
- 在 GLM-4 的前缀测试中，token ID 和 logprob 均匹配的位置从 **59/64 增至 64/64**。<br>
  我把五处差异追溯到 `.Here` 的重新分词，并集成了 @Evangade 的修复。[PR #53692](https://github.com/vllm-project/vllm/pull/53692) · [原始补丁](https://github.com/vllm-project/vllm/pull/43317) · [排查笔记]({{ '/zh/2026/11/25/glm-4-token-prefix.html' | relative_url }})

前两项延迟数据均来自以解码为主的基准，开启批次不变模式（BI=1），输入 32 个 token，输出 128 个 token，batch size 为 8。TP 调优使用两张 RTX PRO 6000（TP=2），图执行优化使用单张同型号 GPU。
{: .benchmark-note}

---
