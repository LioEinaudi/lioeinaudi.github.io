---
layout: home
permalink: /zh/
lang: zh
translation_url: /
list_title: 最新文章
---

# 赵锦彬

**推理系统工程师**  
中央民族大学 · 计算机科学与技术本科，预计 2028 年毕业

我做 LLM 与多模态推理系统——GPU kernel、AI 编译器图优化与确定性推理。我在 [vLLM](https://github.com/vllm-project/vllm/pulls?q=author%3ALioEinaudi) 上调优算子与图执行，并验证跨 batch 的逐位一致性。

---

## 精选贡献

- **TP 分片矩阵乘调优**：在双卡 RTX PRO 6000（Blackwell SM120）上把 Qwen3-8B 的 batch-invariant 整批生成延迟从 **3.00s 降到 1.33s** 并通过跨 batch 位级一致校验，为 SM120 补充 TP=2/4/8 分片形状的 Triton 按 M 分桶配置（[PR #58495](https://github.com/vllm-project/vllm/pull/58495)）。
- **图捕获时选择矩阵乘配置**：在 RTX PRO 6000 上把 Qwen3-1.7B/4B/8B 的整批延迟降低 **54.2%–63.6%**，相对普通推理的额外延迟从 **128%–195% 压到 4%–7%**——为支持的 batch-invariant 配置启用 breakable CUDA Graph，按捕获时的真实 `M` 选配（[PR #57586](https://github.com/vllm-project/vllm/pull/57586)）。
- **GLM-4 测试误报排查**：在同一引擎上把确定性测试从 59/64 修到 **64/64**，并让 GLM-4-9B-Chat 在 RTX PRO 6000 上跑通 `test_batch_invariance.py` 全部 **19** 项用例——定位到 `.Here` 重分词导致前缀 token ID 错位，保留原作者署名集成社区 @Evangade 的 token ID 前缀补丁（[PR #53692](https://github.com/vllm-project/vllm/pull/53692)、[原始补丁 #43317](https://github.com/vllm-project/vllm/pull/43317)、[排查笔记](/zh/2026/11/25/glm-4-token-prefix.html)）。

---
