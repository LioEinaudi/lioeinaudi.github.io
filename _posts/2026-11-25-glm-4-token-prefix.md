---
layout: post
title: "How .Here Caused a False Failure in GLM-4's Batch-Invariance Test"
date: 2026-11-25
lang: en
translation_url: /zh/2026/11/25/glm-4-token-prefix.html
---

*How five characters (`.Here`) failed GLM-4's batch-invariance test on two GPUs — and how passing token IDs instead of re-encoded strings took the same engine from 59/64 to 64/64 ([PR #53692](https://github.com/vllm-project/vllm/pull/53692)).*

I originally set out to add GLM-4 to vLLM's list of verified batch-invariant models, ran the official test suite, and ended up pulling it back out of the PR. Comparing decode against prefill under `FLASH_ATTN`, both an RTX 4090D and an H20 failed on the exact same prompt: positions 0 through 10 passed, position 11 picked a different token, and positions 12 through 15 picked the same tokens with logprobs drifting by up to ~0.16. In the [two-GPU test log](https://github.com/vllm-project/vllm/pull/53692#issuecomment-5472201949), I wrote "GLM has a real failure."

Digging deeper proved me wrong. The culprit was five characters in the output: `.Here`.

## What position 11 is actually checking

The test runs greedy generation on GLM-4-9B-Chat first, recording the chosen token ID and its logprob at every step. Then, for each position, it sends a fresh single-token request (`max_tokens=1`) with the preceding prefix as the prompt. It checks the predicted token ID first, and only if the IDs match does it check exact FP32 logprob equality (see the [merged test source](https://github.com/vllm-project/vllm/blob/89439db7276cf165377c088496e20098c8137727/tests/v1/determinism/test_batch_invariance.py#L779-L867)).

To verify position 11 (0-indexed), you concatenate the original prompt with the first 11 generated tokens, feed that prefix through prefill, and ask the model to predict the 12th token. For the comparison to mean anything, both paths have to see the exact same sequence of token IDs.

The old test took a detour here. To reconstruct the text before position 11, it ran generation again for 11 steps from the original prompt, appended the decoded string back onto the prompt string, and handed the combined text to vLLM. That pushed the entire prefix through the tokenizer a second time.

## One `.Here`, two tokenizations

The failing response happened to contain the substring `…tree.Here`. Isolating the tokens around the period shows what actually changed:

| Prefix source | Text | Token IDs |
| --- | --- | --- |
| Original tokens emitted during decode | `.Here` | `[13, 8419]` |
| Decoded text appended to prompt and re-encoded | `.Here` | `[89329]` |

During decode, GLM emitted `.` (`13`) on one step and `Here` (`8419`) on the next. Once `.` was sampled, it already occupied a slot in the sequence. On re-encoding, however, the BPE tokenizer saw the whole string at once and merged the period and the next four letters into a single vocabulary token (`89329`). Printed as strings, both prefixes read `.Here`, but the underlying tokenization had already shifted.

Position 11 was the first step whose reconstructed prefix contained all of `.Here`. At that point, decode was predicting after `[…, 13, 8419]`, while prefill was predicting after `[…, 89329]`. Even with perfectly deterministic math, feeding two different token histories into the same model can change the next token — and on later positions that happened to pick the same token anyway, the logprobs still reflected two different prefixes.

## Changing only the prefix across 64 positions

To verify that this merge explained all five failures, I ran a controlled A/B test on the same RTX 4090D and the same vLLM engine instance: compiled mode, `FLASH_ATTN`, `VLLM_BATCH_INVARIANT=1`, 4 fixed prompts, 16 generated positions each — changing nothing except how the prefill prefix was built (see the [per-position A/B log](https://github.com/vllm-project/vllm/pull/53692#issuecomment-5486601496)).

| Prefill input | Positions with matching token and exact FP32 logprob |
| --- | --- |
| Generated text appended to prompt and re-encoded | 59 / 64 |
| Original prompt token IDs + actual generated token ID prefix | **64 / 64** |

All five failures in the first run landed on prompt 0, positions 11–15 — the exact five steps where re-encoding altered the prefix token IDs. Preserving the original token boundaries wiped out all five mismatches while keeping the other 59 positions green.

The fix is to read both the prompt IDs and the generated IDs straight off the initial decode output. For position `i`, slicing `generated_ids[:i]` stops right before the token under test:

```python
prompt_ids = list(decode_output.prompt_token_ids or [])
generated_ids = decode_output.outputs[0].token_ids
prefix_ids = prompt_ids + list(generated_ids[:i])
prefill_prompt = TokensPrompt(prompt_token_ids=prefix_ids)
```

Checking the issue history, @xRay2016 had run into the same class of bug on Qwen's tokenizer — merging `.` and `#` (see the [original repro comment](https://github.com/vllm-project/vllm/issues/27433#issuecomment-4358595058)) — and @Evangade had already submitted a [patch](https://github.com/vllm-project/vllm/pull/43317) to build prefixes from token IDs. Since GLM gave a clean concrete reproducer, I verified the fix and, per the maintainer's suggestion, carried the patch into [PR #53692](https://github.com/vllm-project/vllm/pull/53692) with the original author's attribution intact.

Running GLM-4-9B-Chat on an RTX PRO 6000 with the updated test passed all 19 cases in `test_batch_invariance.py`, putting GLM back on the verified model list.

Next time two inference paths disagree on logprobs, the first thing I'll dump is the prefix token IDs at the first failing position. Looking at strings alone, both logs print `…tree.Here` and give nothing away.
