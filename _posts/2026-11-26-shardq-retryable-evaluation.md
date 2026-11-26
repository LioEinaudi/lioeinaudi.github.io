---
layout: post
title: "How ShardQ Runs One Evaluation Across Dozens of Preemptible GPUs"
date: 2026-11-26
lang: en
permalink: /2026/11/26/shardq-retryable-evaluation.html
translation_url: /zh/2026/11/26/shardq-retryable-evaluation.html
---

*ShardQ is a scheduler I built for multimodal evaluations: split a dataset into short ranges, let idle GPUs claim them as they free up, and if a card gets pulled mid-run, let another worker rerun just that shard.*

I used ShardQ to evaluate Qwen3.5-4B across DynaMath, Video-MME, MLVU, and LVBench on a shared preemptible GPU queue, with 58 to 74 single-GPU workers running concurrently. Dividing the dataset evenly across GPUs up front breaks quickly in that setting: the worker count keeps changing, and per-question latency varies enough that one card finishes its slice while another is still decoding a long answer.

ShardQ flips the assignment: slice sample indices into short ranges in a queue, and let whichever GPU frees up first claim the next one. Tracing a single range through the 5,010-question DynaMath run shows how the pieces fit together.

## Slicing 5,010 questions into shards

When registering a job, I specify the model, evaluation tasks, protocol, and resource pool. The pool selects the backend and launcher. In this run, `lmms-eval` handled the evaluation and the launcher requested workers from a cloud idle-GPU queue. ShardQ records the `lmms-eval` Git revision and whether its working tree is dirty, and checks where the evaluation interpreter imports `lmms_eval` from — so comparing two runs starts with verifying their revisions and import paths.

Within each benchmark partition, a shard is just a starting offset and a sample count. For DynaMath, I set the shard size to 8 questions: 0–7, 8–15, down to a 2-question tail shard. **Eight is the unit of claiming, validation, and retry** — separate from inference batch size, which was 1.

Shard size is mostly a choice of how much recomputation you are willing to absorb on a preemption. In this setup, 8 DynaMath questions took roughly half an hour; for the 2,700 Video-MME evaluation samples, I used 1 sample per shard. If you prefer not to hardcode a size, you can pass a target duration with min/max sample bounds so ShardQ estimates the next shard size from the average per-sample runtime of completed shards. The scheduler only stages a small batch of pending shards ahead of demand, so later cuts pick up the updated estimate.

## Claiming a shard and keeping weights resident

Take shard 8–15 sitting ready in the queue. The scheduler and every worker share a filesystem where each shard has its own directory. A worker claims the shard with an atomic `mkdir` on `lease/`: if two workers race for it, one wins and the other looks for another range.

Once claimed, the worker runs `lmms-eval` starting at offset 8 for 8 samples and keeps the per-sample results. ShardQ checks that the output covers IDs 8 through 15 before writing the completion marker for that attempt. Afterward, as long as the next shard has a compatible session configuration, the worker keeps the model resident in GPU memory and claims the next range — so even with Video-MME at 1 sample per shard, it does not reload weights for every sample.

Two roles stay cleanly separated in the code. The **launcher** requests workers from a local or cloud queue and checks whether they are alive; the **backend** loads the model, runs a sample range, validates output, and computes metrics. The scheduler stays out of model details, reading pending work from the shared ledger and pool limits to decide when to cut more shards or request more workers.

## Preemption, recomputation, and bisection

Stick with shard 8–15. Suppose the worker finishes computing predictions for IDs 8 through 12 when its GPU is reclaimed. The `lmms-eval` backend writes per-sample results only after evaluation returns for the whole shard, so this attempt has written neither `records.jsonl` nor a completion marker. If the worker receives a termination signal, it records the preemption itself; if it vanishes abruptly, the scheduler uses the lease heartbeat and the launcher's liveness report to reclaim the shard. The replacement worker restarts from sample 8, while previously completed shards keep their results.

After a restart, the scheduler scans shard directories, leases, and attempt records on the shared filesystem to reconstruct what still needs to run. The replacement worker has to repeat the 5 questions its predecessor already finished. Shorter shards trim wasted work at the expense of more claims, validations, and ledger writes.

Swapping GPUs won't help if an input crashes every time, or if a shard is simply too slow to finish between preemptions. ShardQ counts execution errors and preemptions separately; once either hits its limit, the scheduler splits the shard in two if bisection is enabled, the shard still has multiple samples, and it is below the depth limit — otherwise it quarantines the shard.

In the repository's CPU example, sample 7 raises an exception on every attempt. Bisection isolates it and recovers the other 7 samples in shard 0–7. Samples 8 through 23 finished normally, bringing coverage to 23 of 24 across the partition. The report records `missing_units: [7]`.

## Merging results without missing a question

All workers exiting only means nothing is running; it does not prove DynaMath is done. The real check is simpler: does every question from 0 through 5,009 have a usable scoring record?

During merge, ShardQ selects one successful attempt per completed shard, verifies the record file's checksum, deduplicates rows by sample ID and evaluation filter, and checks index coverage. The `lmms-eval` backend runs the task's own aggregation rule over those per-sample records, so the 2-question tail shard contributes exactly 2 questions' worth of data.

A fully covered partition gets marked `complete`. When partial merges are allowed, missing IDs are written alongside the `partial` result; for formal benchmark runs, `--require-complete` enforces full coverage. Even if a different set of GPUs finishes the run, the score still traces back to the exact questions included.
