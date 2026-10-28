# Candidates: 212 vLLM fix PRs (2025-01..2026-10), see candidates.csv
| category | laptop | gpu80 | multi-gpu | hopper | blackwell | amd | other | total |
| --- | --: | --: | --: | --: | --: | --: | --: | --: |
| numeric | 25 | 17 | 29 | 5 | 11 | 8 | 1 | 96 |
| gross | 13 | 11 | 10 | 6 | 12 | 11 | 3 | 66 |
| crash | 7 | 17 | 3 | 4 | 3 | 3 | 1 | 38 |
| non-numeric | 4 | 2 | 1 | 0 | 2 | 0 | 0 | 9 |
| unclear | 1 | 0 | 1 | 0 | 0 | 0 | 1 | 3 |
Numeric + greedy-visible: **12 laptop** (9 new), **13 gpu80**. Regression=yes: 47. Wheels: 202 both, 3 buggy-only, 7 none. `?` cells counted at face value.
Next 15 for the 4 GB laptop:
1. **53877** GDN packed decode rounds beta to bf16. Error compounds every decode step. Qwen3.5-0.8B, default flags. No introducing PR named.
2. **31931** Regression from #31676: Triton fused_moe multiplies by router weight after the bf16 cast. granite-3.1-1b-a400m, default. Tight fit in 4 GB.
3. **49586** Online FP8 layerwise reload leaves linear biases uninitialized. Qwen2.5-0.5B (qkv bias)?, `--quantization fp8`.
4. **30915** FlexAttention maps reused sliding-window blocks to the wrong logical block. gemma-3-270m, `FLEX_ATTENTION`, prompt > window.
5. **36580** MRV2 slot mapping is wrong for chunked prefill under CUDA-graph padding. Qwen3-0.6B, MRV2. Needs native Linux.
6. **47574** Hybrid fp8 KV reads stale fp8 pages as NaN. gemma-3-270m?, `kv_cache_dtype=fp8 --kv-cache-dtype-skip-layers sliding_window`, input > window.
7. **48010** Transformers backend loses Gemma's embed scaling under CUDA graphs, giving gibberish. gemma-3-270m, `--model-impl transformers`.
8. **32118** A new Mamba request scheduled as 1-token decode reads stale SSM state. granite-4.0-h-350m?, low memory + high concurrency.
9. **41873** Stale `is_prefilling` in padded CUDA-graph rows misclassifies Mamba decodes. granite-4.0-h-350m, full CG + concurrency.
10. **29216** Regression from #26060: returned logprobs are wrong with spec decode + chunked prefill. Tokens are unchanged, so only the KL detector should see it. Qwen3-0.6B, ngram spec decode.
11. **28972** The LoRA RowParallelLinear wrapper adds bias in bf16 outside the GEMM. Needs a model with o_proj/down bias. LoRA on.
12. **18957** V0 prefix-cache tracker goes stale after a failed schedule. Qwen2.5-0.5B, V0, prefix caching + preemption.
13. **28783** fp8 KV scales are not restored after sleep(level=2), giving gibberish. Qwen3-0.6B, `kv_cache_dtype=fp8`, sleep/wake.
14. **57163** sleep(level=2) zeroes compressed-tensors KV scales, giving NaN. TinyLlama-1.1B kvcache-fp8 checkpoint, sleep/wake + reload.
15. **62046** Regression from #54651: the split top-p mask resets its tie count every 8192 entries. Qwen3-0.6B, top-p sampling. Sampling only, not greedy.
