# fimcoder-113m — a from-scratch FIM code model on $30 of expiring Modal credit

**Status 2026-08-31 10:30 UTC:** pretraining in flight (step ~530 / 2695), $8.96 of $30
spent, every downstream path proven against a test checkpoint. Expected finish ~14:20 UTC.

The deliverable is a fill-in-the-middle code model trained from scratch on **2024+ code
only**, served behind an OpenAI-compatible endpoint and wired into **minuet-ai.nvim**, plus a
quantised GGUF that keeps working locally after the credit dies.

---

## What the hardware forced

The workspace has no payment method on file, so Modal refuses H100, H200, B200, L40S and
both A100s client-side:

```
InvalidError('Please add a payment method to use H100 GPU functions.')
```

Available: T4, L4, A10, and multi-GPU `L4:2` / `L4:4`. Value per dollar, using **dense**
bf16 (vendors quote L4/L40S/A10 with 2:1 sparsity — halve it):

| GPU | dense bf16 | $/hr | TFLOP/$ | verdict |
|---|---|---|---|---|
| **L4** | 60.6 | 0.799 | **76** | chosen |
| T4 | 65 (fp16) | 0.590 | 110 | sm_75: no bf16, and SDPA's flash backend needs sm_80+ |
| A10 | 62.5 | 1.102 | 57 | strictly worse than L4 |
| H100 | 989 | 3.949 | 250 | gated |

4× L4 costs exactly 4× one L4, so multi-GPU buys wall-clock for free. Gradient sync is a
single flat all-reduce per optimizer step (<2% of step time at 0.5M tokens/step) rather than
DDP — no autograd hooks, no torch.compile interaction, and `LlamaForCausalLM.forward` never
materialises full logits.

## Model

Compute-optimal for what 4× L4 buys in 4.6h, **counting attention FLOPs** (Chinchilla's
`C = 6ND` ignores them, and they are 33% of the total at seq 2048 for this shape):

```
LlamaForCausalLM   d_model 768 · 14 layers · 12 heads / 4 KV (GQA) · d_ff 2048
                   RMSNorm · SwiGLU · RoPE θ=50k · tied embeddings
vocab              32768  (ours, byte-level BPE, 7 FIM sentinels)
params             88.1M non-embedding + 25.2M tied = 113.3M
FLOPs/token        0.529 (6N) + 0.264 (attention @2048) = 0.793 GFLOP
tokens             ~1.41B   (D/N ≈ 16)
```

Stock `LlamaForCausalLM`, not hand-rolled: a silent bug in a hand-rolled transformer costs
the whole budget and buys nothing, and the standard class means vLLM and llama.cpp load the
result with zero conversion work — which is the entire point of a model that lives in an editor.

**Sequence length 4096.** The first 75% of steps run at 2048; the last 25% — which is also
the WSD decay leg — run at 4096, so the shipped model genuinely handles a 4096-token editor
window with several files of repo context. Attention is 33% of FLOPs/token at 2048 and 50%
at 4096, so paying for the long window only while annealing buys the capability at a fraction
of training at 4096 throughout.

**WSD, not cosine.** Warmup → stable → 1-√ decay. Cosine locks the schedule to a step count
guessed before throughput is known; WSD fixes the horizon from a *measurement* at step 24 and
still ends properly annealed.

## Data — 198 languages, fresh only

`HuggingFaceCode/stack-v3-train`, 8192 parquet parts, one row per repo with contents inline.

```
seq 2048   1,726,351 records = 3.54B tokens   fill 99.9%   4.00M docs
seq 4096     303,265 records = 1.24B tokens   fill 100.0%  0.93M docs (repo-heavy)
val           8,188 records =  16.8M tokens   168 languages
fimeval       2,853 examples, 166 languages, half single-file / half multi-file
```

Language mix is **5.6% total-variation distance** from target (Python 15.6 vs 14.6, TypeScript
9.4 vs 8.9, JavaScript 8.8 vs 8.3, Go 3.5 vs 3.2, Rust 2.4 vs 2.8, Swift 1.4 vs 1.4). Getting
there took three builds; see CONTEXT.md §"what went wrong" for why the first two were 23.9%
and 14.0% off.

Document shapes: **file-FIM 41%, repo-FIM 29%, file-plain 16%, repo-plain 14%** — 70% FIM,
43% multi-file. FIM spans are cut at random **character** positions, not token boundaries,
because real cursors sit mid-token (`def get_us|`); byte-level BPE with `add_prefix_space=False`
makes tokenizing the three pieces separately lossless, so this costs nothing. No document is
ever split across a record.

Filters: `file_timestamp >= 2024-01-01`, 198-language allowlist, `is_vendor == False`,
120 B–100 kB, max line ≤1000, mean line ≤100, alnum ≥0.25, no generated-file headers, exact
dedup on `content_id`. Accepting `permissive` + `no_license` (the only two values present;
`permissive AND 2024+` yields 0.1%, so filtering on it is infeasible) — fine for a personal
editor model, would matter if the weights were published.

## Tokenizer

32,768 byte-level BPE trained on a **language-balanced** 480 MB sample (0.6× the training mix
+ 0.4× uniform, so rare languages get real vocabulary presence). Measured on balanced held-out
fresh code:

| tokenizer | vocab | bytes/token | embedding @ d=768 |
|---|---|---|---|
| **ours** | 32,768 | **3.327** | **25.2M** |
| StarCoder2 | 49,152 | 3.361 | 37.7M |
| Qwen2.5-Coder | 151,665 | 3.781 | 116.5M |
| GPT-2 | 50,257 | 2.218 | 38.6M |

Honest read: ours is **1.0% worse than StarCoder2** — a wash. The justification is equal
compression at 12.5M fewer parameters (11% of the model redirected from a lookup table into
layers), native FIM sentinels so no untrained embeddings later, and 21 s to train.

## Budget

| Step | Where | Cost |
|---|---|---|
| Corpus probe, tokenizer, 3 dataset builds, holdout | Modal CPU fan-out | $6.06 |
| GPU availability probe | Modal | $0.06 |
| Smoke (4× L4, optimiser A/B) | Modal | $1.07 |
| Validation (phase switch, ckpt, resume, export) | 1× L4 | $0.30 |
| First pretrain attempt — OOM on step 1 | 4× L4 | $1.39 |
| Eval + serve path validation | 1× L4 | $0.37 |
| **Pretrain, 4.6h** | **4× L4** | **~$16.0** |
| Full eval, quantise, redeploy | 1× L4 | ~$0.7 |
| | | **~$27.5 of $30** |

Cost is bounded twice: the container `timeout` is a constant in the source (17700 s ⇒ $17.27
ceiling) and the loop stops on its own wall-clock budget well inside it. `pretrain` is a
dry-run without `--go` and refuses to launch if the ceiling would break the remaining budget.
`spend.json` is the ledger, calibrated against the Modal dashboard — estimating from a worker's
own wall clock ran 30% low, because Modal bills the whole container lifetime and re-bills
preempted containers.

## Eval

| What | Why it is the honest test |
|---|---|
| FIM exact-match + edit similarity, per language, single-file **and** multi-file | the only number that says whether it works in an editor |
| Bits per byte vs Qwen2.5-Coder-0.5B | perplexity is not comparable across tokenizers; bits-per-byte is |
| Bits per byte, fresh 2024+ **vs** pre-2023, same corpus and language mix | does training on fresh-only data buy anything, or is the premise imagined |
| Latency at 16 / 64 / 128 new tokens | an autocomplete model that takes a second is not one |

Honest expectation: this loses badly to Qwen2.5-Coder-1.5B, which saw ~3900× more pretraining
tokens. Its value is that every weight is yours, it runs on CPU, and the fresh-vs-stale gap is
a real measurement rather than a demo.

## Ship

```
python3 finish.py download     # HF weights + tokenizer + metrics off the volume
python3 finish.py gguf         # f16 / q8_0 / q4_k_m + prove tokenizer equivalence
python3 finish.py test         # greedy FIM completions on 4 ARM cores
python3 finish.py config <url> # minuet-ai.nvim, single-file and repo-aware
```

The served endpoint is L4 (decoding a 113M model is bandwidth-bound; an H100 would be waste),
`min_containers=0`, `max_containers=1`, 120 s scaledown, and a **bearer token is required** —
a public unauthenticated GPU endpoint is a bad default whatever the blast radius. Verified:
401 without a token, 401 with a wrong one, OpenAI-shaped JSON and SSE with a right one.

`modal app stop` everything at the end. Whatever is still only on Modal when the credit dies
is not really yours.
