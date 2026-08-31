# fimcoder-113m — what it is, how well it works, and why it isn't better

A fill-in-the-middle code model trained from scratch on 2024+ code only, on $30 of
expiring Modal credit, in one night. This document is the honest assessment.

**Short version: it is not smart, and that is the expected result rather than a bug.**
It writes syntactically valid, language-appropriate code and gets short completions right
about one time in nine. The reason is the compute budget, quantified below.

---

## 1. Why it feels stupid

It saw **1.4 billion training tokens**. Here is that number in context:

| model | params | tokens | training compute | vs this model |
|---|---|---|---|---|
| **fimcoder-113m** | 88M non-emb | **1.41B** | 7.4e17 FLOPs | 1x |
| GPT-2 small (2019) | 124M | 10B | 7.4e18 | **10x** |
| Qwen2.5-Coder-0.5B | 494M | 5.5T | 1.6e22 | **21,900x** |
| Qwen2.5-Coder-1.5B | 1.5B | 5.5T | 5.0e22 | **66,600x** |
| StarCoder2-3B | 3B | 3.3T | 5.9e22 | **79,900x** |

The smallest model anyone would actually use for autocomplete had **22,000 times** this
model's training compute. Not 22 times. The entire run cost $16.40 and 4.6 hours on four
L4s, which is roughly one ten-thousandth of the budget behind a production code model.

Two things forced the size. The Modal workspace could not launch H100/A100/L40S without a
payment method on file (the card was declined mid-project), leaving only T4/L4/A10 — and an
L4 delivers 76 dense bf16 TFLOP/s per dollar against an H100's 250, so the *same money*
bought 3.3x less compute. Given 4x L4 for 4.6 hours, 113M params on 1.4B tokens is
Chinchilla-optimal; a 214M model on the same budget would have landed at D/N ≈ 4.8 and been
measurably **worse**, not better.

So the model is doing about as well as ~7e17 FLOPs allows. There is no configuration bug to
find.

## 2. What it did learn

- **Syntax and idiom per language.** Completions are nearly always well-formed code in the
  right language, with balanced brackets.
- **When to stop.** It emits `<|endoftext|>` after a few tokens rather than rambling, so a
  larger `max_tokens` budget costs nothing. This matters more for editor feel than it sounds.
- **Short, local continuations.** `v.push(2)`, `fh.read()`, `if a < b {`, `xs.sum(xs)`.
- **A measurable preference for recent code** (§5), which was the point of the exercise.

What it did not learn: semantics, your codebase, or anything requiring more than one step of
reasoning. `def fib(n)` gets completed with `return n`, not `return fib(n-1) + fib(n-2)`.

## 3. FIM accuracy

1400 held-out examples, 166 languages, held-out repos from parts no training worker touched.
Ground truth capped to the token budget the model was given, so a long middle is not scored
as a miss for being long.

| set | n | exact match | EM (stripped) | edit similarity |
|---|---|---|---|---|
| all | 1400 | 2.4% | 2.6% | 26.2% |
| single-file FIM | 744 | 2.6% | 2.8% | 26.0% |
| multi-file FIM | 656 | 2.1% | 2.3% | 26.3% |

A separate 300-example run through llama.cpp q8_0 measured **first-line exact match at
11.3%** — the metric closest to what an editor shows you.

**Multi-file does not beat single-file.** An earlier 300-example run suggested it did
(1.9% vs 0.7% EM) and that was reported as evidence the 29% repo-level share of the training
mix paid off. At 744/656 examples it reverses and edit similarity is a dead heat. The honest
conclusion is no measurable difference.

Best and worst languages by edit similarity (n >= 6, 74 languages qualify):

```
HTTP 45.6   Hack 44.9   Vue 43.2   CSS 42.0   Blade 41.9   PHP 38.6   JSON 37.8
...
Svelte 15.2   SCSS 14.6   Batchfile 14.2   Jinja 12.6   TeX 12.0   RPM Spec 10.2
```

Markup and config-shaped languages score best, which is what you would expect: they are the
most predictable. The languages you care about sit mid-table.

## 4. Speed

Latency is the one dimension where being tiny is an advantage.

| where | cold (full prefill) | warm (cached prefix) |
|---|---|---|
| L4 GPU | 95-103 ms | — |
| 4x Neoverse-N1 CPU, q8_0, 2627-token window | 5100 ms | **28 ms** |

Cold vs warm is the whole story for editor feel: warm is llama.cpp reusing the cached
prefix, which is what happens on every keystroke after the first inside one buffer. On a
consumer GPU both numbers are small.

## 5. The one interesting result: does training on fresh code help?

Bits per byte (lower is better) on **byte-identical** documents — 1500 per condition,
5.4 MB / 5.1 MB, 56 languages, same corpus, same filters, same code path, 1024-token windows.
Bits per byte rather than perplexity because the two models have different vocabularies.

| model | fresh (2024+) | pre-2023 | gap | relative |
|---|---|---|---|---|
| **fimcoder-113m** | 0.5413 | 0.6412 | +0.0999 | **+15.6%** |
| Qwen2.5-Coder-0.5B | 0.3550 | 0.3989 | +0.0439 | **+11.0%** |

Both models find 2024+ code more predictable, so **most of the effect is a property of the
corpus, not of the training data cutoff.** Our model's *excess* preference for fresh code is
**+0.056 bits/byte, or +4.6 percentage points relative** — real, but far smaller than a
naive reading of our own gap suggests.

This number moved three times as the sample grew (+0.058 vs a seemingly flat baseline at 320
documents, −0.001 at 40, then this at 1500). The 1500-document measurement is the one to
quote; the earlier ones were sampling noise, and the earlier framing of "34x a flat
baseline" was wrong.

In absolute terms this model is **52% worse** than Qwen2.5-Coder-0.5B on fresh code
(0.5413 vs 0.3550), which is what 22,000x less compute buys.

## 6. Training run

```
architecture   LlamaForCausalLM, d_model 768, 14 layers, 12 heads / 4 KV (GQA), d_ff 2048
               RMSNorm, SwiGLU, RoPE theta 50k, tied embeddings, 113.3M params
tokenizer      32,768 byte-level BPE trained on a language-balanced sample
               3.327 bytes/token — 1.0% behind StarCoder2 at 12.5M fewer embedding params
context        2048 for the first 75% of steps, then 4096 for the WSD decay leg
optimiser      AdamW, peak lr 2.4e-3, WSD (2.5% warmup, 1-sqrt decay over the last 15%)
batch          524,288 tokens/step, 4x L4 with one flat all-reduce per step
result         2685 steps, 1.406B tokens, 4.60h, 31.3% MFU, val loss 1.3159 (ppl 3.73)
cost           $16.40 of $30
```

MFU of 31.3% is against L4's **dense** bf16 peak of 60.6 TFLOP/s. NVIDIA quotes 121 for the
L4, which is the 2:1-sparsity figure; using it would have halved every reported MFU and
doubled the compute the plan thought it had.

## 7. Data

`HuggingFaceCode/stack-v3-train`, filtered to `file_timestamp >= 2024-01-01`, 198 languages,
non-vendor, 120 B - 100 kB, max line <= 1000, mean line <= 100, alnum >= 0.25, no
generated-file headers, exact dedup on `content_id`.

```
seq 2048   1,726,351 records = 3.54B tokens   99.9% fill
seq 4096     303,265 records = 1.24B tokens  100.0% fill (repo-heavy mixture)
val            8,188 records =  16.8M tokens  168 languages
fimeval        2,853 examples, 166 languages, half single-file / half multi-file
```

Language mix is **5.6% total-variation distance** from target: Python 15.6% (target 14.6),
TypeScript 9.4 (8.9), JavaScript 8.8 (8.3), Java 5.3 (5.0), C++ 5.1 (4.8), Go 3.5 (3.2),
Rust 2.4 (2.8), Swift 1.4 (1.4). Reaching that took three builds; the first two were 23.9%
and 14.0% off target. See `CONTEXT.md`.

Document shapes: 41% single-file FIM, 29% repo-level FIM, 16% single-file plain, 14%
repo-level plain. FIM spans are cut at random **character** positions rather than token
boundaries, because real cursors sit mid-token.

## 8. If you wanted a model that is actually good

In descending order of value per dollar:

1. **Don't train from scratch.** Continue-pretrain Qwen2.5-Coder-0.5B on the same fresh
   corpus. It starts with 5.5T tokens of pretraining and already has FIM sentinels; the same
   $16 would buy a model that is genuinely useful in an editor. This project's from-scratch
   constraint was a choice, not a requirement.
2. **Get H100 access.** 250 dense TFLOP/s per dollar vs the L4's 76 — 3.3x more compute for
   the same money, which is the difference between 1.4B and ~4.6B tokens.
3. **More tokens at the same parameter count.** D/N here is 16; modern small models are
   deliberately overtrained to D/N of 100-1000. At fixed compute that trades against model
   size, but for a model that runs in an editor, inference speed is worth more than width.
4. Longer training on more languages will not fix the fundamentals. Nothing in the
   hyperparameters is leaving significant quality on the table.

## 9. Reproducing

```
modal run modal_prep.py::probe_all        # measure the fresh-code language distribution
modal run modal_prep.py::tokenizer_all    # train the 32k BPE on a balanced sample
modal run modal_prep.py::build_all        # uint16 shards, seq 2048 and seq 4096
modal run modal_prep.py::holdout          # val shard + FIM eval set
modal run modal_train.py::validate        # $0.25 gate: phase switch, ckpt, resume, export
modal run --detach modal_train.py::pretrain --hours 4.6 --gpu L4:4 --go
./ship.sh                                 # download, GGUF, ONNX, all evals, editor config
```

`cost.py` is the spend ledger — Modal has no billing CLI. It was calibrated against the
dashboard mid-project after running 47% low on CPU fan-outs; container lifetime, not loop
wall-clock, is what gets billed, and preempted containers are re-billed.
