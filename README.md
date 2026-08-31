# fim-coder

A fill-in-the-middle code model trained **from scratch** on 2024+ code only, on $30 of
expiring Modal credit, in one night. Plus the whole pipeline that built it.

```
fimcoder-113m   113.3M params · 4096-token context · 32k code tokenizer with FIM sentinels
                1.406B tokens of 2024-or-newer code across 198 languages
                4.60h on 4x L4 · $16.40 · val loss 1.3159 (ppl 3.73)
```

**Read [REPORT.md](REPORT.md) before you install it.** It is a small model and it behaves
like one: 2.4% exact match and ~11% first-line exact match on held-out FIM. It writes
syntactically valid, language-appropriate code and gets short continuations right about one
time in nine. That is what 1/22,000th of Qwen2.5-Coder-0.5B's training compute buys, and the
report explains the arithmetic.

The one result worth the exercise: on identical held-out documents it predicts 2024+ code
**15.6%** better than pre-2023 code, where Qwen2.5-Coder-0.5B manages 11.0% — an excess
freshness preference of +4.6 percentage points that is attributable to the data cutoff.

## Run it

Weights are not in this repo (see [WEIGHTS.md](WEIGHTS.md)). With `fimcoder-113m-q8_0.gguf`
in the current directory:

```bash
./local/setup_local.sh ./fimcoder-113m-q8_0.gguf
```

That picks a runtime — native `llama-server`, then the CUDA docker image — starts it, runs
four real completions, and prints measured cold/warm latency. Then copy `local/minuet.lua`
into your Neovim config for [minuet-ai.nvim](https://github.com/milanglacier/minuet-ai.nvim).

The prompt format is PSM with this model's own sentinels, which is why minuet needs a custom
`template.prompt`:

```
<|fim_prefix|>{before_cursor}<|fim_suffix|>{after_cursor}<|fim_middle|>
```

Use `/v1/completions`, not llama.cpp's `/infill` — the GGUF does not carry llama.cpp's FIM
token metadata. Keep temperature at 0-0.2.

## Layout

| | |
|---|---|
| `REPORT.md` | measurements, and an honest account of why it isn't better |
| `PLAN.md` | the build spec as executed |
| `CONTEXT.md` | every decision, and all 14 bugs with symptom and fix |
| `WEIGHTS.md` | where the weights are and how to rebuild every format |
| `fimlib.py` | corpus filtering, FIM span selection, document shapes, record packing |
| `trainlib.py` | model config, chunked cross-entropy, data, Muon, WSD schedule, HF export |
| `gpu_config.py` | GPU detection, micro-batch autotuning, MFU tracking |
| `modal_prep.py` | corpus probe, tokenizer training, shard builds, holdout sets |
| `modal_train.py` | pretraining on `L4:4` with a flat all-reduce, plus a $0.25 validation gate |
| `modal_eval.py` | GPU evaluation: FIM accuracy, bits-per-byte, freshness comparison |
| `modal_sft.py` | Qwen LoRA instruction-tuning (base model verified on L4, never run) |
| `modal_serve.py` | bearer-token OpenAI-compatible FIM endpoint |
| `eval_local.py` | CPU evaluation: bits-per-byte, FIM via llama-server, latency |
| `finish.py` | download, GGUF, ONNX, tokenizer equivalence, editor configs |
| `ship.sh` | the whole post-training sequence in one command |
| `cost.py` | spend ledger, because Modal has no billing CLI |
| `local/` | the setup script and Neovim configs |
| `results/` | every measurement as JSON, including the 278-point training curve |
| `models/` | architecture and tokenizer config (no weights) |
| `tokenizer/` | the trained 32k tokenizer |

## Notes on the data

Trained on `HuggingFaceCode/stack-v3-train` filtered to `file_timestamp >= 2024-01-01`. That
corpus contains `permissive` and `no_license` files; `non_permissive` is excluded upstream,
and filtering to `permissive` alone yields 0.1% of files once combined with the freshness
filter. **`no_license` means no license was detected, not public domain** — fine for a
personal model in your own editor, worth thinking about before publishing weights.
