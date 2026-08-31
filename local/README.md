# fimcoder-113m on your machine

A 113M-param fill-in-the-middle code model trained from scratch on 2024+ code only,
4096-token context, 32k code tokenizer with FIM sentinels. Runs on a 4060 with room to
spare — the weights are 116 MB.

## 1. Copy two things over

From **your** machine (this box is the remote one):

```bash
scp ec2:~/src/tries/2026-08-30-modal/out/fimcoder-113m/fimcoder-113m-q8_0.gguf .
# (this repo already contains local/)
sha256sum -c local/SHA256SUMS   # paths in the file are relative to the repo root
```

`q8_0` (116 MB) is what I benchmarked and is effectively lossless here. `q4_k_m`
(71 MB) also works if you want it smaller; on a 4060 there is no reason to.

## 2. Start it

```bash
./local/setup_local.sh ./fimcoder-113m-q8_0.gguf
```

It finds a runtime in this order — `llama-server` on PATH, the official
`ghcr.io/ggml-org/llama.cpp:server-cuda` docker image, or it prints the CUDA build
commands. Then it runs four real completions and measures latency, so you see what
your hardware actually does rather than what I predicted.

Measured on the machine that trained it (4 ARM CPU cores, no GPU), for reference:

```
rust         50 ms   ->  push(2);
python       46 ms   ->  read()
typescript   37 ms   ->  sum(xs);
go           44 ms   ->  < b {

2627-token window:  cold 5100 ms   warm 28 ms
```

**Cold vs warm is the whole story for editor feel.** Cold is a full prefill of the
window; warm is llama.cpp reusing the cached prefix, which is what happens once you
are typing inside one buffer. That is a 180x difference on CPU. On a 4060 expect cold
in the low hundreds of milliseconds and warm in single digits.

## 3. Wire up nvim

Copy `minuet.lua` into your lazy.nvim plugin directory (or paste the spec into your
config). `minuet_repo.lua` is a drop-in replacement for its `template.prompt` that
feeds your other open buffers as repo context, in the `<|repo_name|>` /
`<|file_sep|>` shape 29% of the training data used.

If completions feel slow on the *first* request in a file, lower `context_window`.
If they feel wrong, lower `temperature` — this model is best near greedy.

## 4. What to expect

Honest numbers from 300 held-out FIM examples across 166 languages:

| | exact match | edit similarity | first line exact |
|---|---|---|---|
| all | 1.3% | 25.8% | 11.3% |
| multi-file | 1.9% | 25.1% | 13.0% |
| single-file | 0.7% | 26.6% | 9.6% |

It reliably produces syntactically valid, language-appropriate code and gets short
continuations right maybe one time in nine. It does not know your codebase's
semantics. `v.push(2)`, `fh.read()`, `if a < b {` — yes. Anything requiring a train of
thought — no.

Best languages measured: Hack 55%, Vue 53%, Svelte 37%, Lua 36% edit similarity.
Worst: Swift 16%, PLpgSQL 14%, TeX 9%.

The one genuinely interesting property: it predicts **2024+ code 8.7% better than
pre-2023 code** (0.6084 vs 0.6666 bits/byte), where Qwen2.5-Coder-0.5B is flat at
0.4% on the identical documents. Training on fresh-only data left a measurable mark.
It is still 49% behind Qwen in absolute terms, which is what 1.4B training tokens
against ~5.5T buys you.

## Other formats

`out/fimcoder-113m/` also has ONNX (`onnx-int4`, 74 MB, verified to produce
byte-identical greedy output to torch) and the HF safetensors checkpoint, which loads
in transformers or vLLM unmodified.
