# Weights

Published as **[GitHub Release assets](https://github.com/aman-mehtar/fim-coder/releases/latest)**,
not committed to git — they total ~700 MB, GitHub rejects files over 100 MB in a repo, and
release assets do not count against repository size or LFS quota.

```bash
gh release download v0.1.0 -R aman-mehtar/fim-coder            # everything
gh release download v0.1.0 -R aman-mehtar/fim-coder -p '*q8_0*' # just the one you want
```

| format | size | what it is for |
|---|---|---|
| `hf/` (safetensors) | 218 MB | the source of truth; loads in transformers and vLLM unmodified |
| `fimcoder-113m-f16.gguf` | 218 MB | llama.cpp, lossless |
| `fimcoder-113m-q8_0.gguf` | 116 MB | llama.cpp, what the benchmarks used |
| `fimcoder-113m-q4_k_m.gguf` | 71 MB | llama.cpp, smallest |
| `fimcoder-113m-onnx-int4.tar.gz` | 71 MB | onnxruntime-genai, KV-cached graph |

`onnx-fp32` (436 MB) is not published; regenerate it with `python3 finish.py onnx` if you
need full precision.

Checksums for the two GGUFs people actually use are in `local/SHA256SUMS`.

## Rebuilding from the HF checkpoint

```bash
python3 finish.py gguf    # f16 -> q8_0 -> q4_k_m, then verifies tokenizer equivalence
python3 finish.py onnx    # fp32 + int4 via onnxruntime-genai, then verifies greedy output
```

`finish.py gguf` patches your local `llama.cpp/conversion/base.py` to map this tokenizer's
fingerprint to the `starcoder` pre-tokenizer, which is byte-for-byte the same
(`Digits(individual_digits=True)` then `ByteLevel` with `add_prefix_space=False`). It reads
the fingerprint from the converter rather than recomputing it — recomputing by hand gave a
different value, and a wrong pre-tokenizer does not error, it silently produces plausible
garbage. The check that matters is the one it runs afterwards: **120/120 documents tokenize
identically** between HF `tokenizers` and `llama-tokenize`.

`finish.py onnx` greedy-decodes the same prompt through torch and through ONNX and requires
identical token ids. Both fp32 and int4 pass.

## Retraining from nothing

See `README.md` §Reproducing. Roughly $19 and six hours on Modal, most of it the 4.6h
pretraining run.

## Publishing

If you push these weights anywhere public, note the licensing caveat in `README.md`: the
training corpus is `permissive` + `no_license` code, and `no_license` is unlicensed rather
than public domain.
