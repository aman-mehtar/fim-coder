# CONTEXT — verified facts and everything that went wrong

Handoff doc. Everything here was **measured in-session**. Read `PLAN.md` for what is being
built; read this for why, and for the traps already paid for.

**Status 2026-08-31 10:30 UTC:** pretraining in flight, $8.96 of $30 spent, ~$5 of genuine
buffer after the run and eval. Credit expires end of 2026-08-31.

---

## 1. The constraint that reshaped everything

The workspace cannot launch H100/H200/B200/L40S/A100 — Modal's client refuses **locally**,
before scheduling anything:

```
InvalidError('Please add a payment method to use H100 GPU functions.')
```

Two traps in that: it is raised at *app creation*, so one gated `gpu=` anywhere in a file kills
the whole app (probe GPU types one file at a time), and it is not a balance problem — credit
spends normally on CPU and on T4/L4/A10.

The user added a card mid-session; Stripe declined it. So: **L4 only**, and the model dropped
from the originally planned 304M to 113M as the compute budget fell ~8×.

## 2. Vendor TFLOPS for L4/L40S/A10 are sparsity numbers

NVIDIA quotes bf16 tensor throughput **with 2:1 structured sparsity** for the Ada/consumer-class
datacenter cards. Training never uses sparsity. Dense = SMs × 512 bf16 FLOP/SM/clock × boost:

| GPU | quoted | dense | |
|---|---|---|---|
| L4 | 121 | **60.6** | 58 SM × 2.04 GHz |
| L40S | 362 | **183** | 142 SM × 2.52 GHz |
| A10 | 125 | **62.5** | 72 SM × 1.695 GHz |

H100/H200 (989), A100 (312) and B200 (2250) are already dense and need no correction. T4 is
Turing: 65 TFLOP/s fp16, no bf16, no sparsity — and PyTorch SDPA's flash backend needs sm_80+,
so T4 loses attention performance too.

Symptom that exposed it: an MFU tracker reporting 17.9% while the model actually ran at ~36%,
and a "MFU < 25%" warning firing falsely. `gpu_config.py` had the sparse numbers with a comment
claiming sparsity was off.

## 3. Bugs that cost money or would have

| Bug | Symptom | Fix |
|---|---|---|
| Autotune ignored optimizer state | accepted micro_bs 16 at 19.1 GB / 22 GB, then **OOM on step 1** after 23 min (**$1.39**) — AdamW allocates `exp_avg`/`exp_avg_sq` lazily on the *first* `step()`, 0.9 GB, after the probe finished | `autotune_micro_batch(n_params=...)` subtracts `8 bytes/param` from the budget; also refines by 1.5× since power-of-two steps waste up to 33% |
| `PYTORCH_CUDA_ALLOC_CONF` set too late | silently a no-op: `apply_perf_env()` runs after `GPUInfo.detect()` has already initialised the CUDA allocator | moved into the container image `.env()` |
| Online best-fit packing | **60.5% fill** — a stream of ~1100-token documents pairs with nothing at seq 2048, so 40% of every record was padding, i.e. 40% of compute | buffer ~6000 docs and pack largest-first (best-fit **decreasing**) → 99.9% |
| One document per repo per language | 1.6 docs/repo, 1.7B tokens instead of 4B, so no per-language cap ever bound and the mix collapsed to raw availability (23.9% off target) | walk every file of an under-quota language; ratio-ordered with scarcest-share tie-break |
| Cap slack inflated the mix | capped languages landed at `1.15 × share × target/achieved` = 1.24× their share (14.0% off) | solve the fixed point from measured availability; slack 1.02 |
| `warmup_frac` × placeholder horizon | first 24 steps ran at **lr ≈ 1e-9** (25-million-step warmup) because `total` is 1e9 until throughput is measured | fixed short warmup until the horizon is real |
| Resume restored the checkpoint's `total` | resume exited immediately having done 0 steps | keep the original WSD horizon; only an explicit step count overrides |
| NCCL port reuse | `ncclRemoteError` on the second arm of a multi-arm smoke — re-init on a socket in TIME_WAIT | fresh `MASTER_PORT` per call |
| `modal.parameter()` on a web-endpoint class | container **crash-looped**: a bare request does not supply the parameter, so it used a stale default tag | plain class attribute from an image env var, plus a fallback to whatever checkpoint exists |
| `authorization: str = ""` in a FastAPI handler | read as a **query** parameter, not a header, so every authenticated request was 401 while looking correctly rejected | `fastapi.Header`, guarded by try/except (fastapi is not in the local CLI env) |
| `modal.Function.with_options` | does not exist in modal 1.2.6 — would have crashed at launch | GPU/cpu/memory/timeout as decorator constants, one function per configuration |
| Hand-recomputed llama.cpp `chkhsh` | my value `c102c8b8…` ≠ the converter's `586f46ac…`; a wrong pre-tokenizer does not error, it produces plausible garbage | read the hash from `convert_hf_to_gguf.py --verbose` and patch that |
| `llama-tokenize` escape processing | 3/60 phantom tokenizer mismatches — it processes `\n`, `\t`, `\r` in its input by default | `--no-escape`; then **120/120 documents identical** |

Also: the local cost ledger ran **30% low** against the Modal dashboard, because Modal bills the
whole container lifetime (image pull, startup, volume commit) and re-bills preempted containers
that restart. The CPU fan-out took several preemptions. `cost.py` now carries `OVERHEAD = 1.30`
for forecasts and trusts the dashboard for actuals.

## 4. Dataset facts (`HuggingFaceCode/stack-v3-train`)

8192 parquet parts, ~21k repos/part, 3 row groups each, one row per repository, contents inline.
Ungated. Crawl completed Aug 2025.

- **`file_timestamp` is int64 Unix SECONDS**, not ISO. 2024-01-01 = `1704067200`.
- **Parts are sorted by `repo_path`** — reading `parts[0:N]` samples a narrow alphabetical slice
  of GitHub. Shuffle part indices with a fixed seed; keep probe / train / val / eval slices disjoint.
- **The freshness filter rewrites the language mix.** C++ is 15.2% of the corpus but 3.1% of
  2024+ code; C# is 2.0% overall but 12.8% fresh. Any per-language budget computed from the
  published corpus stats is wrong — measure the fresh distribution first (that is what the probe
  is for: 384 row groups, 706k repos, 10.2M files, $0.37).
- **`license_type` ∈ {`permissive`, `no_license`}** only (1.1% / 98.9%); `non_permissive` is
  excluded upstream. `permissive AND 2024+` = 0.1%.
- **`stars: 0` is normal.** A `MIN_STARS` filter silently drops most repos.
- Measured: 1 row group = ~1840 usable repos, 77.7 MB of kept text, ~23M tokens.
- **Jupyter Notebook and Linear Programming have zero fresh files** that survive the size cap
  (notebooks routinely exceed 100 kB), so they were dropped from the 200-language mix → 198.

Arrow trick that makes prep cheap: `pc.list_flatten` + `pc.list_parent_indices` on the `files`
column, build the mask from metadata fields only, then `.take()`. File bodies are never
materialised for rows that fail the filter — 158 GB decoded down to 29.8 GB kept, and only the
survivors ever reach Python.

## 5. Environment

```
Modal        workspace amanmstudies, client 1.2.6 (needs PATH=$HOME/.local/bin:$PATH)
             no billing CLI exists — spend lives in spend.json, calibrated to the dashboard
Local box    4x Neoverse-N1 (ARM64), 15GB RAM, ~6GB free disk, no GPU
             system python is 3.9 — the modal CLI runs under it, so modal_*.py must avoid
             PEP 604 (`X | None`) in signatures. Use .venv (3.12) for everything else.
llama.cpp    built at ~/llama.cpp (llama-quantize, llama-cli, llama-tokenize)
             conversion/base.py patched with our chkhsh -> starcoder
gigatoken    0.10.0, ~1000x faster than HF tokenizers; cannot reload our own tokenizer.json
             ("Unsupported pre_tokenizer type: Sequence"), so benchmarks route through HF
```

## 6. Decisions already litigated — do not reopen

| Option | Why rejected |
|---|---|
| General/chat model from scratch | budget buys ~1.4B tokens; SmolLM2-360M used 4T. Nothing to plug it into. |
| Qwen3.5 as the FIM base | `Qwen3_5ForConditionalGeneration`: multimodal wrapper, 248,320 vocab, no FIM sentinels |
| Fine-tune Qwen2.5-Coder instead of from-scratch | offered explicitly with costs when the compute collapsed; user chose from-scratch |
| Qwen LoRA instruct-tune (was ~15% of budget) | **dropped by the user** once the budget tightened |
| 15k vocab | measured ~22% more tokens/char vs 32k; net ~10% less code per dollar |
| Qwen2.5-Coder's 152k tokenizer | best compression but 116M embedding params on a 113M model |
| Multi-GPU for quality | cost is GPU-seconds. 4×L4 for 4.5h == 1×L4 for 18h in dollars and tokens. It buys wall-clock, never quality — which is still worth it here, because the credit expires today. |
| `license_type == "permissive"` filter | 0.1% yield with the 2024+ filter |

## 7. Working style that paid off

- Never launch a GPU into unsmoked code. The $0.30 validation run caught two bugs that would
  each have wasted a 4.6-hour job.
- Verify against the real API/data before trusting a number. Wrong-until-measured this session:
  license yield, language mix, `MIN_STARS`, `file_timestamp` type, L4's peak FLOPS, packing
  efficiency, the llama.cpp fingerprint, and my own cost estimates.
- Prove the *downstream* paths against a throwaway checkpoint while the real run trains. Export,
  quantisation, tokenizer equivalence, eval and serving were all validated before there was
  anything worth shipping.
