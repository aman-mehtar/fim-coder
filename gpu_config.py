"""
GPU autodetection, MFU accounting, and micro-batch autotuning.

Modal may silently upgrade a `gpu="H100"` request to an H200 at no extra cost (their docs
confirm the price does not change). H200 has the same compute as H100 but 141GB HBM3e at
4.8TB/s instead of 80GB at 3.35TB/s. Free memory is only useful if we actually spend it, so
this module detects what we landed on and scales the micro-batch to match.

Usage inside a Modal GPU function:

    from gpu_config import GPUInfo, autotune_micro_batch, MFUTracker

    gpu = GPUInfo.detect()
    gpu.apply_perf_env()
    micro_bs = autotune_micro_batch(build_model_fn, seq_len=2048, gpu=gpu)
    tracker = MFUTracker(gpu, n_params_non_embed, seq_len=2048)
    ...
    tracker.step(tokens_this_step)   # -> logs tok/s and MFU
"""

from __future__ import annotations

import dataclasses
import time

# Dense bf16 tensor-core peak, sparsity OFF (the number you can actually hit in training).
# Vendor specs quote 2x these with 2:4 sparsity; using those would halve reported MFU.
PEAK_BF16_TFLOPS = {
    "H100": 989.0,   # SXM5
    "H200": 989.0,   # same SM count/clocks as H100 SXM, more+faster HBM
    "B200": 2250.0,
    "B300": 2500.0,
    "A100": 312.0,
    # The Ada/consumer-class cards are the ones vendors quote WITH sparsity, and the
    # earlier numbers here (L40S 362, L4 121, A10 125) were exactly those. Dense
    # values below are SMs x 512 bf16 FLOP/SM/clock x boost clock, which matches the
    # datasheets once sparsity is switched off. Using the sparse figure halves every
    # reported MFU and, worse, doubles the compute you think you have bought.
    "L40S": 183.0,   # 142 SM x 2.52 GHz
    "L4": 60.6,      # 58 SM x 2.04 GHz
    "A10": 62.5,     # 72 SM x 1.695 GHz
    "T4": 65.0,      # Turing: fp16 dense, no bf16 and no sparsity at all
}


@dataclasses.dataclass
class GPUInfo:
    name: str
    family: str          # H100 / H200 / B200 / ...
    total_gb: float
    peak_bf16_flops: float
    sm_count: int
    capability: tuple[int, int]

    @classmethod
    def detect(cls) -> "GPUInfo":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("no CUDA device visible")

        props = torch.cuda.get_device_properties(0)
        name = props.name  # e.g. "NVIDIA H200"
        upper = name.upper().replace("-", "").replace(" ", "")

        family = "H100"
        for key in ("B300", "B200", "H200", "H100", "A100", "L40S", "L4", "A10", "T4"):
            if key in upper:
                family = key
                break

        total_gb = props.total_memory / 1024**3
        # Distinguish H200 (141GB) from H100 (80GB) even if the name string is unhelpful.
        if family == "H100" and total_gb > 100:
            family = "H200"

        info = cls(
            name=name,
            family=family,
            total_gb=total_gb,
            peak_bf16_flops=PEAK_BF16_TFLOPS.get(family, 989.0) * 1e12,
            sm_count=props.multi_processor_count,
            capability=(props.major, props.minor),
        )
        print(
            f"[gpu] {info.name} | family={info.family} | {info.total_gb:.0f}GB | "
            f"{info.sm_count} SMs | sm_{info.capability[0]}{info.capability[1]} | "
            f"peak bf16 {info.peak_bf16_flops / 1e12:.0f} TFLOP/s"
        )
        if info.family == "H200":
            print("[gpu] free H200 upgrade detected -> scaling micro-batch up")
        return info

    @property
    def is_hopper_or_newer(self) -> bool:
        return self.capability[0] >= 9

    def apply_perf_env(self) -> None:
        """Everything that buys MFU without touching the model definition."""
        import torch

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

        # Prefer flash attention; fall back to mem-efficient. Keep math off so a silent
        # fallback to the slow path shows up as an error rather than as bad MFU.
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel  # torch >= 2.5

            self._sdpa_ctx = sdpa_kernel(
                [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
            )
        except Exception:
            self._sdpa_ctx = None
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(False)

        # NOTE: PYTORCH_CUDA_ALLOC_CONF must be set BEFORE the CUDA allocator
        # initialises, and detect() has already touched the device by the time we get
        # here, so setting it now is a no-op. It belongs in the container image env.
        # Left as a fallback for callers that construct GPUInfo without detect().
        import os

        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def autotune_micro_batch(
    build_step,
    seq_len: int,
    gpu: GPUInfo,
    start: int | None = None,
    max_bs: int = 128,
    n_params: int = 0,
    opt_bytes_per_param: int = 8,
) -> int:
    """Largest micro-batch that runs without OOM, with room left for the optimizer.

    `build_step(micro_bs)` must run one full fwd+bwd and raise on OOM. It does NOT
    include the optimizer's state: AdamW allocates exp_avg and exp_avg_sq lazily on
    the FIRST `step()`, which is after autotuning is over. Ignoring that is how a
    probe accepts a micro-batch at 87% of memory and then dies on step 1 -- measured
    the expensive way: peak 19.1GB of 22GB accepted, then OOM allocating 1022 MiB.
    So pass `n_params` and reserve for it explicitly.
    """
    import torch

    if start is None:
        start = 16 if gpu.total_gb > 100 else 8

    reserve = n_params * opt_bytes_per_param / 1024**3
    budget = 0.90 * gpu.total_gb - reserve
    if n_params:
        print(f"[autotune] budget {budget:.1f}GB = 90% of {gpu.total_gb:.0f}GB minus "
              f"{reserve:.1f}GB of optimizer state")

    def fits(bs):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            build_step(bs)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"[autotune] micro_bs={bs:3d} OOM")
            return None
        peak = torch.cuda.max_memory_allocated() / 1024**3
        ok = peak <= budget
        print(f"[autotune] micro_bs={bs:3d} peak {peak:.1f}GB / budget {budget:.1f}GB"
              f"{'' if ok else '  -> over budget'}")
        return peak if ok else None

    best, bs = 0, start
    while bs <= max_bs:
        if fits(bs) is None:
            break
        best = bs
        bs *= 2
    if best:
        mid = int(best * 1.5)                  # power-of-two steps waste up to 33%
        if mid <= max_bs and fits(mid) is not None:
            best = mid

    if best == 0:
        raise RuntimeError(f"even micro_bs={start} does not fit at seq_len={seq_len}")
    torch.cuda.empty_cache()
    print(f"[autotune] micro_bs={best} @ seq_len={seq_len} "
          f"({best * seq_len:,} tokens/micro-step)")
    return best


class MFUTracker:
    """Model FLOPs Utilisation, using the 6ND + attention formulation.

    Counts only *useful* math (the standard Chinchilla convention):
      fwd+bwd  ≈ 6 * N_non_embed * T   +   12 * L * T^2 * d   (attention scores)
    Reports both instantaneous and rolling MFU so a regression is visible immediately.
    """

    def __init__(
        self,
        gpu: GPUInfo,
        n_params_non_embed: int,
        seq_len: int,
        n_layers: int,
        d_model: int,
        log_every: int = 10,
    ):
        self.gpu = gpu
        self.n = n_params_non_embed
        self.seq_len = seq_len
        self.attn_flops_per_token = 12.0 * n_layers * d_model * seq_len
        self.log_every = log_every
        self.step_idx = 0
        self.t_last = time.perf_counter()
        self.tokens_since = 0
        self.total_tokens = 0
        self.t_start = self.t_last
        self.best_mfu = 0.0

    def flops_per_token(self) -> float:
        return 6.0 * self.n + self.attn_flops_per_token

    def step(self, tokens: int) -> dict | None:
        self.step_idx += 1
        self.tokens_since += tokens
        self.total_tokens += tokens
        if self.step_idx % self.log_every:
            return None

        now = time.perf_counter()
        dt = now - self.t_last
        tps = self.tokens_since / dt
        mfu = tps * self.flops_per_token() / self.gpu.peak_bf16_flops
        self.best_mfu = max(self.best_mfu, mfu)

        elapsed = now - self.t_start
        avg_mfu = (
            (self.total_tokens / elapsed) * self.flops_per_token() / self.gpu.peak_bf16_flops
        )
        self.t_last = now
        self.tokens_since = 0

        stats = {
            "step": self.step_idx,
            "tok_per_s": tps,
            "mfu": mfu,
            "avg_mfu": avg_mfu,
            "total_tokens": self.total_tokens,
            "elapsed_h": elapsed / 3600,
        }
        print(
            f"[mfu] step {self.step_idx:>6} | {tps / 1e3:6.1f}k tok/s | "
            f"MFU {mfu * 100:5.1f}% (avg {avg_mfu * 100:4.1f}%) | "
            f"{self.total_tokens / 1e9:5.2f}B tok | {elapsed / 3600:4.2f}h"
        )
        if self.step_idx == self.log_every and mfu < 0.25:
            print(
                "[mfu] WARNING: MFU < 25%. Check that torch.compile actually compiled, "
                "that SDPA chose flash, and that micro_bs is at the autotuned value."
            )
        return stats

    def projected_tokens(self, budget_hours: float) -> int:
        """Extrapolate the total token count for a given wall-clock budget."""
        elapsed = time.perf_counter() - self.t_start
        if elapsed <= 0 or self.total_tokens == 0:
            return 0
        return int(self.total_tokens / elapsed * budget_hours * 3600)


def recommended_batch_plan(gpu: GPUInfo, micro_bs: int, seq_len: int, target_tokens: int = 524_288):
    """Grad-accum steps to hit ~0.5M tokens per optimizer step."""
    per_micro = micro_bs * seq_len
    accum = max(1, round(target_tokens / per_micro))
    actual = per_micro * accum
    print(
        f"[batch] micro_bs={micro_bs} x seq={seq_len} x accum={accum} "
        f"= {actual:,} tokens/step"
    )
    return accum, actual
