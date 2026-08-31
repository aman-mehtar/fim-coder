"""
Model, data, optimiser and schedule for fimcoder-300m. Pure PyTorch — no Modal —
so every piece can be tested on a CPU box before a GPU is ever billed.

Deliberate choices:

* Stock `LlamaForCausalLM`. A hand-rolled transformer risks a silent bug that costs
  $20 and buys nothing; the standard class means vLLM and llama.cpp load the result
  with zero conversion work, which is the entire point of a model that lives in an
  editor.
* Chunked cross-entropy instead of `labels=`. Full logits for a 2048x24 batch at
  vocab 32768 are ~6.4 GB once upcast to fp32; computing the head and the loss in
  slices bounds that to a few hundred MB and lets the micro-batch stay large.
* WSD (warmup-stable-decay) LR, not cosine. Cosine locks the schedule to a step
  count guessed before the first measurement of throughput; WSD lets the decay
  start whenever we decide to stop and still yields a properly annealed model.
* Sequential-block reads with random shard/offset. Records were already shuffled in
  16k blocks at write time, so this gives decorrelated batches without random 4 KB
  reads against a network volume.
"""

from __future__ import annotations

import glob
import math
import os

import numpy as np
import torch


# ------------------------------------------------------------------- model

def model_config(vocab_size: int, seq: int = 4096, d_model: int = 1024, n_layers: int = 24,
                 n_heads: int = 16, n_kv: int = 4, d_ff: int = 2816, rope_theta: float = 50_000.0):
    """LlamaConfig for the 304M FIM model.

    `max_position_embeddings` is set to the FINAL context (4096) from the start so a
    checkpoint saved during the seq-2048 phase still declares the context the
    finished model serves. rope_theta 50k rather than 10k because the last phase of
    training runs at 4096 and a larger base extrapolates better at that length.
    """
    from transformers import LlamaConfig

    kw = dict(
        vocab_size=vocab_size,
        # SDPA picks the flash kernel on Hopper; `flash_attention_2` needs the
        # flash-attn package, which we deliberately do not build into the image.
        attn_implementation="sdpa",
        hidden_size=d_model,
        intermediate_size=d_ff,
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv,
        hidden_act="silu",
        max_position_embeddings=seq,
        rms_norm_eps=1e-5,
        tie_word_embeddings=True,
        attention_bias=False,
        mlp_bias=False,
        use_cache=False,
    )
    # transformers 5.x moved rope_theta into a rope_parameters dict; accept both.
    import inspect

    names = set(inspect.signature(LlamaConfig.__init__).parameters)
    if "rope_theta" in names:
        kw["rope_theta"] = rope_theta
    elif "rope_parameters" in names:
        kw["rope_parameters"] = {"rope_type": "default", "rope_theta": rope_theta}
    else:
        kw["rope_scaling"] = {"rope_type": "default", "rope_theta": rope_theta}
    cfg = LlamaConfig(**kw)
    cfg._attn_implementation = "sdpa"
    return cfg


def build_model(cfg, device="cuda", dtype=torch.bfloat16):
    from transformers import LlamaForCausalLM

    torch.manual_seed(1234)
    model = LlamaForCausalLM(cfg)
    model.config.use_cache = False
    model = model.to(device=device, dtype=dtype)
    return model


def param_counts(model):
    emb = model.get_input_embeddings().weight.numel()
    total = sum(p.numel() for p in model.parameters())
    return {"total": total, "embed": emb, "non_embed": total - emb}


# ------------------------------------------------------------------- loss

def chunked_ce(lm_head, hidden, labels, ignore_index=-100, chunk=8192):
    """Cross-entropy over `lm_head(hidden)` without materialising all logits.

    hidden: (B, T, C) already shifted by the caller. labels: (B, T) likewise.
    Returns (sum_loss, n_tokens) so gradient accumulation can normalise exactly
    once, over the true token count rather than per-micro-batch means.
    """
    h = hidden.reshape(-1, hidden.size(-1))
    y = labels.reshape(-1)
    keep = y != ignore_index
    h, y = h[keep], y[keep]
    n = y.numel()
    total = hidden.new_zeros((), dtype=torch.float32)
    for i in range(0, n, chunk):
        logits = lm_head(h[i : i + chunk]).float()
        total = total + torch.nn.functional.cross_entropy(
            logits, y[i : i + chunk], reduction="sum"
        )
    return total, n


def forward_loss(model, body, ids, pad_id, chunk=8192):
    """One forward pass returning (sum_loss, n_tokens). Padding never contributes."""
    x, y = ids[:, :-1], ids[:, 1:]
    h = body(input_ids=x).last_hidden_state
    y = y.masked_fill(y == pad_id, -100)
    return chunked_ce(model.lm_head, h, y, chunk=chunk)


# ------------------------------------------------------------------- data

class ShardData:
    """Packed uint16 records on disk, served as decorrelated micro-batches.

    Each micro-batch is a contiguous block from a random shard at a random offset.
    Records were shuffled in 16k blocks when written, so contiguity here does not
    mean correlation, and sequential reads keep a network-backed volume happy.
    """

    def __init__(self, pattern: str, seq: int, seed: int = 0):
        self.paths = sorted(glob.glob(pattern))
        if not self.paths:
            raise FileNotFoundError(pattern)
        self.seq = seq
        self.mm = []
        for p in self.paths:
            n = os.path.getsize(p) // (2 * seq)
            if n:
                self.mm.append(np.memmap(p, dtype=np.uint16, mode="r",
                                         shape=(n, seq)))
        self.counts = [m.shape[0] for m in self.mm]
        self.total = sum(self.counts)
        self.rng = np.random.default_rng(seed)
        self.served = 0

    def __repr__(self):
        return (f"ShardData({len(self.mm)} shards, {self.total:,} records, "
                f"{self.total * self.seq / 1e9:.2f}B tokens, seq={self.seq})")

    def batch(self, bs: int, device="cuda"):
        s = int(self.rng.integers(len(self.mm)))
        n = self.counts[s]
        if n <= bs:
            block = self.mm[s][:]
        else:
            o = int(self.rng.integers(n - bs))
            block = self.mm[s][o : o + bs]
        self.served += bs
        t = torch.from_numpy(np.ascontiguousarray(block).astype(np.int64))
        return t.to(device, non_blocking=True)

    def epochs_served(self) -> float:
        return self.served / max(self.total, 1)


# --------------------------------------------------------------- optimisers

@torch.no_grad()
def _newton_schulz(G, steps: int = 5, eps: float = 1e-7):
    """Quintic Newton-Schulz iteration: approximates the orthogonal factor of G.

    Runs in bf16 on purpose — the iteration only needs the singular values pushed
    toward 1, not high precision, and bf16 makes it nearly free.
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    X = X / (X.norm() + eps)
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    """Momentum-orthogonalised update for 2-D hidden weights (Jordan et al., 2024).

    Only the transformer's matrix parameters go here. Embeddings, the LM head, norms
    and any 1-D tensor keep AdamW: orthogonalisation is meaningless for them and
    actively harmful for the tied embedding.
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5,
                 weight_decay=0.0):
        super().__init__(list(params), dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                           ns_steps=ns_steps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(p.grad)
                buf = st["buf"]
                buf.mul_(g["momentum"]).add_(p.grad)
                upd = p.grad.lerp_(buf, g["momentum"]) if g["nesterov"] else buf
                upd = _newton_schulz(upd, g["ns_steps"]).view_as(p)
                if g["weight_decay"]:
                    p.mul_(1 - g["lr"] * g["weight_decay"])
                # RMS of an orthogonal matrix update is 1/sqrt(max(dim)); this factor
                # makes the step size comparable across differently-shaped weights.
                scale = max(1.0, p.size(-2) / p.size(-1)) ** 0.5
                p.add_(upd.to(p.dtype), alpha=-g["lr"] * scale)


def make_optimizers(model, opt: str, lr: float, muon_lr: float = 0.02,
                    wd: float = 0.1, betas=(0.9, 0.95)):
    """Returns (list_of_optimizers, list_of_base_lrs) so the schedule can scale both."""
    decay, no_decay, matrices = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_matrix = p.ndim == 2 and "embed_tokens" not in n and "lm_head" not in n
        if opt == "muon" and is_matrix:
            matrices.append(p)
        elif p.ndim >= 2:
            decay.append(p)
        else:
            no_decay.append(p)

    adam = torch.optim.AdamW(
        [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, betas=betas, eps=1e-8, fused=True,
    )
    if opt == "muon" and matrices:
        return [adam, Muon(matrices, lr=muon_lr, weight_decay=wd)], [lr, muon_lr]
    return [adam], [lr]


def wsd_lr(step: int, total: int, warmup: float = 0.02, decay: float = 0.15,
           floor: float = 0.02) -> float:
    """Warmup -> stable -> 1-sqrt decay, as a multiplier on the base LR.

    The 1-sqrt shape (Hagele et al., 2024) beats linear and cosine for the decay
    leg of WSD, and `total` can be revised mid-run: whatever step we choose to stop
    at, the last `decay` fraction anneals properly.
    """
    w = max(1, int(total * warmup))
    if step < w:
        return (step + 1) / w
    d0 = int(total * (1 - decay))
    if step < d0:
        return 1.0
    frac = min(1.0, (step - d0) / max(1, total - d0))
    return max(floor, 1.0 - math.sqrt(frac))


def set_lr(opts, bases, mult: float):
    for o, base in zip(opts, bases):
        for g in o.param_groups:
            g["lr"] = base * mult


# ------------------------------------------------------------------- export

def export_hf(model, tok_dir: str, out_dir: str, seq: int = 4096):
    """Write a checkpoint that vLLM, transformers and llama.cpp all load unmodified.

    `save_pretrained` alone is not enough: a bare `tokenizer.json` has no
    `tokenizer_config.json`, so `AutoTokenizer.from_pretrained` fails and every
    downstream tool with it. Writing the small config files here is what makes the
    result a model you can actually serve rather than a tensor dump.
    """
    import json
    import shutil

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(tok_dir, "specials.json")) as fh:
        sp = json.load(fh)
    eot, pad = sp["<|endoftext|>"], sp["<|fim_pad|>"]

    model.config.eos_token_id = eot
    model.config.bos_token_id = eot
    model.config.pad_token_id = pad
    model.config.max_position_embeddings = seq
    model.save_pretrained(out_dir, safe_serialization=True)

    shutil.copyfile(os.path.join(tok_dir, "tokenizer.json"),
                    os.path.join(out_dir, "tokenizer.json"))
    shutil.copyfile(os.path.join(tok_dir, "specials.json"),
                    os.path.join(out_dir, "specials.json"))

    added = {
        str(i): {"content": t, "lstrip": False, "normalized": False, "rstrip": False,
                 "single_word": False, "special": True}
        for t, i in sp.items()
    }
    with open(os.path.join(out_dir, "tokenizer_config.json"), "w") as fh:
        json.dump({
            "tokenizer_class": "PreTrainedTokenizerFast",
            "model_max_length": seq,
            "clean_up_tokenization_spaces": False,
            "bos_token": "<|endoftext|>",
            "eos_token": "<|endoftext|>",
            "pad_token": "<|fim_pad|>",
            "unk_token": None,
            "add_bos_token": False,
            "add_eos_token": False,
            "added_tokens_decoder": added,
            "extra_special_tokens": {
                "fim_prefix": "<|fim_prefix|>", "fim_middle": "<|fim_middle|>",
                "fim_suffix": "<|fim_suffix|>", "fim_pad": "<|fim_pad|>",
                "repo_name": "<|repo_name|>", "file_sep": "<|file_sep|>",
            },
        }, fh, indent=1)
    with open(os.path.join(out_dir, "special_tokens_map.json"), "w") as fh:
        json.dump({"bos_token": "<|endoftext|>", "eos_token": "<|endoftext|>",
                   "pad_token": "<|fim_pad|>"}, fh, indent=1)
    with open(os.path.join(out_dir, "generation_config.json"), "w") as fh:
        json.dump({"eos_token_id": eot, "pad_token_id": pad, "bos_token_id": eot,
                   "do_sample": False, "max_new_tokens": 128,
                   "transformers_version": "5.16.1"}, fh, indent=1)
    return out_dir
