"""
Pretraining for fimcoder-200m on 4x L4.

    modal run modal_train.py::smoke_all             # ~$1, gates everything below
    modal run --detach modal_train.py::pretrain     # the real run

GPU choice was forced, not chosen: this workspace can only launch T4, L4 and A10
without a payment method on file -- L40S, A100 and H100 are all gated. Measured
value per dollar on what IS available: L4 121 TFLOP/s at $0.799/hr = 151 TFLOP/$,
A10 125 at $1.102 = 113, T4 65 at $0.590 = 110 (and T4 is Turing, so no bf16 at
all). L4 wins, and 4 of them cost the same per GPU-hour as one, so `L4:4` buys a
4x shorter wall clock for free -- which matters when the credit expires tomorrow.

Gradient sync is a single flat all-reduce per optimizer step, not DDP. With ~0.5M
tokens per step the sync is under 2% of step time, and doing it explicitly avoids
every DDP/torch.compile interaction and the memory blow-up of letting
`LlamaForCausalLM.forward` materialise full logits.

Cost control, by construction:
  * every GPU function carries an explicit `timeout`, set from the requested budget;
  * the loop stops on a wall-clock budget, not on "until it looks done";
  * total step count is fixed from MEASURED throughput after warmup, and the WSD
    decay leg is scheduled against that, so the run always ends annealed.

Context length: the bulk of training runs at seq 2048, and the final phase (the
WSD decay leg) runs at seq 4096, so the delivered model genuinely handles a
4096-token editor window with several files of repo context. Attention is 27% of
FLOPs/token at 2048 and 43% at 4096, so paying for the longer window only during
annealing buys the capability at a fraction of what training at 4096 throughout
would cost.
"""

import json
import os
import time

import modal

app = modal.App("fimcoder-train")
vol = modal.Volume.from_name("fimcoder", create_if_missing=True)
VOL = "/vol"

img = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.12.1",
        "transformers==5.16.1",   # pins tokenizers/hub/safetensors itself
        "numpy==2.3.4",
        "hf-transfer==0.1.9",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "TOKENIZERS_PARALLELISM": "false",
        # Must be set before the CUDA allocator initialises, so it goes here rather
        # than in apply_perf_env(), which runs after the device has been touched.
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        # 4 ranks x 4 inductor workers on 4 cores thrashes; one worker each is faster.
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
    })
    .add_local_python_source("trainlib", "gpu_config", "fimlib")
)


def _tok_meta():
    """Vocab size and pad id, read from the trained tokenizer on the volume."""
    from tokenizers import Tokenizer

    t = Tokenizer.from_file(f"{VOL}/tokenizer/tokenizer.json")
    with open(f"{VOL}/tokenizer/specials.json") as fh:
        sp = json.load(fh)
    return t.get_vocab_size(), sp["<|fim_pad|>"], sp


@app.function(image=img, volumes={VOL: vol}, cpu=4.0, memory=8192, timeout=900)
def checkapi():
    """Verify the transformers API on CPU before any GPU second is billed."""
    import torch
    import transformers

    import trainlib as T

    vocab, pad, sp = _tok_meta()
    print(f"torch {torch.__version__} | transformers {transformers.__version__}")
    print(f"vocab {vocab} pad {pad} specials {sp}")

    from transformers import LlamaConfig
    import inspect
    sig = inspect.signature(LlamaConfig.__init__)
    print("LlamaConfig params:", ", ".join(list(sig.parameters)[1:]))
    d = LlamaConfig().to_dict()
    print("rope-ish keys:", {k: v for k, v in d.items() if "rope" in k or "tie" in k})
    cfg = T.model_config(vocab, seq=4096)
    print("config:", cfg.hidden_size, cfg.num_hidden_layers, cfg.num_key_value_heads,
          cfg.intermediate_size, "tied:", getattr(cfg, "tie_word_embeddings", None))
    m = T.build_model(cfg, device="cpu", dtype=torch.float32)
    pc = T.param_counts(m)
    print(f"params total {pc['total'] / 1e6:.1f}M  embed {pc['embed'] / 1e6:.1f}M  "
          f"non-embed {pc['non_embed'] / 1e6:.1f}M")
    print("tied weights share storage:",
          m.lm_head.weight.data_ptr() == m.get_input_embeddings().weight.data_ptr())

    ids = torch.randint(0, vocab, (2, 64))
    ids[0, -8:] = pad
    body = m.model
    out = body(input_ids=ids[:, :-1])
    print("body output type:", type(out).__name__, "hidden:",
          tuple(out.last_hidden_state.shape))
    loss, n = T.forward_loss(m, body, ids, pad, chunk=512)
    print(f"chunked_ce -> loss/token {loss.item() / n:.4f} over {n} tokens "
          f"(expect ~{__import__('math').log(vocab):.2f} at init)")
    loss.backward()
    g = sum(1 for p in m.parameters() if p.grad is not None)
    print(f"backward ok, {g} tensors have grads")
    for impl in ("sdpa", "flash_attention_2", "eager"):
        try:
            cfg2 = T.model_config(vocab, seq=256)
            cfg2._attn_implementation = impl
            T.build_model(cfg2, device="cpu", dtype=torch.float32)
            print(f"  attn impl available: {impl}")
        except Exception as e:
            print(f"  attn impl {impl}: {type(e).__name__}")
    return {"vocab": vocab, "pad": pad, "params": pc,
            "transformers": str(transformers.__version__), "torch": str(torch.__version__)}


# ------------------------------------------------------------------ training

def _stage_shards(seq: int, dst="/data"):
    """Copy shards from the network volume to container disk once, up front.

    Training then reads local pages instead of hitting the volume on every batch.
    """
    import shutil

    src = f"{VOL}/data/s{seq}"
    os.makedirs(f"{dst}/s{seq}", exist_ok=True)
    t0 = time.time()
    n = 0
    for f in sorted(os.listdir(src)):
        if f.endswith(".bin"):
            shutil.copyfile(f"{src}/{f}", f"{dst}/s{seq}/{f}")
            n += os.path.getsize(f"{src}/{f}")
    print(f"[data] staged {n / 1e9:.2f} GB of seq-{seq} shards in {time.time() - t0:.0f}s")
    return f"{dst}/s{seq}/train_*.bin"


def _save(path, model, opts, step, extra, weights_only=False):
    import torch

    tmp = path + ".tmp"
    obj = {"model": model.state_dict(), "step": step, **extra}
    if not weights_only:
        obj["opts"] = [o.state_dict() for o in opts]
    torch.save(obj, tmp)
    os.replace(tmp, path)
    return os.path.getsize(path)


def _val_loss(model, body, data, pad, bs, world=1, iters=12):
    """Val loss over `iters` batches per rank, reduced across ranks."""
    import torch
    import torch.distributed as dist

    import trainlib as T

    model.eval()
    tot, ntok = 0.0, 0
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        for _ in range(iters):
            ids = data.batch(bs)
            l, n = T.forward_loss(model, body, ids, pad)
            tot += l.item()
            ntok += n
    model.train()
    if world > 1:
        t = torch.tensor([tot, float(ntok)], device="cuda")
        dist.all_reduce(t)
        tot, ntok = t[0].item(), t[1].item()
    return tot / max(ntok, 1)




class Sync:
    """One flat all-reduce of every gradient per optimizer step.

    Buffer is allocated once. At ~0.5M tokens per step the transfer is <2% of step
    time, which is the whole reason this can replace DDP: no autograd hooks, no
    bucket tuning, and no interaction with torch.compile.
    """

    def __init__(self, params, world: int):
        import torch

        self.params = [p for p in params if p.requires_grad]
        self.world = world
        self.n = sum(p.numel() for p in self.params)
        self.buf = (torch.zeros(self.n, device="cuda", dtype=torch.float32)
                    if world > 1 else None)

    def __call__(self):
        if self.world <= 1:
            return
        import torch.distributed as dist

        off = 0
        for p in self.params:
            k = p.numel()
            if p.grad is not None:
                self.buf[off : off + k].copy_(p.grad.reshape(-1))
            else:
                self.buf[off : off + k].zero_()
            off += k
        dist.all_reduce(self.buf)
        self.buf.mul_(1.0 / self.world)
        off = 0
        for p in self.params:
            k = p.numel()
            if p.grad is None:
                p.grad = self.buf[off : off + k].view_as(p).clone()
            else:
                p.grad.copy_(self.buf[off : off + k].view_as(p))
            off += k


def _worker(rank: int, world: int, p: dict):
    """One training process. rank 0 does all logging, checkpointing and saving."""
    import numpy as np
    import torch
    import torch.distributed as dist

    import gpu_config as G
    import trainlib as T

    torch.cuda.set_device(rank)
    if world > 1:
        dist.init_process_group("nccl", rank=rank, world_size=world)
    lead = rank == 0
    tag = p["tag"]
    out = f"{VOL}/runs/{tag}"
    if lead:
        os.makedirs(out, exist_ok=True)

    gpu = G.GPUInfo.detect() if lead else _quiet_gpu(rank)
    gpu.apply_perf_env()
    vocab, pad, _ = _tok_meta()
    seq1, seq2 = p["seq1"], p["seq2"]

    if lead:
        pat1 = _stage_shards(seq1)
    if world > 1:
        dist.barrier()
    pat1 = f"/data/s{seq1}/train_*.bin"
    data = T.ShardData(pat1, seq1, seed=p.get("seed", 0) * 97 + rank)
    val = (T.ShardData(f"{VOL}/data/s2048/val_*.bin", 2048, seed=99 + rank)
           if p.get("val", True) else None)
    if lead:
        print(f"[data] {data}")

    cfg = T.model_config(vocab, seq=max(seq1, seq2), d_model=p["d_model"],
                         n_layers=p["n_layers"], n_heads=p["n_heads"],
                         n_kv=p["n_kv"], d_ff=p["d_ff"])
    model = T.build_model(cfg, device=f"cuda:{rank}", dtype=torch.float32)
    pcnt = T.param_counts(model)
    if lead:
        print(f"[model] {pcnt['total'] / 1e6:.1f}M total, "
              f"{pcnt['non_embed'] / 1e6:.1f}M non-embed, "
              f"{6 * pcnt['non_embed'] / 1e9:.3f} + "
              f"{12 * p['n_layers'] * p['d_model'] * seq1 / 1e9:.3f} GFLOP/token")
    body = torch.compile(model.model, dynamic=False) if p.get("compile", True) else model.model
    opts, bases = T.make_optimizers(model, p["opt"], p["lr"], p["muon_lr"], p["wd"])
    sync = Sync(model.parameters(), world)

    def try_step(mbs, seq):
        ids = torch.randint(0, vocab, (mbs, seq), device="cuda", dtype=torch.long)
        with torch.autocast("cuda", torch.bfloat16):
            loss, n = T.forward_loss(model, body, ids, pad)
        (loss / max(n, 1)).backward()
        for o in opts:
            o.zero_grad(set_to_none=True)

    if p.get("micro_bs"):
        micro = p["micro_bs"]
    else:
        micro = (G.autotune_micro_batch(lambda b: try_step(b, seq1), seq1, gpu,
                                        start=4, max_bs=p.get("max_micro", 32),
                                        n_params=pcnt["total"])
                 if lead else 0)
        if world > 1:                       # every rank must agree on the batch shape
            t = torch.tensor([micro], device="cuda")
            dist.broadcast(t, src=0)
            micro = int(t.item())
    per_step = micro * seq1 * accum_for(p, micro, seq1, world) * world
    accum = accum_for(p, micro, seq1, world)
    if lead:
        print(f"[batch] {world} x micro {micro} x seq {seq1} x accum {accum} "
              f"= {per_step:,} tokens/step")
    return _loop(p, rank, world, lead, gpu, model, body, opts, bases, sync, data, val,
                 pcnt, cfg, micro, accum, seq1, seq2, pad, out, tag)


def accum_for(p, micro, seq, world):
    return max(1, round(p["tokens_per_step"] / max(micro * seq * world, 1)))


def _quiet_gpu(rank):
    import gpu_config as G
    import io
    import contextlib

    with contextlib.redirect_stdout(io.StringIO()):
        return G.GPUInfo.detect()


def _loop(p, rank, world, lead, gpu, model, body, opts, bases, sync, data, val,
          pcnt, cfg, micro, accum, seq1, seq2, pad, out, tag):
    import dataclasses
    import numpy as np
    import torch
    import torch.distributed as dist

    import gpu_config as G
    import trainlib as T

    agg = dataclasses.replace(gpu, peak_bf16_flops=gpu.peak_bf16_flops * world)
    tr = G.MFUTracker(agg, pcnt["non_embed"], seq1, p["n_layers"], p["d_model"],
                      log_every=p["log_every"])
    hist, t_start, t_ckpt = [], time.time(), time.time()
    budget_s = p["budget_hours"] * 3600
    total = p.get("max_steps") or 10**9
    fixed = bool(p.get("max_steps"))
    step, phase, seq, cum, mark = 0, 1, seq1, 0, None
    model.train()

    ck = p.get("resume")
    if ck and os.path.exists(ck):
        st = torch.load(ck, map_location=f"cuda:{rank}", weights_only=False)
        model.load_state_dict(st["model"])
        for o, sd in zip(opts, st.get("opts", [])):
            o.load_state_dict(sd)
        step, fixed = st["step"], True
        # Keep the ORIGINAL WSD horizon on a crash-resume, so the decay leg still
        # lands where it was scheduled -- unless the caller explicitly asks for a
        # different step count, which is how a run gets extended.
        if not p.get("max_steps"):
            total = st.get("total", total)
        hist = st.get("hist", [])
        if lead:
            print(f"[resume] {ck} at step {step} of {total}")

    while step < total and time.time() - t_start < budget_s:
        if not fixed and step == max(6, p["measure_at"] // 3):
            mark = (time.time(), cum)          # after torch.compile has settled
        if not fixed and step == p["measure_at"] and mark:
            rate = (cum - mark[1]) / max(time.time() - mark[0], 1e-9)
            left = budget_s - (time.time() - t_start)
            f2 = min(1.0, max(0.0, 1.0 - p["phase2_frac"]))
            slow = 1.0 + f2 * (0.35 if seq2 > seq1 else 0.0)
            ps = micro * seq1 * accum * world
            total = step + max(50, int(rate * left * 0.95 / slow / ps))
            fixed = True
            if lead:
                print(f"[plan] {rate / 1e3:.0f}k tok/s -> total_steps={total}, "
                      f"~{total * ps / 1e9:.2f}B tokens, phase2 at "
                      f"{int(total * p['phase2_frac'])}, decay from "
                      f"{int(total * (1 - p['decay_frac']))}")

        if phase == 1 and fixed and seq2 != seq1 and step >= int(total * p["phase2_frac"]):
            if lead:
                print(f"[phase] step {step}: switching to seq {seq2} for the decay leg")
                _stage_shards(seq2)
            if world > 1:
                dist.barrier()
            data = T.ShardData(f"/data/s{seq2}/train_*.bin", seq2,
                               seed=p.get("seed", 0) * 97 + rank + 7)
            micro = max(1, micro // (seq2 // seq1))
            accum = accum_for(p, micro, seq2, world)
            seq, phase = seq2, 2
            tr = G.MFUTracker(agg, pcnt["non_embed"], seq2, p["n_layers"], p["d_model"],
                              log_every=p["log_every"])
            if lead:
                print(f"[data] {data}\n[batch] {world} x micro {micro} x seq {seq2} "
                      f"x accum {accum} = {micro * seq2 * accum * world:,} tokens/step")

        # Before the horizon is measured, `total` is a placeholder of 1e9, which makes
        # warmup_frac resolve to a 25M-step warmup and the LR effectively zero. Use a
        # fixed short warmup until `total` is real. (Observed: 24 steps at lr ~1e-9.)
        mult = (T.wsd_lr(step, total, p["warmup_frac"], p["decay_frac"]) if fixed
                else min(1.0, (step + 1) / max(1, p["measure_at"] * 2)))
        T.set_lr(opts, bases, mult)
        norm = micro * (seq - 1) * accum
        lsum, ntok = 0.0, 0
        for _ in range(accum):
            ids = data.batch(micro)
            with torch.autocast("cuda", torch.bfloat16):
                loss, n = T.forward_loss(model, body, ids, pad)
            (loss / norm).backward()
            lsum += loss.detach()
            ntok += n
        sync()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), p["clip"])
        for o in opts:
            o.step()
            o.zero_grad(set_to_none=True)
        step += 1

        stat = torch.tensor([float(lsum), float(ntok)], device="cuda")
        if world > 1:
            dist.all_reduce(stat)
        gl, gn_tok = stat[0].item(), int(stat[1].item())
        cum += gn_tok
        tl = gl / max(gn_tok, 1)
        s = tr.step(gn_tok) if lead else None
        if s:
            s.update({"loss": tl, "lr": bases[0] * mult, "gnorm": float(gn),
                      "step_i": step, "seq": seq, "epochs": data.epochs_served()})
            hist.append(s)
            print(f"      loss {tl:.4f} | lr {bases[0] * mult:.2e} | gnorm {float(gn):.2f} "
                  f"| ep {data.epochs_served():.2f}")
        if not np.isfinite(tl):
            if lead:
                print("[abort] non-finite loss")
            break

        if val and step % p["val_every"] == 0:
            vl = _val_loss(model, body, val, pad, max(1, micro // 2), world)
            if lead:
                print(f"[val ] step {step}: {vl:.4f} (ppl {np.exp(vl):.2f})")
                hist.append({"step_i": step, "val_loss": vl})
        if lead and time.time() - t_ckpt > p["ckpt_every_s"]:
            sz = _save(f"{out}/ckpt_{step % 2}.pt", model, opts, step,
                       {"cfg": cfg.to_dict(), "total": total, "hist": hist})
            vol.commit()
            t_ckpt = time.time()
            print(f"[ckpt] step {step} -> ckpt_{step % 2}.pt ({sz / 1e9:.2f} GB)")

    res = {"tag": tag, "steps": step, "total_planned": total, "seq_final": seq,
           "tokens": cum, "elapsed_h": (time.time() - t_start) / 3600,
           "best_mfu": tr.best_mfu, "params": pcnt, "gpu": f"{world}x {gpu.name}",
           "world": world, "micro_bs": micro, "accum": accum,
           "final_loss": next((h["loss"] for h in reversed(hist) if "loss" in h), None),
           "hist": hist}
    if val:
        res["val_loss"] = _val_loss(model, body, val, pad, max(1, micro // 2), world, 24)
    if lead:
        if val:
            print(f"[val ] final {res['val_loss']:.4f} (ppl {np.exp(res['val_loss']):.2f})")
        if p.get("save_hf", True):
            hf = T.export_hf(model.to(torch.bfloat16), f"{VOL}/tokenizer", f"{out}/hf",
                             seq=max(seq1, seq2))
            print(f"[save] {hf}: " + ", ".join(sorted(os.listdir(hf))))
        with open(f"{out}/result.json", "w") as fh:
            json.dump(res, fh, indent=1)
        vol.commit()
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    return res if lead else None


def _run(p: dict) -> dict:
    """Rank 0 runs in THIS process so `vol.commit()` uses the real Modal client;
    ranks 1..N-1 are spawned children. Cleaner than mp.spawn for our purposes: the
    result comes back as a return value and mid-run checkpoints commit normally."""
    world = int(p.get("world", 1))
    if world <= 1:
        return _worker(0, 1, p)

    import torch.multiprocessing as mp

    # A fresh port per call. Re-initialising NCCL on the SAME port inside one
    # container fails with ncclRemoteError once the previous group's socket is in
    # TIME_WAIT -- which is exactly what a multi-arm smoke test does.
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(29500 + (int(time.time()) % 200))
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    ctx = mp.get_context("spawn")
    kids = [ctx.Process(target=_worker, args=(r, world, p), daemon=False)
            for r in range(1, world)]
    for k in kids:
        k.start()
    try:
        res = _worker(0, world, p)
    finally:
        for k in kids:
            k.join(timeout=600)
            if k.is_alive():
                k.terminate()
    return res


BASE = dict(
    # ~214M params. Compute-optimal for what 4x L4 buys in ~6h is N_non_embed ~ 170M
    # (Chinchilla C = 120 N^2); 1024/16 lands there with power-of-two shapes, which
    # matters more on an L4 than shaving the last few percent off the shape.
    d_model=1024, n_layers=16, n_heads=16, n_kv=4, d_ff=2816,
    seq1=2048, seq2=4096, tokens_per_step=524_288,
    opt="adamw", lr=2.0e-3, muon_lr=0.025, wd=0.1, clip=1.0,
    warmup_frac=0.025, decay_frac=0.15, phase2_frac=0.75,
    log_every=10, val_every=250, ckpt_every_s=1200, measure_at=24,
    budget_hours=5.8, compile=True, val=True, save_hf=True, world=4,
)


# modal 1.2.6 has no Function.with_options, so GPU / cpu / memory / timeout are
# fixed in the decorator. That is actually the safer shape for a budget ceiling:
# the timeout is a constant in the source, not something computed at call time.
# Each entry below is (gpu, cpu, GiB, timeout_s); the timeout IS the hard cost cap.
def _cfg_cost(gpu: str, cpu: float, gib: float, secs: float) -> float:
    import cost as C

    kind, _, cnt = gpu.partition(":")
    return secs * (C.GPU[kind.replace("A100-40GB", "A100-40").replace("A100-80GB", "A100-80")]
                   * int(cnt or 1) + cpu * C.CPU_CORE_S + gib * C.MEM_GIB_S)


@app.function(image=img, gpu="L4:4", volumes={VOL: vol}, cpu=4.0, memory=16384,
              timeout=17700, retries=0)
def train_l4x4(overrides: dict) -> dict:
    """4x L4, hard-capped at 4h55m of container life (~$17.3 absolute worst case)."""
    p = dict(BASE)
    p.update(overrides)
    print("[cfg] " + json.dumps({k: v for k, v in p.items() if k != "resume"}))
    return _run(p)


@app.function(image=img, gpu="L4", volumes={VOL: vol}, cpu=2.0, memory=12288,
              timeout=36000, retries=0)
def train_l4x1(overrides: dict) -> dict:
    """1x L4, hard-capped at 10h of container life (~$9.9 absolute worst case)."""
    p = dict(BASE)
    p.update(overrides)
    p["world"] = 1
    print("[cfg] " + json.dumps({k: v for k, v in p.items() if k != "resume"}))
    return _run(p)


TRAINERS = {
    "L4:4": (train_l4x4, "L4:4", 4.0, 16.0, 17700),
    "L4": (train_l4x1, "L4", 2.0, 12.0, 36000),
}


@app.function(image=img, gpu="L4", volumes={VOL: vol}, cpu=2.0, memory=12288,
              timeout=2400, retries=0)
def val_run(cfgs: list) -> list:
    return smoke_body(cfgs)


@app.function(image=img, gpu="L4:4", volumes={VOL: vol}, cpu=4.0, memory=16384,
              timeout=5400, retries=0)
def smoke(cfgs: list) -> list:
    """A/B a few configurations in one container: same data, same steps, real loop.

    One container amortises the cold start, the image pull and the volume staging
    across every arm, which is most of what a short run costs.
    """
    return smoke_body(cfgs)


def smoke_body(cfgs: list) -> list:
    out = []
    for c in cfgs:
        p = dict(BASE)
        p.update(c)
        print(f"\n{'=' * 72}\n=== {p['tag']} ===\n{'=' * 72}")
        try:
            out.append(_run(p))
        except Exception:
            import traceback

            traceback.print_exc()
            out.append({"tag": p["tag"], "error": traceback.format_exc()[-800:]})
    return out


@app.local_entrypoint()
def smoke_all(steps: int = 60, world: int = 4):
    """Gate before real spend: optimiser and shape A/B, plus a seq-4096 check.

    Cheap on purpose (~$1 of a $30 budget) and it exercises the exact loop, data,
    packing and checkpoint path the real run uses. Never launch a GPU into unsmoked
    code; this protects the ~$21 run that follows.
    """
    common = dict(max_steps=steps, val_every=max(20, steps // 2), ckpt_every_s=10**9,
                  save_hf=False, budget_hours=0.6, phase2_frac=2.0, world=world,
                  tokens_per_step=262_144, measure_at=10**9)
    cfgs = [
        dict(common, tag="A-adamw-2e3", opt="adamw", lr=2.0e-3),
        dict(common, tag="B-adamw-4e3", opt="adamw", lr=4.0e-3),
        dict(common, tag="C-muon", opt="muon", lr=2.0e-3, muon_lr=0.025),
        dict(common, tag="D-768x24", opt="adamw", lr=2.0e-3, max_steps=max(20, steps // 2),
             d_model=768, n_layers=24, n_heads=12, n_kv=4, d_ff=2048),
        dict(common, tag="E-seq4096", opt="adamw", lr=2.0e-3, seq1=4096, seq2=4096,
             max_steps=20, val_every=10**9),
    ]
    res = smoke.remote(cfgs)
    with open("smoke_result.json", "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"\n{'tag':<16}{'steps':>6}{'params':>9}{'tok/s':>9}{'MFU':>7}"
          f"{'loss':>8}{'val':>8}{'min':>6}")
    for r in res:
        if "error" in r:
            print(f"{r['tag']:<16}  ERROR")
            print("    " + r["error"].replace("\n", "\n    ")[-500:])
            continue
        tps = r["tokens"] / max(r["elapsed_h"] * 3600, 1e-9)
        print(f"{r['tag']:<16}{r['steps']:>6}{r['params']['total'] / 1e6:>8.0f}M"
              f"{tps / 1e3:>8.0f}k{r['best_mfu'] * 100:>6.1f}%"
              f"{(r.get('final_loss') or 0):>8.3f}{(r.get('val_loss') or 0):>8.3f}"
              f"{r['elapsed_h'] * 60:>6.1f}")
    print("\nwrote smoke_result.json")


@app.local_entrypoint()
def pretrain(hours: float = 4.5, gpu: str = "L4:4", opt: str = "adamw",
             lr: float = 2.4e-3, muon_lr: float = 0.025, tag: str = "fimcoder-113m",
             micro_bs: int = 0, resume: str = "", phase2_frac: float = 0.75,
             tokens_per_step: int = 524288, d_model: int = 768, n_layers: int = 14,
             n_heads: int = 12, n_kv: int = 4, d_ff: int = 2048,
             budget_usd: float = 0.0, go: bool = False):
    """The real run. Launch with `modal run --detach` so a dropped link is harmless.

    Cost is bounded twice over: the container timeout is a constant in the source
    (see TRAINERS), and the loop stops on its own wall-clock budget well inside it.
    This refuses to launch if the container ceiling would break the remaining budget.
    """
    import cost as C

    fn, gname, cpu, gib, cap = TRAINERS[gpu]
    world = int(gname.partition(":")[2] or 1)
    rate = _cfg_cost(gname, cpu, gib, 1.0)
    spent = sum(x["cost"] for x in C.load())
    left = C.BUDGET - spent
    worst = cap * rate
    expect = hours * 3600 * rate + 240 * rate          # + container start and staging

    print(f"[cost] {gname} + {cpu} cores + {gib:.0f}GiB = ${rate * 3600:.2f}/hr")
    print(f"[cost] expect ${expect:.2f} for {hours:.2f}h | container ceiling "
          f"${worst:.2f} at {cap}s | spent ${spent:.2f} | left ${left:.2f}")
    if hours * 3600 + 900 > cap:
        print(f"[cost] REFUSING: {hours:.2f}h leaves under 15 min of slack inside the "
              f"{cap}s container timeout. Use --hours {(cap - 900) / 3600:.2f} or less.")
        return
    reserve = budget_usd or (left - 2.0)
    if worst > reserve:
        print(f"[cost] REFUSING: ceiling ${worst:.2f} > ${reserve:.2f} allowed. "
              f"Raise --budget-usd only if you mean it.")
        return
    if not go:
        print("\n[dry-run] add --go to actually launch.")
        return

    o = dict(tag=tag, budget_hours=hours, opt=opt, lr=lr, muon_lr=muon_lr,
             phase2_frac=phase2_frac, tokens_per_step=tokens_per_step, world=world,
             d_model=d_model, n_layers=n_layers, n_heads=n_heads, n_kv=n_kv, d_ff=d_ff)
    if micro_bs:
        o["micro_bs"] = micro_bs
    if resume:
        o["resume"] = resume
    r = fn.remote(o)
    with open(f"result_{tag}.json", "w") as fh:
        json.dump(r, fh, indent=1)
    print(f"\n{r['steps']} steps, {r['tokens'] / 1e9:.2f}B tokens, {r['elapsed_h']:.2f}h "
          f"on {r['gpu']}, best MFU {r['best_mfu'] * 100:.1f}%, "
          f"loss {r.get('final_loss')}, val {r.get('val_loss')}")
    C.add(tag, gpu=gname.partition(":")[0], gpus=world, cpu=cpu, mem=gib,
          secs=int(r["elapsed_h"] * 3600) + 240, overhead=1.0)
    C.show()


@app.local_entrypoint()
def validate(resume_only: bool = False):
    """Exercise the three things a long run can waste itself on, for ~$0.25.

    Arm A of the earlier smoke already proved data, packing, autotune, the chunked
    loss and the step loop. What it did NOT touch is exactly what would silently
    ruin a 9-hour run: the seq-2048 -> seq-4096 phase switch, writing a checkpoint,
    resuming from one, and the HF export. So test those on a tiny model, alone.
    """
    tiny = dict(d_model=256, n_layers=4, n_heads=4, n_kv=2, d_ff=704, world=1,
                tokens_per_step=32_768, log_every=4, val_every=8, measure_at=10**9,
                budget_hours=0.25, micro_bs=4, compile=False)
    a = dict(tiny, tag="V-phase-ckpt-export", max_steps=16, phase2_frac=0.5,
             ckpt_every_s=30, save_hf=True)
    b = dict(tiny, tag="V-resume", max_steps=24, phase2_frac=2.0, ckpt_every_s=10**9,
             save_hf=False, resume=f"{VOL}/runs/V-phase-ckpt-export/ckpt_0.pt")
    res = val_run.remote([b] if resume_only else [a, b])
    with open("validate_result.json", "w") as fh:
        json.dump(res, fh, indent=1)
    ok = True
    for r in res:
        if "error" in r:
            ok = False
            print(f"FAIL {r['tag']}\n{r['error']}")
        else:
            print(f"pass {r['tag']:<22} steps {r['steps']:>3} seq_final {r['seq_final']} "
                  f"tokens {r['tokens']:,} val {r.get('val_loss')}")
    a_r = next((r for r in res if r.get("tag") == "V-phase-ckpt-export"), {})
    b_r = next((r for r in res if r.get("tag") == "V-resume"), {})
    if a_r and a_r.get("seq_final") != 4096:
        ok = False
        print("FAIL phase switch did not reach seq 4096")
    if b_r.get("steps", 0) <= 16:
        ok = False
        print(f"FAIL resume stopped at step {b_r.get('steps')}, expected 24")
    print("\nVALIDATE " + ("OK - safe to launch the long run" if ok else "FAILED"))
