"""
LoRA instruction-tune of Qwen3.5-0.8B-Base on smoltalk2.

    modal run modal_sft.py::probe_base        # ~$0.3, does it even load text-only?
    modal run --detach modal_sft.py::sft      # ~$3.4

Known risk, checked before spending rather than assumed: Qwen3.5-0.8B-Base is
`Qwen3_5ForConditionalGeneration` -- a multimodal wrapper with a 248,320-token vocab
and an MTP head. Text-only loading under transformers 5.16, and whether logits that
wide fit at seq 2048, both need verifying. `probe_base` does exactly that and falls
back to Qwen3-1.7B-Base / Qwen2.5-1.5B if it fails.
"""

import json
import os
import time

import modal

app = modal.App("fimcoder-sft")
vol = modal.Volume.from_name("fimcoder", create_if_missing=True)
VOL = "/vol"

img = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.12.1",
        "transformers==5.16.1",
        "peft==0.19.1",
        "datasets==4.4.1",
        "accelerate==1.12.0",
        "numpy==2.3.4",
        "hf-transfer==0.1.9",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("gpu_config")
)

CANDIDATES = ["Qwen/Qwen3.5-0.8B-Base", "Qwen/Qwen3-1.7B-Base", "Qwen/Qwen2.5-1.5B"]


@app.function(image=img, gpu="L4", volumes={VOL: vol}, cpu=3.0, memory=20480,
              timeout=1800)
def probe_base(names=None) -> dict:
    """Load each candidate text-only, run one forward+backward at seq 2048, report."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    out = []
    for name in names or CANDIDATES:
        r = {"name": name}
        try:
            cfg = AutoConfig.from_pretrained(name)
            r["arch"] = cfg.architectures
            tc = getattr(cfg, "text_config", None) or cfg
            r["vocab"] = getattr(tc, "vocab_size", None)
            r["layers"] = getattr(tc, "num_hidden_layers", None)
            r["hidden"] = getattr(tc, "hidden_size", None)
            tok = AutoTokenizer.from_pretrained(name)
            r["chat_template"] = bool(getattr(tok, "chat_template", None))
            m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).cuda()
            m.gradient_checkpointing_enable()
            m.config.use_cache = False
            r["params_M"] = round(sum(p.numel() for p in m.parameters()) / 1e6, 1)
            ids = torch.randint(0, min(r["vocab"] or 1000, 1000), (1, 2048), device="cuda")
            t0 = time.time()
            loss = m(input_ids=ids, labels=ids).loss
            loss.backward()
            torch.cuda.synchronize()
            r["step_s"] = round(time.time() - t0, 2)
            r["peak_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 1)
            r["loss"] = round(loss.item(), 3)
            r["ok"] = True
            del m
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        except Exception as e:
            r["ok"] = False
            r["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        print(json.dumps(r))
        out.append(r)
        if r.get("ok"):
            break          # first one that works is the one we want
    with open(f"{VOL}/sft_probe.json", "w") as fh:
        json.dump(out, fh, indent=1)
    vol.commit()
    return out


def _render(tok, msgs, max_len: int):
    """Tokenise a chat and mask everything that is not an assistant turn.

    Loss on assistant tokens only: training on the user's words teaches the model to
    imitate users, which is the opposite of instruction following.
    """
    ids: list = []
    lab: list = []
    for m in msgs:
        role = m.get("role")
        content = m.get("content")
        if not isinstance(content, str) or not content:
            continue
        seg = tok.apply_chat_template([{"role": role, "content": content}],
                                      tokenize=True, add_generation_prompt=False)
        if isinstance(seg, dict):
            seg = seg["input_ids"]
        ids.extend(seg)
        lab.extend(seg if role == "assistant" else [-100] * len(seg))
        if len(ids) >= max_len:
            break
    return ids[:max_len], lab[:max_len]


@app.function(image=img, gpu="L4", volumes={VOL: vol}, cpu=3.0, memory=24576,
              timeout=16200, retries=0)
def sft_run(p: dict) -> dict:
    import numpy as np
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import gpu_config as G

    gpu = G.GPUInfo.detect()
    gpu.apply_perf_env()
    name = p["base"]
    out = f"{VOL}/runs/{p['tag']}"
    os.makedirs(out, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).cuda()
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    lcfg = LoraConfig(
        r=p["r"], lora_alpha=p["alpha"], lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()

    # smoltalk2 ships several configs; a bare split= can fail on the config name, so
    # try the obvious ones and fall back to smoltalk rather than dying at minute one.
    ds = None
    for name, cfg in [(p["dataset"], None), (p["dataset"], "SFT"),
                      (p["dataset"], "smoltalk2"), ("HuggingFaceTB/smoltalk", "all")]:
        try:
            ds = (load_dataset(name, cfg, split="train", streaming=True) if cfg
                  else load_dataset(name, split="train", streaming=True))
            print(f"[data] streaming {name}" + (f" [{cfg}]" if cfg else ""))
            break
        except Exception as e:
            print(f"[data] {name} {cfg}: {type(e).__name__}: {str(e)[:120]}")
    if ds is None:
        raise RuntimeError("no usable SFT dataset")
    ds = ds.shuffle(seed=0, buffer_size=4000)

    seq, micro, accum = p["seq"], p["micro_bs"], p["accum"]
    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad],
                            lr=p["lr"], betas=(0.9, 0.95), weight_decay=0.0, fused=True)
    budget_s = p["budget_hours"] * 3600
    t0 = time.time()
    it = iter(ds)
    step = 0
    hist = []
    t_save = time.time()
    total = p["max_steps"]

    def next_batch():
        xs, ys = [], []
        while len(xs) < micro:
            try:
                ex = next(it)
            except StopIteration:
                return None
            msgs = ex.get("messages") or ex.get("conversations")
            if not msgs:
                continue
            i, l = _render(tok, msgs, seq)
            if sum(1 for t in l if t != -100) < 16:
                continue
            pad = seq - len(i)
            xs.append(i + [tok.pad_token_id or 0] * pad)
            ys.append(l + [-100] * pad)
        return (torch.tensor(xs, device="cuda"), torch.tensor(ys, device="cuda"))

    while step < total and time.time() - t0 < budget_s:
        lr = p["lr"] * min(1.0, (step + 1) / max(1, int(total * 0.03)))
        lr *= 1.0 if step < total * 0.85 else max(0.05, 1 - (step - total * 0.85) / (total * 0.15))
        for g in opt.param_groups:
            g["lr"] = lr
        tot, ntok = 0.0, 0
        for _ in range(accum):
            b = next_batch()
            if b is None:
                it = iter(ds)
                continue
            x, y = b
            am = (y != -100).sum()
            loss = model(input_ids=x, attention_mask=(x != (tok.pad_token_id or 0)).long(),
                         labels=y).loss
            (loss / accum).backward()
            tot += loss.item()
            ntok += int(am)
        torch.nn.utils.clip_grad_norm_([q for q in model.parameters() if q.requires_grad], 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if p.get("save_every_s") and time.time() - t_save > p["save_every_s"]:
            # Save the adapter periodically. Without this a run killed at an unknown
            # deadline loses everything, since the merge only happens at the end.
            model.save_pretrained(f"{out}/adapter")
            vol.commit()
            t_save = time.time()
            print(f"[sft] adapter checkpointed at step {step}")
        if step % 10 == 0:
            el = time.time() - t0
            row = {"step": step, "loss": tot / accum, "lr": lr, "tok": ntok,
                   "elapsed_h": el / 3600}
            hist.append(row)
            print(f"[sft] step {step:>5}/{total} loss {tot / accum:.4f} lr {lr:.2e} "
                  f"| {ntok} sup tok | {el / 60:.1f} min")
    print("[sft] merging LoRA into the base weights")
    merged = model.merge_and_unload()
    merged.config.use_cache = True
    merged.save_pretrained(f"{out}/hf", safe_serialization=True)
    tok.save_pretrained(f"{out}/hf")
    res = {"tag": p["tag"], "base": name, "steps": step,
           "elapsed_h": (time.time() - t0) / 3600, "hist": hist}
    with open(f"{out}/result.json", "w") as fh:
        json.dump(res, fh, indent=1)
    vol.commit()
    return json.loads(json.dumps(res))


@app.local_entrypoint()
def probe():
    """Which base model actually loads text-only, and at what memory cost."""
    print(json.dumps(probe_base.remote(), indent=1))


@app.local_entrypoint()
def sft(hours: float = 2.0, base: str = "", tag: str = "qwen-instruct-ours",
        dataset: str = "HuggingFaceTB/smoltalk2", seq: int = 1024, micro_bs: int = 2,
        accum: int = 8, lr: float = 1e-4, r: int = 32, alpha: int = 64,
        max_steps: int = 100000, save_every_s: int = 900):
    """Probe the base model first, then tune whichever candidate actually loads."""
    if not base:
        pr = probe_base.remote()
        ok = [x for x in pr if x.get("ok")]
        if not ok:
            print("no usable base model:\n" + json.dumps(pr, indent=1))
            return
        base = ok[-1]["name"]
        print(f"[base] using {base} ({ok[-1].get('params_M')}M params, "
              f"vocab {ok[-1].get('vocab')}, {ok[-1].get('peak_gb')}GB peak at seq 2048)")
    p = dict(base=base, tag=tag, dataset=dataset, seq=seq, micro_bs=micro_bs,
             accum=accum, lr=lr, r=r, alpha=alpha, budget_hours=hours,
             max_steps=max_steps, save_every_s=save_every_s)
    cap = int(hours * 3600) + 1500
    res = sft_run.with_options(timeout=cap).remote(p)
    with open(f"result_{tag}.json", "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"\n{res['steps']} steps in {res['elapsed_h']:.2f}h; "
          f"final loss {res['hist'][-1]['loss'] if res['hist'] else None}")
    print(f"cost: python3 cost.py add {tag} --gpu H100 --cpu 3 --mem 32 "
          f"--secs {int(res['elapsed_h'] * 3600)}")
