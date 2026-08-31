"""
Evaluation for fimcoder-300m.

    modal run modal_eval.py::evaluate --tag fimcoder-300m

Four things get measured, in descending order of how much they tell us:

1. **FIM exact-match and edit similarity**, single-file AND multi-file, per language.
   This is the number that decides whether the model is usable in an editor.
2. **Bits per byte** on held-out fresh code, ours vs Qwen2.5-Coder-0.5B. Perplexity is
   not comparable across tokenizers; bits-per-byte is, and it is the honest way to
   claim anything about a model with its own vocabulary.
3. **Bits per byte on `allenai/code_fresh_0825_1225`** (Aug-Dec 2025 repos, after
   Qwen2.5-Coder's cutoff) versus the same metric on pre-2023 code. If training on
   fresh-only data buys anything, it shows up as a smaller gap here than on old code.
4. **Latency at 128 new tokens**, because an autocomplete model that takes a second
   is not an autocomplete model.
"""

import json
import os
import time

import modal

app = modal.App("fimcoder-eval")
vol = modal.Volume.from_name("fimcoder", create_if_missing=True)
VOL = "/vol"

img = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.12.1",
        "transformers==5.16.1",
        "numpy==2.3.4",
        "pyarrow==21.0.0",
        "huggingface-hub",
        "hf-transfer==0.1.9",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("trainlib", "gpu_config", "fimlib")
)


def _edit_sim(a: str, b: str) -> float:
    """Normalised Levenshtein similarity, iterative and allocation-light."""
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return 1.0 - prev[-1] / max(len(a), len(b))


def _load_ours(tag: str, device="cuda"):
    import torch
    from tokenizers import Tokenizer
    from transformers import LlamaForCausalLM

    import fimlib as F

    d = f"{VOL}/runs/{tag}/hf"
    m = LlamaForCausalLM.from_pretrained(d, dtype=torch.bfloat16).to(device).eval()
    m.config.use_cache = True
    return m, Tokenizer.from_file(f"{d}/tokenizer.json"), F.Specials(d)


def _greedy(model, prompts_ids, eot, pad, max_new=64, device="cuda", bs=16):
    """Left-padded batched greedy decode. Returns one id list per prompt."""
    import torch

    out = []
    for i in range(0, len(prompts_ids), bs):
        chunk = prompts_ids[i : i + bs]
        L = max(len(c) for c in chunk)
        ids = torch.full((len(chunk), L), pad, dtype=torch.long)
        att = torch.zeros((len(chunk), L), dtype=torch.long)
        for j, c in enumerate(chunk):
            ids[j, L - len(c) :] = torch.tensor(c)
            att[j, L - len(c) :] = 1
        ids, att = ids.to(device), att.to(device)
        with torch.no_grad():
            gen = model.generate(input_ids=ids, attention_mask=att, do_sample=False,
                                 max_new_tokens=max_new, eos_token_id=eot,
                                 pad_token_id=pad, use_cache=True)
        for j in range(len(chunk)):
            new = gen[j, L:].tolist()
            if eot in new:
                new = new[: new.index(eot)]
            out.append(new)
    return out


def _fim_prompt(tok, sp, row):
    """PSM prompt in exactly the shape minuet-ai.nvim posts."""
    ctx = tok.encode(row["context"], add_special_tokens=False).ids if row["context"] else []
    pre = tok.encode(row["prefix"], add_special_tokens=False).ids
    suf = tok.encode(row["suffix"], add_special_tokens=False).ids
    return [*ctx, sp.pre, *pre, sp.suf, *suf, sp.mid]


def _bits_per_byte(model, encode, docs, device="cuda", seq=2048, stride=1024):
    """Sum -log2 p over a document divided by its UTF-8 byte length.

    Tokenizer-independent, so a 32k-vocab model and a 152k-vocab model can be
    compared honestly. Long documents are scored with a sliding window and only the
    newly-revealed positions contribute, so no token is scored without context.
    """
    import torch

    nats = 0.0
    nbytes = 0
    for text in docs:
        ids = encode(text)
        if len(ids) < 16:
            continue
        nbytes += len(text.encode("utf-8"))
        prev = 0
        for start in range(0, len(ids), stride):
            window = ids[start : start + seq]
            if len(window) < 2:
                break
            x = torch.tensor([window], device=device)
            with torch.no_grad(), torch.autocast(device, torch.bfloat16):
                logits = model(input_ids=x).logits.float()
            lp = torch.log_softmax(logits[0, :-1], -1)
            tgt = x[0, 1:]
            tok_lp = lp.gather(-1, tgt[:, None])[:, 0]
            first = max(0, prev - start - 1)      # skip positions already scored
            nats -= tok_lp[first:].sum().item()
            prev = start + len(window)
            if start + seq >= len(ids):
                break
    return nats / max(nbytes, 1) / 0.6931471805599453, nbytes


UUID = "990b4288-3824-41ac-94a0-b6fd6fa23ffe"
PART = f"datasets/HuggingFaceCode/stack-v3-train/data/part-{{:05d}}-{UUID}-c000.snappy.parquet"
STALE_MAX = 1_672_531_200          # 2023-01-01 UTC: comfortably before the fresh window


def _sample_corpus(part_idx: int, ts_min: int, ts_max: int, want_mb: float = 6.0):
    """Pull a language-balanced text sample from one row group, in a date window."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    import fimlib as F

    mix = F.load_lang_mix("/root/lang_mix.json")
    langs = F.lang_array(set(mix))
    fs = HfFileSystem()
    pf = pq.ParquetFile(fs.open(PART.format(part_idx), "rb"))
    tbl = pf.read_row_group(0, columns=F.ARROW_COLS)
    per_lang = {l: want_mb * 1e6 * sh for l, sh in mix.items()}
    got: dict[str, float] = {}
    docs = []
    for _repo, files in F.iter_repos(tbl, langs, pc, ts_min, ts_max):
        for f in files:
            lg = f["lang"]
            if got.get(lg, 0) >= max(per_lang.get(lg, 0), 25_000):
                continue
            if len(f["content"]) < 800:
                continue
            docs.append((lg, f["content"][:16_000]))
            got[lg] = got.get(lg, 0) + len(f["content"])
    # Repo iteration order clusters files by repository, so a plain [:n] slice would
    # score one project rather than the language mix. Shuffle before the caller truncates.
    __import__("random").Random(4242).shuffle(docs)
    return docs


@app.function(image=img.add_local_file("lang_mix.json", "/root/lang_mix.json"),
              gpu="L4", volumes={VOL: vol}, cpu=3.0, memory=20480, timeout=4200)
def evaluate(tag: str = "fimcoder-113m", max_fim: int = 1400, max_new: int = 64,
             baseline: str = "Qwen/Qwen2.5-Coder-0.5B", bpb_docs: int = 220) -> dict:
    import numpy as np
    import torch

    import gpu_config as G

    gpu = G.GPUInfo.detect()
    gpu.apply_perf_env()
    model, tok, sp = _load_ours(tag)
    enc = lambda s: tok.encode(s, add_special_tokens=False).ids
    res: dict = {"tag": tag, "gpu": gpu.name}

    # ---------------------------------------------------------- 1. FIM accuracy
    rows = [json.loads(l) for l in open(f"{VOL}/data/fimeval.jsonl")]
    rows = [r for r in rows if r["n_ids"] < 1900][:max_fim]
    prompts = [_fim_prompt(tok, sp, r) for r in rows]
    t0 = time.time()
    gens = _greedy(model, prompts, sp.eot, sp.pad, max_new=max_new, bs=24)
    fim_secs = time.time() - t0

    per: dict = {}
    for r, g in zip(rows, gens):
        got = tok.decode(g, skip_special_tokens=False)
        want = r["middle"]
        # Ground truth is capped to what the model was asked to produce, so a long
        # middle is not scored as a miss just for being longer than max_new tokens.
        want_cap = tok.decode(enc(want)[:max_new], skip_special_tokens=False)
        em = float(got == want_cap)
        ems = float(got.strip() == want_cap.strip())
        sim = _edit_sim(got, want_cap)
        for key in (r["shape"], f"{r['shape']}|{r['lang']}", "all"):
            d = per.setdefault(key, {"n": 0, "em": 0.0, "em_strip": 0.0, "sim": 0.0})
            d["n"] += 1
            d["em"] += em
            d["em_strip"] += ems
            d["sim"] += sim
    for k, d in per.items():
        for m in ("em", "em_strip", "sim"):
            d[m] /= d["n"]
    res["fim"] = per
    res["fim_examples"] = [
        {"lang": r["lang"], "shape": r["shape"], "prefix_tail": r["prefix"][-160:],
         "want": r["middle"][:160], "got": tok.decode(g, skip_special_tokens=False)[:160]}
        for r, g in list(zip(rows, gens))[:12]
    ]
    a = per.get("all", {})
    print(f"\n[fim] n={a.get('n')} EM {a.get('em', 0) * 100:.1f}% "
          f"EM(strip) {a.get('em_strip', 0) * 100:.1f}% edit-sim {a.get('sim', 0) * 100:.1f}% "
          f"in {fim_secs:.0f}s")
    for shape in ("file_fim", "repo_fim"):
        d = per.get(shape)
        if d:
            print(f"  {shape:<10} n={d['n']:<5} EM {d['em'] * 100:5.1f}%  "
                  f"EM(strip) {d['em_strip'] * 100:5.1f}%  edit-sim {d['sim'] * 100:5.1f}%")

    # ------------------------------------------------ 2/3. bits-per-byte, fresh vs stale
    import fimlib as F

    order = list(range(8192))
    __import__("random").Random(1234).shuffle(order)
    eval_parts = order[600:604]
    fresh = _sample_corpus(eval_parts[0], F.FRESH, 0)[:bpb_docs]
    stale = _sample_corpus(eval_parts[1], 0, STALE_MAX)[:bpb_docs]
    print(f"\n[bpb] {len(fresh)} fresh docs, {len(stale)} pre-2023 docs "
          f"from the same corpus and language mix")

    bpb: dict = {}
    for label, docs in (("fresh", fresh), ("stale", stale)):
        v, nb = _bits_per_byte(model, enc, [d for _l, d in docs])
        bpb[f"ours_{label}"] = v
        print(f"  ours   {label:<6} {v:.4f} bits/byte over {nb / 1e6:.1f} MB")

    if baseline:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        del model
        torch.cuda.empty_cache()
        bt = AutoTokenizer.from_pretrained(baseline)
        bm = AutoModelForCausalLM.from_pretrained(baseline, dtype=torch.bfloat16).cuda().eval()
        benc = lambda s: bt(s, add_special_tokens=False)["input_ids"]
        for label, docs in (("fresh", fresh), ("stale", stale)):
            v, nb = _bits_per_byte(bm, benc, [d for _l, d in docs])
            bpb[f"base_{label}"] = v
            print(f"  {baseline.split('/')[-1]:<13}{label:<6} {v:.4f} bits/byte")
        del bm
        torch.cuda.empty_cache()
        model, tok, sp = _load_ours(tag)

    bpb["ours_fresh_advantage"] = bpb.get("ours_stale", 0) - bpb.get("ours_fresh", 0)
    bpb["base_fresh_advantage"] = bpb.get("base_stale", 0) - bpb.get("base_fresh", 0)
    res["bpb"] = bpb
    print(f"  freshness gap (stale - fresh):  ours {bpb['ours_fresh_advantage']:+.4f}  "
          f"baseline {bpb['base_fresh_advantage']:+.4f} bits/byte")

    # ----------------------------------------------------------- 4. editor latency
    lat = {}
    for n_new in (16, 64, 128):
        p = prompts[0][-1024:]
        _greedy(model, [p], sp.eot, sp.pad, max_new=4, bs=1)     # warm the cache
        t0 = time.time()
        for _ in range(5):
            _greedy(model, [p], sp.eot, sp.pad, max_new=n_new, bs=1)
        lat[f"ms_at_{n_new}"] = (time.time() - t0) / 5 * 1000
        print(f"[lat] {n_new:>3} new tokens: {lat[f'ms_at_{n_new}']:.0f} ms")
    res["latency"] = lat

    with open(f"{VOL}/runs/{tag}/eval.json", "w") as fh:
        json.dump(res, fh, indent=1)
    vol.commit()
    return json.loads(json.dumps(res))


@app.local_entrypoint()
def run_eval(tag: str = "fimcoder-113m", max_fim: int = 1400, bpb_docs: int = 220,
             baseline: str = "Qwen/Qwen2.5-Coder-0.5B"):
    r = evaluate.remote(tag=tag, max_fim=max_fim, bpb_docs=bpb_docs, baseline=baseline)
    with open(f"eval_{tag}.json", "w") as fh:
        json.dump(r, fh, indent=1)
    a = r["fim"]["all"]
    print(f"\nFIM  n={a['n']}  EM {a['em'] * 100:.1f}%  EM(strip) {a['em_strip'] * 100:.1f}%  "
          f"edit-sim {a['sim'] * 100:.1f}%")
    print("bits/byte: " + json.dumps({k: round(v, 4) for k, v in r["bpb"].items()}))
    print("latency:   " + json.dumps({k: round(v) for k, v in r["latency"].items()}))
    rows = [(k.split("|")[1], v) for k, v in r["fim"].items() if k.startswith("file_fim|")]
    rows.sort(key=lambda x: -x[1]["sim"])
    print(f"\nbest languages (file-level FIM, edit-sim):")
    for lg, d in rows[:12]:
        print(f"  {lg:<22}n={d['n']:<4} EM {d['em'] * 100:5.1f}%  sim {d['sim'] * 100:5.1f}%")
    print(f"worst:")
    for lg, d in rows[-8:]:
        print(f"  {lg:<22}n={d['n']:<4} EM {d['em'] * 100:5.1f}%  sim {d['sim'] * 100:5.1f}%")
    print(f"\nwrote eval_{tag}.json")


@app.function(image=img.add_local_file("lang_mix.json", "/root/lang_mix.json"),
              gpu="L4", volumes={VOL: vol}, cpu=4.0, memory=24576, timeout=3000)
def freshness(tag: str = "fimcoder-113m", n_docs: int = 1500, max_chars: int = 14_000,
              baseline: str = "Qwen/Qwen2.5-Coder-0.5B", seq: int = 1024) -> dict:
    """Settle the one claim this project exists to make, on a sample big enough to mean it.

    Measured so far: +0.0582 fresh-vs-stale for ours on 320 documents, but -0.0008 on a
    different 40-document draw. Two samples disagreeing on the SIGN means the effect is
    not established. This scores ~10x the text, both models, identical documents, one
    code path, on a GPU where it costs cents instead of hours.
    """
    import random

    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    import torch
    from collections import Counter
    from huggingface_hub import HfFileSystem

    import fimlib as F

    mix = F.load_lang_mix("/root/lang_mix.json")
    langs = F.lang_array(set(mix))
    fs = HfFileSystem()
    order = list(range(8192))
    random.Random(1234).shuffle(order)

    sets = {}
    for name, lo, hi, parts in (("fresh", F.FRESH, 0, order[610:614]),
                                ("stale", 0, STALE_MAX, order[614:618])):
        per = {l: 60e6 * sh for l, sh in mix.items()}
        got: Counter = Counter()
        docs = []
        for pi in parts:
            pf = pq.ParquetFile(fs.open(PART.format(pi), "rb"))
            for batch in pf.iter_batches(batch_size=400, columns=F.ARROW_COLS):
                for _r, files in F.iter_repos(pa.Table.from_batches([batch]),
                                              langs, pc, lo, hi):
                    for f in files:
                        lg = f["lang"]
                        if got[lg] >= max(per.get(lg, 0), 60_000) or len(f["content"]) < 800:
                            continue
                        docs.append({"lang": lg, "text": f["content"][:max_chars]})
                        got[lg] += len(f["content"])
                if len(docs) >= n_docs * 6:
                    break
            if len(docs) >= n_docs * 6:
                break
        random.Random(4242).shuffle(docs)
        sets[name] = docs[:n_docs]
        nb = sum(len(d["text"].encode()) for d in sets[name])
        print(f"{name}: {len(sets[name])} docs, {nb / 1e6:.2f} MB, "
              f"{len({d['lang'] for d in sets[name]})} languages", flush=True)

    out = {"n_docs": n_docs, "seq": seq, "models": {}}
    for label in ("ours", "base"):
        if label == "ours":
            model, tok, _sp = _load_ours(tag)
            enc = lambda s: tok.encode(s, add_special_tokens=False).ids
            name = tag
        else:
            if not baseline:
                continue
            from transformers import AutoModelForCausalLM, AutoTokenizer

            bt = AutoTokenizer.from_pretrained(baseline)
            model = AutoModelForCausalLM.from_pretrained(
                baseline, dtype=torch.bfloat16).cuda().eval()
            enc = lambda s: bt(s, add_special_tokens=False)["input_ids"]
            name = baseline.split("/")[-1]
        row = {}
        for cond in ("fresh", "stale"):
            t0 = time.time()
            v, nb = _bits_per_byte(model, enc, [d["text"] for d in sets[cond]], seq=seq,
                                   stride=seq)
            row[cond] = {"bpb": round(v, 4), "bytes": nb, "secs": round(time.time() - t0)}
            print(f"  {name:<24}{cond:<6}{v:.4f} bits/byte over {nb / 1e6:.2f} MB "
                  f"({row[cond]['secs']}s)", flush=True)
        row["gap"] = round(row["stale"]["bpb"] - row["fresh"]["bpb"], 4)
        row["rel_gap_pct"] = round(row["gap"] / row["stale"]["bpb"] * 100, 2)
        print(f"  {name:<24}gap (stale-fresh) {row['gap']:+.4f}  "
              f"= {row['rel_gap_pct']:+.2f}% relative")
        out["models"][name] = row
        del model
        torch.cuda.empty_cache()
    with open(f"{VOL}/freshness.json", "w") as fh:
        json.dump(out, fh, indent=1)
    vol.commit()
    return json.loads(json.dumps(out))


@app.local_entrypoint()
def run_freshness(tag: str = "fimcoder-113m", n_docs: int = 1500):
    r = freshness.remote(tag=tag, n_docs=n_docs)
    with open("freshness.json", "w") as fh:
        json.dump(r, fh, indent=1)
    print("\n" + json.dumps(r["models"], indent=1))
