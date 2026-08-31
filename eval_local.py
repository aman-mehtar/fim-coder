"""
Evaluation on this box: 4 Neoverse-N1 cores, no GPU, and free.

    python3 eval_local.py samples          # build identical fresh / pre-2023 text sets
    python3 eval_local.py bpb <model_dir>  # bits per byte (torch, fp32, CPU)
    python3 eval_local.py fim <gguf>       # FIM exact-match + edit similarity via llama.cpp
    python3 eval_local.py latency <gguf>   # what an editor will actually feel

Measured on this box for the 113M config: a torch fp32 forward at seq 2048 takes 4.8 s
(426 tok/s), so bits-per-byte is minutes per model; Qwen2.5-Coder-0.5B is ~4.4x that, which
is why its sample is smaller and why the run starts before ours is even trained. FIM
generation goes through llama.cpp instead, where a quantised 113M model decodes an order of
magnitude faster than torch fp32.
"""

import json
import math
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LLAMA = os.path.expanduser("~/llama.cpp")
SAMPLES = os.path.join(HERE, "out", "samples")
FRESH_MIN = 1_704_067_200          # 2024-01-01
STALE_MAX = 1_672_531_200          # 2023-01-01
UUID = "990b4288-3824-41ac-94a0-b6fd6fa23ffe"
PART = f"datasets/HuggingFaceCode/stack-v3-train/data/part-{{:05d}}-{UUID}-c000.snappy.parquet"


def _part_order():
    import random

    o = list(range(8192))
    random.Random(1234).shuffle(o)
    return o


def build_samples(n_docs=60, max_chars=12_000):
    """One fresh and one pre-2023 set from the same corpus, language mix and code path.

    Both models are scored on byte-identical text; that is the only way the fresh-vs-stale
    comparison means anything.
    """
    import random

    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    sys.path.insert(0, HERE)
    import fimlib as F

    os.makedirs(SAMPLES, exist_ok=True)
    mix = F.load_lang_mix()
    langs = F.lang_array(set(mix))
    fs = HfFileSystem()
    order = _part_order()

    for name, lo, hi, part in (("fresh", FRESH_MIN, 0, order[600]),
                               ("stale", 0, STALE_MAX, order[601])):
        t0 = time.time()
        pf = pq.ParquetFile(fs.open(PART.format(part), "rb"))
        per_lang = {l: 12e6 * sh for l, sh in mix.items()}
        got, docs = {}, []
        # iter_batches, not read_row_group: this session runs in a memory cgroup with
        # `memory.oom.group` set, so one oversized allocation kills the whole process
        # group silently. A row group's `files` column is ~700 MB decoded and the
        # pre-2023 window keeps most of it; streaming 200 repos at a time bounds that,
        # and lets the loop stop as soon as it has enough documents.
        import pyarrow as pa

        for batch in pf.iter_batches(batch_size=200, columns=F.ARROW_COLS):
            tbl = pa.Table.from_batches([batch])
            for _repo, files in F.iter_repos(tbl, langs, pc, lo, hi):
                for f in files:
                    lg = f["lang"]
                    if got.get(lg, 0) >= max(per_lang.get(lg, 0), 25_000):
                        continue
                    if len(f["content"]) < 800:
                        continue
                    docs.append({"lang": lg, "text": f["content"][:max_chars]})
                    got[lg] = got.get(lg, 0) + len(f["content"])
            del tbl, batch
            if len(docs) >= n_docs * 12:
                break
        random.Random(4242).shuffle(docs)
        docs = docs[:n_docs]
        p = f"{SAMPLES}/{name}.jsonl"
        with open(p, "w") as fh:
            for d in docs:
                fh.write(json.dumps(d) + "\n")
        nb = sum(len(d["text"].encode()) for d in docs)
        print(f"{name:6} {len(docs)} docs, {nb / 1e6:.2f} MB, "
              f"{len({d['lang'] for d in docs})} languages, {time.time() - t0:.0f}s -> {p}")


def bits_per_byte(model_dir, sample, seq=1024, log_every=25):
    """Sum -log2 p over each document, divided by its UTF-8 byte length.

    Tokenizer-independent, so a 32k-vocab model and a 152k-vocab one compare honestly.

    Two memory decisions, both forced by the cgroup this box runs in: windows default
    to 1024 rather than 2048, and the loss goes through `cross_entropy(reduction="sum")`
    instead of materialising a full `log_softmax`. For Qwen2.5-Coder's 151,936-wide
    vocabulary a 2048-token window's log-probs alone are 1.2 GB. Both models must use
    the SAME window size for the comparison to mean anything.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(4)
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).eval()
    model.config.use_cache = False
    docs = [json.loads(l) for l in open(sample)]
    nats = nbytes = ntok = win = 0
    t0 = time.time()
    for i, d in enumerate(docs):
        ids = tok(d["text"], add_special_tokens=False)["input_ids"]
        if len(ids) < 16:
            continue
        nbytes += len(d["text"].encode())
        for s in range(0, len(ids), seq):
            w = ids[s : s + seq]
            if len(w) < 2:
                break
            x = torch.tensor([w])
            with torch.no_grad():
                lg = model(input_ids=x).logits[0, :-1].float()
                nats += torch.nn.functional.cross_entropy(
                    lg, x[0, 1:], reduction="sum").item()
            del lg
            ntok += len(w) - 1
            win += 1
        if (i + 1) % log_every == 0:
            print(f"  {i + 1}/{len(docs)} docs, {win} windows, {ntok:,} tokens, "
                  f"{time.time() - t0:.0f}s", flush=True)
    bpb = nats / max(nbytes, 1) / math.log(2)
    name = os.path.basename(model_dir.rstrip("/")) or model_dir
    print(f"{name:<30}{os.path.basename(sample):<14}{bpb:8.4f} bits/byte  "
          f"({nbytes / 1e6:.2f} MB, {ntok:,} tokens, "
          f"{nbytes / max(ntok, 1):.2f} bytes/token, {time.time() - t0:.0f}s)")
    return {"model": name, "sample": os.path.basename(sample), "bpb": round(bpb, 4),
            "bytes": nbytes, "tokens": ntok, "seq": seq,
            "secs": round(time.time() - t0)}


def bpb_all(model_dir, seq=1024, tag=""):
    rows = [bits_per_byte(model_dir, f"{SAMPLES}/{n}.jsonl", seq)
            for n in ("fresh", "stale")]
    gap = rows[1]["bpb"] - rows[0]["bpb"]
    print(f"  freshness gap (stale - fresh): {gap:+.4f} bits/byte")
    out = {"model": model_dir, "rows": rows, "gap": round(gap, 4)}
    p = os.path.join(HERE, "out", f"bpb_{tag or os.path.basename(model_dir.rstrip('/'))}.json")
    with open(p, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"  wrote {p}")
    return out



def fim_eval(gguf, n=300, max_new=48, port=8081, parallel=4, ctx=4096,
             max_secs=1500):
    """FIM exact-match and edit similarity through llama.cpp, on this box.

    Driven through `llama-server` rather than one `llama-cli` per example: process
    startup would dominate, and the server path is exactly what an editor talks to, so
    the numbers describe the thing you will actually run.

    Time-boxed. Prefill dominates -- these prompts are 1200-1900 tokens -- and on four
    ARM cores a 113M model needs seconds per example, so a fixed example count can
    silently turn into hours. Whatever finishes inside `max_secs` gets scored, and the
    count is reported alongside the numbers.
    """
    import urllib.error
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor

    rows = [json.loads(l) for l in open(f"{os.path.dirname(SAMPLES)}/fimeval.jsonl")]
    rows = [r for r in rows if r["n_ids"] < ctx - 64][:n]
    print(f"{len(rows)} examples, {len({r['lang'] for r in rows})} languages, "
          f"{sum(1 for r in rows if r['shape'] == 'repo_fim')} multi-file")

    srv = subprocess.Popen(
        [f"{LLAMA}/build/bin/llama-server", "-m", gguf, "--host", "127.0.0.1",
         "--port", str(port), "-c", str(ctx * parallel), "-np", str(parallel),
         "-t", "4", "--no-warmup"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(120):
            try:
                urllib.request.urlopen(f"{base}/health", timeout=2).read()
                break
            except Exception:
                time.sleep(1)
        else:
            print("llama-server did not come up")
            return None

        def one(r):
            prompt = (r["context"] + "<|fim_prefix|>" + r["prefix"]
                      + "<|fim_suffix|>" + r["suffix"] + "<|fim_middle|>")
            body = json.dumps({"prompt": prompt, "n_predict": max_new,
                               "temperature": 0, "cache_prompt": False,
                               "stop": ["<|endoftext|>", "<|file_sep|>"]}).encode()
            req = urllib.request.Request(f"{base}/completion", data=body,
                                        headers={"Content-Type": "application/json"})
            try:
                return json.loads(urllib.request.urlopen(req, timeout=300).read())["content"]
            except Exception as e:
                return f"<<ERR {type(e).__name__}>>"

        t0 = time.time()
        gens, done = [], []
        with ThreadPoolExecutor(max_workers=parallel) as ex:
            for i in range(0, len(rows), parallel * 4):
                chunk = rows[i : i + parallel * 4]
                gens.extend(ex.map(one, chunk))
                done.extend(chunk)
                el = time.time() - t0
                print(f"  {len(done)}/{len(rows)} examples, {el:.0f}s "
                      f"({el / len(done) * 1000:.0f} ms each)", flush=True)
                if el > max_secs:
                    print(f"  time box {max_secs}s reached; scoring {len(done)} examples")
                    break
        rows = done
        secs = time.time() - t0
    finally:
        srv.terminate()
        srv.wait(timeout=30)
    return _score_fim(rows, gens, secs, max_new)


def _edit_sim(a, b):
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


def _score_fim(rows, gens, secs, max_new, tok_json=None):
    """Exact match, edit similarity and first-line match, split by shape and language.

    Ground truth is capped to the token budget the model was given, so a long middle is
    not scored as a miss purely for being longer than `max_new`. First-line match is
    reported because that is what an editor actually shows you.
    """
    from collections import defaultdict

    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(tok_json or os.path.join(HERE, "tokenizer", "tokenizer.json"))
    agg = defaultdict(lambda: {"n": 0, "em": 0.0, "ems": 0.0, "sim": 0.0, "line": 0.0})
    errs = 0
    for r, g in zip(rows, gens):
        if g.startswith("<<ERR"):
            errs += 1
            continue
        want = tok.decode(tok.encode(r["middle"], add_special_tokens=False).ids[:max_new],
                          skip_special_tokens=False)
        em = float(g == want)
        ems = float(g.strip() == want.strip())
        sim = _edit_sim(g, want)
        line = float(g.split("\n", 1)[0].rstrip() == want.split("\n", 1)[0].rstrip())
        for k in ("all", r["shape"], f"lang:{r['lang']}"):
            d = agg[k]
            d["n"] += 1
            d["em"] += em
            d["ems"] += ems
            d["sim"] += sim
            d["line"] += line
    for d in agg.values():
        for m in ("em", "ems", "sim", "line"):
            d[m] /= max(d["n"], 1)
    a = agg["all"]
    print(f"\n{'set':<22}{'n':>5}{'EM':>8}{'EM.strip':>10}{'edit-sim':>10}{'1st-line':>10}")
    for k in ("all", "file_fim", "repo_fim"):
        if k in agg:
            d = agg[k]
            print(f"{k:<22}{d['n']:>5}{d['em'] * 100:>7.1f}%{d['ems'] * 100:>9.1f}%"
                  f"{d['sim'] * 100:>9.1f}%{d['line'] * 100:>9.1f}%")
    langs = sorted(((k[5:], v) for k, v in agg.items() if k.startswith("lang:") and v["n"] >= 4),
                   key=lambda x: -x[1]["sim"])
    if langs:
        print("\nby language (n >= 4), best and worst by edit similarity:")
        for lg, d in langs[:10] + [("...", None)] + langs[-6:]:
            if d is None:
                print("  ...")
                continue
            print(f"  {lg:<22}{d['n']:>4}{d['em'] * 100:>7.1f}%{d['ems'] * 100:>9.1f}%"
                  f"{d['sim'] * 100:>9.1f}%{d['line'] * 100:>9.1f}%")
    print(f"\n{len(gens)} completions in {secs:.0f}s "
          f"({secs / max(len(gens), 1) * 1000:.0f} ms each, 4 slots), {errs} errors")
    out = {"all": dict(agg["all"]), "by_shape": {k: dict(v) for k, v in agg.items()
                                                if k in ("file_fim", "repo_fim")},
           "by_lang": {k[5:]: dict(v) for k, v in agg.items() if k.startswith("lang:")},
           "secs": secs, "errors": errs, "max_new": max_new}
    with open(os.path.join(HERE, "out", "fim_local.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    return out


def latency(gguf, ctx=4096, prompt_tokens=1200, news=(16, 48, 96), reps=3):
    """What an editor feels: prefill a realistic window, then time N new tokens."""
    import urllib.request
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(os.path.join(HERE, "tokenizer", "tokenizer.json"))
    rows = [json.loads(l) for l in open(f"{os.path.dirname(SAMPLES)}/fimeval.jsonl")]
    r = next(x for x in rows if x["n_ids"] > prompt_tokens)
    prompt = ("<|fim_prefix|>" + r["prefix"] + "<|fim_suffix|>" + r["suffix"]
              + "<|fim_middle|>")
    npro = len(tok.encode(prompt, add_special_tokens=False).ids)
    srv = subprocess.Popen(
        [f"{LLAMA}/build/bin/llama-server", "-m", gguf, "--host", "127.0.0.1",
         "--port", "8082", "-c", str(ctx), "-t", "4", "--no-warmup"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(120):
            try:
                urllib.request.urlopen("http://127.0.0.1:8082/health", timeout=2).read()
                break
            except Exception:
                time.sleep(1)
        out = {"prompt_tokens": npro, "gguf": os.path.basename(gguf)}
        for n in news:
            ts = []
            for _ in range(reps):
                body = json.dumps({"prompt": prompt, "n_predict": n, "temperature": 0,
                                   "cache_prompt": False}).encode()
                req = urllib.request.Request(
                    "http://127.0.0.1:8082/completion", data=body,
                    headers={"Content-Type": "application/json"})
                t0 = time.time()
                urllib.request.urlopen(req, timeout=600).read()
                ts.append(time.time() - t0)
            out[f"ms_at_{n}"] = round(min(ts) * 1000)
            print(f"  {npro}-token prompt + {n:>3} new tokens: {min(ts) * 1000:.0f} ms")
    finally:
        srv.terminate()
        srv.wait(timeout=30)
    with open(os.path.join(HERE, "out", "latency_local.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    return out


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "samples"
    if cmd == "samples":
        build_samples(*[int(x) for x in sys.argv[2:3]])
    elif cmd == "fim":
        fim_eval(sys.argv[2], n=int(sys.argv[3]) if len(sys.argv) > 3 else 400)
    elif cmd == "latency":
        latency(sys.argv[2])
    elif cmd == "bpb":
        bpb_all(sys.argv[2], seq=int(sys.argv[3]) if len(sys.argv) > 3 else 1024,
                tag=sys.argv[4] if len(sys.argv) > 4 else "")
