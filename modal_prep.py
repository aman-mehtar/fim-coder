"""
Modal data pipeline for fimcoder-300m.

    modal run modal_prep.py::probe_all          # ~$0.15, measures everything
    modal run modal_prep.py::make_tokenizer     # ~$0.05, trains the 32k BPE
    modal run modal_prep.py::build_all          # ~$1-3,  writes uint16 shards

Why the probe exists: the 2024+ freshness filter changes the language mix
drastically (measured: C++ 15.2% of the corpus but 3.1% of fresh code; C# 2.0%
but 12.8%). Any per-language budget computed from the published corpus stats is
therefore wrong. The probe measures the *fresh* distribution over a few hundred
row groups so the build can set honest caps and repetition factors.
"""

import json
import os
import random
import time

import modal

DATASET = "HuggingFaceCode/stack-v3-train"
UUID = "990b4288-3824-41ac-94a0-b6fd6fa23ffe"
PART_TMPL = f"datasets/{DATASET}/data/part-{{:05d}}-{UUID}-c000.snappy.parquet"
N_PARTS = 8192
SEED = 1234

app = modal.App("fimcoder-prep")
vol = modal.Volume.from_name("fimcoder", create_if_missing=True)

img = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "pyarrow==21.0.0",
        "huggingface-hub==0.36.0",
        "tokenizers==0.23.1",
        "gigatoken==0.10.0",
        "numpy==2.3.4",
        "hf-transfer==0.1.9",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_python_source("fimlib", "train_tokenizer")
    .add_local_file("lang_mix.json", "/root/lang_mix.json")
)

MIX_PATH = "/root/lang_mix.json"
VOL = "/vol"

F_SPECIALS = [
    "<|endoftext|>", "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>",
    "<|fim_pad|>", "<|repo_name|>", "<|file_sep|>",
]


def part_order() -> list[int]:
    """One canonical shuffle of all 8192 parts, shared by every stage.

    Parts are sorted by repo_path, so reading them in order would sample a narrow
    alphabetical slice of GitHub. Shuffling once with a fixed seed keeps probe /
    val / train slices disjoint and reproducible without an `fs.ls` round trip.
    """
    o = list(range(N_PARTS))
    random.Random(SEED).shuffle(o)
    return o


# --------------------------------------------------------------------- probe

PROBE_WORKERS = 32
PROBE_RG_PER_WORKER = 6
PROBE_SAMPLE_MB = 620          # text kept for tokenizer training, total across workers


def _open_part(fs, idx):
    import pyarrow.parquet as pq

    return pq.ParquetFile(fs.open(PART_TMPL.format(idx), "rb"))


def _read_row_groups(fs, idx, n_rg, rng, cols):
    """Yield tables for `n_rg` randomly chosen row groups of one part."""
    pf = _open_part(fs, idx)
    order = list(range(pf.num_row_groups))
    rng.shuffle(order)
    for rg in order[:n_rg]:
        yield pf.read_row_group(rg, columns=cols)


@app.function(image=img, volumes={VOL: vol}, cpu=2.0, memory=8192,
              timeout=2400, retries=2)
def probe(wid: int, parts: list[int], sample_quota: dict[str, int]) -> dict:
    """Measure the fresh-code language distribution and bank a balanced text sample."""
    import pyarrow.compute as pc
    from collections import Counter
    from huggingface_hub import HfFileSystem

    import fimlib as F

    mix = F.load_lang_mix(MIX_PATH)
    langs = F.lang_array(set(mix))
    fs = HfFileSystem()
    rng = random.Random(9000 + wid)

    nbytes = Counter()      # kept content bytes per language
    nfiles = Counter()
    got = Counter()         # bytes written to the sample per language
    n_repos = n_files = 0
    raw_decoded = 0
    t_read = t_work = 0.0
    t0 = time.time()

    os.makedirs(f"{VOL}/tsample", exist_ok=True)
    out = open(f"{VOL}/tsample/w{wid:03d}.bin", "wb")

    for idx in parts:
        try:
            ta = time.time()
            for tbl in _read_row_groups(fs, idx, 1, rng, F.ARROW_COLS):
                raw_decoded += tbl.column("files").nbytes
                t_read += time.time() - ta
                tb = time.time()
                for repo, files in F.iter_repos(tbl, langs, pc):
                    n_repos += 1
                    n_files += len(files)
                    for f in files:
                        lg, c = f["lang"], f["content"]
                        nbytes[lg] += len(c)
                        nfiles[lg] += 1
                        if got[lg] < sample_quota.get(lg, 0):
                            b = c.encode("utf-8", "ignore")
                            out.write(f"{lg}\t{len(b)}\n".encode())
                            out.write(b)
                            got[lg] += len(b)
                t_work += time.time() - tb
                ta = time.time()
        except Exception as e:                      # a bad part must not kill the probe
            print(f"[w{wid}] part {idx} failed: {type(e).__name__}: {e}")

    out.close()
    vol.commit()
    return {
        "wid": wid, "parts": len(parts), "repos": n_repos, "files": n_files,
        "raw_decoded": raw_decoded, "kept": sum(nbytes.values()),
        "sample": sum(got.values()), "t_read": t_read, "t_work": t_work,
        "wall": time.time() - t0, "nbytes": dict(nbytes), "nfiles": dict(nfiles),
    }


@app.local_entrypoint()
def probe_all(workers: int = PROBE_WORKERS, rg: int = PROBE_RG_PER_WORKER):
    """Fan out the probe, then print the fresh-language table the build needs."""
    import fimlib as F

    mix = F.load_lang_mix()
    order = part_order()
    n = workers * rg

    # Tokenizer sample: halfway between the training mix and uniform, so rare
    # languages get real vocabulary presence instead of a rounding error.
    u = 1.0 / len(mix)
    tmix = {k: 0.6 * v + 0.4 * u for k, v in mix.items()}
    s = sum(tmix.values())
    quota = {k: int(PROBE_SAMPLE_MB * 1e6 * v / s / workers) for k, v in tmix.items()}

    slices = [order[i * rg : (i + 1) * rg] for i in range(workers)]
    t0 = time.time()
    res = list(probe.starmap([(i, sl, quota) for i, sl in enumerate(slices)]))
    wall = time.time() - t0

    agg, files = {}, {}
    for r in res:
        for k, v in r["nbytes"].items():
            agg[k] = agg.get(k, 0) + v
        for k, v in r["nfiles"].items():
            files[k] = files.get(k, 0) + v
    tot = sum(agg.values())
    repos = sum(r["repos"] for r in res)
    raw = sum(r["raw_decoded"] for r in res)
    rgs = sum(r["parts"] for r in res)

    print(f"\n=== probe: {rgs} row groups, {repos:,} repos, "
          f"{sum(r['files'] for r in res):,} files ===")
    print(f"decoded {raw / 1e9:.1f} GB -> kept {tot / 1e9:.2f} GB ({tot / raw * 100:.1f}%)")
    print(f"wall {wall:.0f}s | mean worker {sum(r['wall'] for r in res) / len(res):.0f}s "
          f"(read {sum(r['t_read'] for r in res) / len(res):.0f}s, "
          f"filter {sum(r['t_work'] for r in res) / len(res):.0f}s)")
    print(f"per row group: {tot / rgs / 1e6:.1f} MB kept, "
          f"{tot / rgs / 3.55 / 1e6:.1f}M tokens, {repos / rgs:.0f} repos")
    print(f"languages seen: {len(agg)} / {len(mix)}   sample banked: "
          f"{sum(r['sample'] for r in res) / 1e6:.0f} MB")

    stats = {"rgs": rgs, "repos": repos, "kept": tot, "raw": raw,
             "bytes": agg, "files": files, "wall": wall}
    with open("probe_stats.json", "w") as fh:
        json.dump(stats, fh, indent=1)

    print(f"\n{'lang':<24}{'MB':>9}{'fresh%':>8}{'target%':>9}{'x-short':>8}")
    for lg in sorted(mix, key=lambda k: -mix[k]):
        have = agg.get(lg, 0) / tot if tot else 0
        short = mix[lg] / have if have else float("inf")
        flag = "  <<" if short > 3 else ""
        print(f"{lg:<24}{agg.get(lg, 0) / 1e6:>9.1f}{have * 100:>8.3f}"
              f"{mix[lg] * 100:>9.3f}{short:>8.1f}{flag}")
    missing = [k for k in mix if k not in agg]
    if missing:
        print(f"\nnot seen at all ({len(missing)}): {', '.join(missing)}")
    print("\nwrote probe_stats.json")


# ---------------------------------------------------------------- tokenizer

def _read_sample(paths):
    """Parse the probe's `lang\\tlen\\n<bytes>` framing into per-language byte lists."""
    out: dict[str, list[bytes]] = {}
    for p in paths:
        with open(p, "rb") as fh:
            while True:
                head = fh.readline()
                if not head:
                    break
                try:
                    lg, n = head.decode().rstrip("\n").rsplit("\t", 1)
                    body = fh.read(int(n))
                except Exception:
                    break
                if len(body) < int(n):
                    break
                out.setdefault(lg, []).append(body)
    return out


def _balance(banked, tmix, target_bytes, floor=700_000):
    """Cap each language at tmix[l]*T bytes, choosing T so the total hits target_bytes.

    Languages with less than their cap simply contribute everything they have; the
    floor guarantees even a scarce language gets enough text for BPE to pick up its
    keywords and operators.
    """
    have = {k: sum(map(len, v)) for k, v in banked.items()}
    lo, hi = 0.0, 400 * target_bytes

    def total(T):
        return sum(min(have[k], max(int(tmix.get(k, 0) * T), floor)) for k in have)

    for _ in range(60):
        mid = (lo + hi) / 2
        if total(mid) < target_bytes:
            lo = mid
        else:
            hi = mid
    T = (lo + hi) / 2
    return {k: min(have[k], max(int(tmix.get(k, 0) * T), floor)) for k in have}, have


@app.function(image=img, volumes={VOL: vol}, cpu=8.0, memory=32768, timeout=3600)
def make_tokenizer(vocab_size: int = 32768, sample_mb: int = 480, held_frac: float = 0.12) -> dict:
    """Train the 32k byte-level BPE on a language-balanced sample, on Modal.

    Runs here rather than locally because the sample already lives on the Volume
    and gigatoken trains 32k merges on ~400 MB in seconds. Only the 1.5 MB
    tokenizer.json comes back.
    """
    import glob

    import gigatoken as gt
    from tokenizers import Tokenizer as HFTok

    import fimlib as F
    import train_tokenizer as TT

    mix = F.load_lang_mix(MIX_PATH)
    u = 1.0 / len(mix)
    tmix = {k: 0.6 * v + 0.4 * u for k, v in mix.items()}
    s = sum(tmix.values())
    tmix = {k: v / s for k, v in tmix.items()}

    banked = _read_sample(sorted(glob.glob(f"{VOL}/tsample/w*.bin")))
    caps, have = _balance(banked, tmix, int(sample_mb * 1e6))
    print(f"banked {sum(have.values()) / 1e6:.0f} MB over {len(have)} langs -> "
          f"using {sum(caps.values()) / 1e6:.0f} MB")

    sep = b"<|endoftext|>"
    train_p, held_p = f"{VOL}/tok_train.txt", f"{VOL}/tok_held.txt"
    mix_used = {}
    with open(train_p, "wb") as tr, open(held_p, "wb") as hd:
        for lg, docs in sorted(banked.items()):
            budget = caps[lg]
            used = 0
            random.Random(11).shuffle(docs)
            cut = int(budget * (1 - held_frac))
            for d in docs:
                if used >= budget:
                    break
                (tr if used < cut else hd).write(d + sep)
                used += len(d)
            mix_used[lg] = used
    tot = sum(mix_used.values())
    top = sorted(mix_used.items(), key=lambda x: -x[1])[:14]
    print("sample mix: " + ", ".join(f"{k} {v / tot * 100:.1f}%" for k, v in top))

    t0 = time.time()
    vocab, merges = gt.train_bpe(
        gt.TextFileSource([train_p], separator=sep), vocab_size, F_SPECIALS
    )
    secs = time.time() - t0

    os.makedirs(f"{VOL}/tokenizer", exist_ok=True)
    spec = TT.to_hf_tokenizer(vocab, merges)
    tp = f"{VOL}/tokenizer/tokenizer.json"
    with open(tp, "w") as fh:
        json.dump(spec, fh)
    tok = HFTok.from_file(tp)
    ids = {t: tok.token_to_id(t) for t in F_SPECIALS}
    assert all(v is not None for v in ids.values()), ids
    with open(f"{VOL}/tokenizer/specials.json", "w") as fh:
        json.dump(ids, fh, indent=2)
    vol.commit()
    print(f"trained {tok.get_vocab_size()} vocab in {secs:.0f}s, specials {ids}")
    return {"secs": secs, "vocab": tok.get_vocab_size(), "specials": ids,
            "sample_mb": tot / 1e6, "mix_used": mix_used}


@app.function(image=img, volumes={VOL: vol}, cpu=8.0, memory=32768, timeout=3600)
def bench_tokenizer() -> dict:
    """Fertility on the balanced held-out slice. Every candidate goes through HF
    `tokenizers` so the comparison uses one identical code path."""
    from tokenizers import Tokenizer as HFTok

    with open(f"{VOL}/tok_held.txt", "rb") as fh:
        raw = fh.read()
    docs = [d.decode("utf-8", "ignore") for d in raw.split(b"<|endoftext|>") if d]
    nbytes = sum(len(d.encode()) for d in docs)
    print(f"held-out {nbytes / 1e6:.0f} MB / {len(docs):,} files")

    rows = []

    def measure(label, tok, vs):
        n = sum(len(e.ids) for e in tok.encode_batch_fast(docs))
        rows.append({"tok": label, "vocab": vs, "bytes_per_tok": nbytes / n,
                     "embed_M": vs * 1024 / 1e6})
        print(f"{label:<32}{vs:>8}{nbytes / n:>10.3f}{vs * 1024 / 1e6:>9.1f}M")

    print(f"{'tokenizer':<32}{'vocab':>8}{'bytes/tok':>10}{'embed':>10}")
    t = HFTok.from_file(f"{VOL}/tokenizer/tokenizer.json")
    measure("OURS (32k, code+FIM)", t, t.get_vocab_size())
    for repo, label in [
        ("openai-community/gpt2", "GPT-2 (50k)"),
        ("bigcode/starcoder2-tokenizer", "StarCoder2 (49k)"),
        ("Qwen/Qwen2.5-Coder-1.5B", "Qwen2.5-Coder (152k)"),
    ]:
        try:
            measure(label, HFTok.from_pretrained(repo), HFTok.from_pretrained(repo).get_vocab_size())
        except Exception as e:
            print(f"{label:<32}  skipped: {type(e).__name__}")
    base = rows[0]["bytes_per_tok"]
    for r in rows[1:]:
        r["vs_ours_pct"] = (base / r["bytes_per_tok"] - 1) * 100
        print(f"  ours vs {r['tok']:<26}{r['vs_ours_pct']:+6.1f}% bytes/token, "
              f"{(r['vocab'] - rows[0]['vocab']) * 1024 / 1e6:+6.1f}M params")
    return {"rows": rows, "held_mb": nbytes / 1e6}


@app.local_entrypoint()
def tokenizer_all(vocab_size: int = 32768, sample_mb: int = 480):
    r = make_tokenizer.remote(vocab_size=vocab_size, sample_mb=sample_mb)
    b = bench_tokenizer.remote()
    with open("tokenizer_report.json", "w") as fh:
        json.dump({"train": r, "bench": b}, fh, indent=1)
    print(f"\nsample {r['sample_mb']:.0f} MB, vocab {r['vocab']}, "
          f"trained in {r['secs']:.0f}s -> tokenizer_report.json")


# ------------------------------------------------------------------- build

MAX_DOCS_PER_REPO = 40
# Files per language per repo. Under-quota languages get a much larger allowance:
# with a flat cap of 8, the languages that needed the most data realised only 33% of
# what the corpus held (measured), because a Rust repo with 40 Rust files donated 8.
FILES_PER_LANG_FULL = 8
FILES_PER_LANG_SHORT = 32
REPOS_PER_FLUSH = 250          # keeps per-language caps from overshooting inside a batch
SHORT_DOC_FRAC = 0.12          # windows a real editor sends near the top of a file
SHORT_DOC_SEQS = (256, 384, 512, 768, 1024)
FLUSH_RECORDS = 16_384


@app.function(image=img, volumes={VOL: vol}, cpu=4.0, memory=16384,
              timeout=5400, retries=2)
def build(wid: int, parts: list, cap: dict, repeat: dict,
          split: str = "train", seq: int = 2048, mix_shapes=None) -> dict:
    """Stream parquet -> filter -> FIM documents -> packed uint16 records.

    One document per *file* of an under-quota language, not one per repo: sampling a
    single file per repo left 90% of the corpus unread and the per-language budgets
    unreachable (measured on the first build: 1.6 docs/repo, 1.7B tokens, and a mix
    23.9% away from target because no cap ever bound).
    """
    import numpy as np
    import pyarrow.compute as pc
    from collections import Counter
    from huggingface_hub import HfFileSystem
    from tokenizers import Tokenizer

    import fimlib as F

    mix = F.load_lang_mix(MIX_PATH)
    tok = Tokenizer.from_file(f"{VOL}/tokenizer/tokenizer.json")
    sp = F.Specials(f"{VOL}/tokenizer")
    budget = F.LangBudget(mix, 1)
    budget.cap = dict(cap)
    budget.target = {k: int(v / 1.02) for k, v in cap.items()}
    packer = F.Packer(sp, seq=seq)
    rng = random.Random(4242 + wid)
    fs = HfFileSystem()
    if mix_shapes:
        F.MIX = dict(mix_shapes)

    os.makedirs(f"{VOL}/data/s{seq}", exist_ok=True)
    path = f"{VOL}/data/s{seq}/{split}_{wid:03d}.bin"
    fh = open(path, "wb")
    shapes = Counter()
    buf: list = []
    n_rec = n_doc = n_repo = 0
    t0 = time.time()
    pi = -1

    def write_records():
        nonlocal buf, n_rec
        if buf:
            rng.shuffle(buf)
            np.asarray(buf, dtype=np.uint16).tofile(fh)
            n_rec += len(buf)
            buf = []

    def emit(plans, pieces):
        """Tokenise one batch of planned documents, assemble, account, pack."""
        nonlocal n_doc
        if not plans:
            return
        encs = tok.encode_batch_fast(pieces, add_special_tokens=False)
        use: list = []
        for p in plans:
            o, k = p["_off"], len(p["pieces"])
            ids = F.assemble(p, [encs[o + i].ids for i in range(k)], sp, p["_seq"], use)
            if not ids:
                continue
            for lg, n in zip(p["plangs"], use):
                if n:
                    budget.add(lg, n)
            shapes[p["shape"]] += 1
            n_doc += 1
            buf.extend(packer.add(ids))
        if len(buf) >= FLUSH_RECORDS:
            write_records()

    open_langs = set(mix)
    langs_arr = F.lang_array(open_langs)

    for pi, idx in enumerate(parts):
        try:
            pf = _open_part(fs, idx)
            for rg in range(pf.num_row_groups):
                tbl = pf.read_row_group(rg, columns=F.ARROW_COLS)
                plans: list = []
                pieces: list = []
                since = 0
                for repo, files in F.iter_repos(tbl, langs_arr, pc):
                    n_repo += 1
                    by: dict = {}
                    for f in files:
                        by.setdefault(f["lang"], []).append(f)
                    here = {lg for lg in by if budget.allow(lg)}
                    if not here:
                        continue
                    n = 0
                    # Least-full first, scarcest target share as tie-break: a repo that
                    # happens to contain Zig spends its slots on Zig, not on its README.
                    for lg in sorted(here, key=lambda l: (budget.ratio(l), mix.get(l, 1.0))):
                        fl = by[lg]
                        nf = (FILES_PER_LANG_FULL if budget.ratio(lg) > 0.9
                              else FILES_PER_LANG_SHORT)
                        if len(fl) > nf:
                            fl = rng.sample(fl, nf)
                        for f in fl:
                            for _ in range(min(3, repeat.get(lg, 1))):
                                p = F.plan_doc(repo, files, rng, target=f, sib_langs=here)
                                if p is None:
                                    continue
                                p["_off"] = len(pieces)
                                p["_seq"] = (seq if rng.random() > SHORT_DOC_FRAC
                                             else rng.choice(SHORT_DOC_SEQS))
                                pieces.extend(p["pieces"])
                                plans.append(p)
                                n += 1
                                if n >= MAX_DOCS_PER_REPO:
                                    break
                            if n >= MAX_DOCS_PER_REPO:
                                break
                        if n >= MAX_DOCS_PER_REPO:
                            break
                    since += 1
                    if since >= REPOS_PER_FLUSH:
                        emit(plans, pieces)
                        plans, pieces, since = [], [], 0
                emit(plans, pieces)
                open_langs = budget.open_langs()
                langs_arr = F.lang_array(open_langs or {"Python"})
                if not open_langs:
                    break
            if not open_langs:
                print(f"[w{wid}] all quotas met after {pi + 1} parts")
                break
        except Exception as e:
            print(f"[w{wid}] part {idx} failed: {type(e).__name__}: {e}")

    buf.extend(packer.flush())
    write_records()
    fh.close()
    vol.commit()
    return {"wid": wid, "split": split, "path": path, "records": n_rec,
            "tokens": n_rec * seq, "docs": n_doc, "repos": n_repo,
            "fill": packer.fill, "shapes": dict(shapes), "got": dict(budget.got),
            "parts_used": pi + 1, "wall": time.time() - t0}


# Part slices. Disjoint by construction so val/eval repos never appear in train.
PROBE_PARTS = 384
TRAIN_PARTS = 200
VAL_SLICE = slice(PROBE_PARTS + TRAIN_PARTS, PROBE_PARTS + TRAIN_PARTS + 6)
EVAL_SLICE = slice(PROBE_PARTS + TRAIN_PARTS + 6, PROBE_PARTS + TRAIN_PARTS + 12)


def _plan_budgets(target_tokens: float, n_rgs: int, workers: int, chars_per_tok=3.33):
    """Turn measured per-row-group availability into per-worker caps and repeat factors."""
    import fimlib as F

    mix = F.load_lang_mix()
    with open("probe_stats.json") as fh:
        st = json.load(fh)
    per_rg = {l: st["bytes"].get(l, 0) / st["rgs"] / chars_per_tok for l in mix}
    avail = {l: per_rg[l] * n_rgs for l in mix}
    repeat, cap = {}, {}
    for l, sh in mix.items():
        want = sh * target_tokens
        repeat[l] = max(1, min(3, round(want / avail[l]))) if avail[l] else 1
        cap[l] = int(want * 1.02 / workers)
    return mix, cap, repeat, avail


@app.local_entrypoint()
def build_all(target_b: float = 3.4, workers: int = 50, parts: int = TRAIN_PARTS,
              seq: int = 2048, start: int = PROBE_PARTS, repo_heavy: bool = False):
    """Fan out the shard build, then write meta.json describing the result.

    Run twice: once at seq 2048 for the bulk of training, once at seq 4096 for the
    context-extension phase (see PLAN.md). `repo_heavy` shifts the shape mixture
    toward multi-file documents, which is the point of the longer window.
    """
    order = part_order()
    train = order[start : start + parts]
    mix, cap, repeat, avail = _plan_budgets(target_b * 1e9, parts * 3, workers)
    shapes_mix = ({"repo_fim": 0.45, "file_fim": 0.30, "repo_plain": 0.15,
                   "file_plain": 0.10} if repo_heavy else None)
    rep3 = sum(1 for v in repeat.values() if v == 3)
    print(f"{parts} parts (~{parts * 3} row groups) over {workers} workers; "
          f"{rep3} languages at 3x repetition")

    chunks = [train[i::workers] for i in range(workers)]
    t0 = time.time()
    res = list(build.starmap(
        [(i, c, cap, repeat, "train", seq, shapes_mix) for i, c in enumerate(chunks)]))
    wall = time.time() - t0

    recs = sum(r["records"] for r in res)
    docs = sum(r["docs"] for r in res)
    got: dict[str, int] = {}
    shapes: dict[str, int] = {}
    for r in res:
        for k, v in r["got"].items():
            got[k] = got.get(k, 0) + v
        for k, v in r["shapes"].items():
            shapes[k] = shapes.get(k, 0) + v
    tot = sum(got.values()) or 1

    print(f"\n=== build seq={seq}: {recs:,} records = {recs * seq / 1e9:.2f}B tokens, "
          f"{docs:,} docs, fill {sum(r['fill'] for r in res) / len(res) * 100:.1f}% ===")
    print(f"wall {wall:.0f}s | mean worker {sum(r['wall'] for r in res) / len(res):.0f}s")
    print("shapes: " + ", ".join(f"{k} {v / docs * 100:.0f}%" for k, v in sorted(shapes.items())))
    tv = sum(abs(got.get(l, 0) / tot - mix[l]) for l in mix) / 2
    print(f"language mix: total-variation distance from target {tv * 100:.1f}%")
    print(f"\n{'lang':<24}{'Mtok':>9}{'got%':>8}{'target%':>9}")
    for l in sorted(mix, key=lambda k: -got.get(k, 0))[:30]:
        print(f"{l:<24}{got.get(l, 0) / 1e6:>9.1f}{got.get(l, 0) / tot * 100:>8.3f}{mix[l] * 100:>9.3f}")
    zero = [l for l in mix if not got.get(l)]
    if zero:
        print(f"\nzero tokens ({len(zero)}): {', '.join(zero)}")

    meta = {"seq": seq, "records": recs, "tokens": recs * seq, "docs": docs,
            "shapes": shapes, "lang_tokens": got, "tv_distance": tv,
            "shards": [{"path": r["path"], "records": r["records"]} for r in res],
            "parts": parts, "target_b": target_b, "mix": mix, "wall": wall}
    with open(f"data_meta_{seq}.json", "w") as fh:
        json.dump(meta, fh, indent=1)
    print(f"\nwrote data_meta_{seq}.json")


@app.function(image=img, volumes={VOL: vol}, cpu=4.0, memory=16384, timeout=3600, retries=2)
def build_fimeval(parts_eval: list, per_lang: int = 14, seq: int = 2048) -> dict:
    """FIM eval set: unpacked, with ground truth kept as TEXT.

    Text rather than ids, so the same examples can be scored against a model with a
    different tokenizer and different FIM sentinels (Qwen2.5-Coder).
    """
    import pyarrow.compute as pc
    from collections import Counter
    from huggingface_hub import HfFileSystem
    from tokenizers import Tokenizer

    import fimlib as F

    mix = F.load_lang_mix(MIX_PATH)
    tok = Tokenizer.from_file(f"{VOL}/tokenizer/tokenizer.json")
    sp = F.Specials(f"{VOL}/tokenizer")
    fs = HfFileSystem()
    langs = F.lang_array(set(mix))
    os.makedirs(f"{VOL}/data", exist_ok=True)
    rng = random.Random(78)
    seen: Counter = Counter()
    rows = []

    for idx in parts_eval:
        pf = _open_part(fs, idx)
        for rg in range(pf.num_row_groups):
            tbl = pf.read_row_group(rg, columns=F.ARROW_COLS)
            for repo, files in F.iter_repos(tbl, langs, pc):
                for shape in ("file_fim", "repo_fim"):
                    lgs = [l for l in {f["lang"] for f in files}
                           if seen[(l, shape)] < per_lang]
                    if not lgs:
                        continue
                    lg = rng.choice(lgs)
                    p = F.plan_doc(repo, files, rng, shape=shape, force_lang=lg)
                    if p is None or p["shape"] != shape:
                        continue
                    e = [tok.encode(x, add_special_tokens=False).ids for x in p["pieces"]]
                    ids = F.assemble(p, e, sp, seq)
                    if not ids or sp.mid not in ids:
                        continue
                    ip, isf, im = ids.index(sp.pre), ids.index(sp.suf), ids.index(sp.mid)
                    if not (ip < isf < im):
                        continue
                    d = lambda x, y: tok.decode(ids[x:y], skip_special_tokens=False)
                    mid = d(im + 1, len(ids) - 1)
                    if len(mid.strip()) < 3:
                        continue
                    rows.append({"lang": lg, "shape": shape, "repo": repo,
                                 "context": d(0, ip), "prefix": d(ip + 1, isf),
                                 "suffix": d(isf + 1, im), "middle": mid,
                                 "n_ids": len(ids)})
                    seen[(lg, shape)] += 1
        if len(rows) >= per_lang * 2 * len(mix):
            break
    rng.shuffle(rows)
    with open(f"{VOL}/data/fimeval.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    vol.commit()
    return {"rows": len(rows), "langs": len({r["lang"] for r in rows}),
            "shapes": dict(Counter(r["shape"] for r in rows)),
            "per_lang_hit": sum(1 for v in seen.values() if v >= per_lang)}


@app.local_entrypoint()
def holdout(val_tokens_m: float = 14.0):
    """Val shard + FIM eval set, from parts no training worker ever touches.

    The val shard is produced by the SAME `build` function as training data, so a
    val loss compares like with like instead of measuring a second code path.
    """
    import fimlib as F

    order = part_order()
    mix = F.load_lang_mix()
    cap = {l: max(4096, int(sh * val_tokens_m * 1e6 * 1.02)) for l, sh in mix.items()}
    repeat = {l: 1 for l in mix}
    v = build.remote(0, order[VAL_SLICE], cap, repeat, "val", 2048)
    print(f"val: {v['records']:,} records, {v['tokens'] / 1e6:.1f}M tokens, "
          f"fill {v['fill'] * 100:.1f}%, {sum(1 for x in v['got'].values() if x)} langs")
    e = build_fimeval.remote(order[EVAL_SLICE])
    print("fimeval: " + json.dumps(e))


FRESH_MIN = 1_704_067_200          # 2024-01-01
STALE_MAX = 1_672_531_200          # 2023-01-01


@app.function(image=img, volumes={VOL: vol}, cpu=4.0, memory=16384, timeout=1800, retries=2)
def eval_samples(n_docs: int = 60, max_chars: int = 12_000) -> dict:
    """Build byte-identical fresh (2024+) and pre-2023 text sets for local scoring.

    Runs here rather than on the laptop because a row group's `files` column is ~700 MB
    decoded and the local session lives in a memory cgroup with `memory.oom.group` set,
    so one oversized allocation kills the whole process group with no traceback. Two
    ~10 MB JSONL files come back; the expensive part -- scoring both models -- stays
    local and free.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    from collections import Counter

    import fimlib as F

    mix = F.load_lang_mix(MIX_PATH)
    langs = F.lang_array(set(mix))
    order = part_order()
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    os.makedirs(f"{VOL}/samples", exist_ok=True)
    out = {}

    for name, lo, hi, part in (("fresh", FRESH_MIN, 0, order[600]),
                               ("stale", 0, STALE_MAX, order[601])):
        t0 = time.time()
        pf = _open_part(fs, part)
        per_lang = {l: 12e6 * sh for l, sh in mix.items()}
        got: Counter = Counter()
        docs = []
        for batch in pf.iter_batches(batch_size=400, columns=F.ARROW_COLS):
            tbl = pa.Table.from_batches([batch])
            for _repo, files in F.iter_repos(tbl, langs, pc, lo, hi):
                for f in files:
                    lg = f["lang"]
                    if got[lg] >= max(per_lang.get(lg, 0), 25_000):
                        continue
                    if len(f["content"]) < 800:
                        continue
                    docs.append({"lang": lg, "text": f["content"][:max_chars]})
                    got[lg] += len(f["content"])
            if len(docs) >= n_docs * 15:
                break
        random.Random(4242).shuffle(docs)
        docs = docs[:n_docs]
        p = f"{VOL}/samples/{name}.jsonl"
        with open(p, "w") as fh:
            for d in docs:
                fh.write(json.dumps(d) + "\n")
        nb = sum(len(d["text"].encode()) for d in docs)
        out[name] = {"docs": len(docs), "bytes": nb,
                     "langs": len({d["lang"] for d in docs}), "secs": time.time() - t0}
        print(f"{name:6} {len(docs)} docs, {nb / 1e6:.2f} MB, "
              f"{out[name]['langs']} languages, {time.time() - t0:.0f}s")
    vol.commit()
    return out


@app.local_entrypoint()
def samples(n_docs: int = 60):
    print(json.dumps(eval_samples.remote(n_docs=n_docs), indent=1))
