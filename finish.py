"""
Pull everything off Modal and make it work locally, forever.

    python3 finish.py download            # HF weights + tokenizer + metrics
    python3 finish.py gguf                # f16 / q8_0 / q4_k_m + tokenizer check
    python3 finish.py onnx                # cached ONNX graph, fp32 + int4, verified
    python3 finish.py test                # CPU FIM completion via llama-cli
    python3 finish.py config              # minuet-ai.nvim + Continue.dev snippets
    python3 finish.py all

This exists because the credit expires and this box has no GPU. Whatever is still
only on a Modal volume when the credit dies is not really yours, and a 113M model
quantised to q4_k_m runs perfectly well on four ARM cores.
"""

import json
import os
import shutil
import subprocess
import time
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TAG = os.environ.get("FIMCODER_TAG", "fimcoder-113m")
OUT = os.path.join(HERE, "out", TAG)
LLAMA = os.path.expanduser("~/llama.cpp")
VOL = "fimcoder"
SPECIALS = ["<|endoftext|>", "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>",
            "<|fim_pad|>", "<|repo_name|>", "<|file_sep|>"]


def sh(cmd, **kw):
    print(f"$ {cmd}")
    return subprocess.run(cmd, shell=True, text=True, **kw)


def modal_get(remote, local):
    os.makedirs(os.path.dirname(local) or ".", exist_ok=True)
    env = dict(os.environ, PATH=os.path.expanduser("~/.local/bin") + ":" + os.environ["PATH"])
    r = subprocess.run(f"modal volume get --force {VOL} {remote} {local}",
                       shell=True, text=True, capture_output=True, env=env)
    ok = r.returncode == 0 and os.path.exists(local)
    print(f"  {'ok  ' if ok else 'MISS'} {remote}")
    return ok


def download():
    os.makedirs(f"{OUT}/hf", exist_ok=True)
    for f in ("config.json", "generation_config.json", "model.safetensors",
              "special_tokens_map.json", "specials.json", "tokenizer.json",
              "tokenizer_config.json"):
        modal_get(f"runs/{TAG}/hf/{f}", f"{OUT}/hf/{f}")
    for f in ("result.json", "eval.json"):
        modal_get(f"runs/{TAG}/{f}", f"{OUT}/{f}")
    modal_get("data/fimeval.jsonl", f"{OUT}/fimeval.jsonl")
    for f in ("tokenizer/tokenizer.json", "tokenizer/specials.json"):
        modal_get(f, f"{OUT}/{f}")
    n = sum(os.path.getsize(os.path.join(dp, f))
            for dp, _, fs in os.walk(OUT) for f in fs)
    print(f"\n{OUT}: {n / 1e6:.0f} MB")
    r = os.path.join(OUT, "result.json")
    if os.path.exists(r):
        d = json.load(open(r))
        print(f"trained {d['steps']} steps, {d['tokens'] / 1e9:.2f}B tokens, "
              f"{d['elapsed_h']:.2f}h on {d['gpu']}, MFU {d['best_mfu'] * 100:.1f}%, "
              f"val loss {d.get('val_loss')}")


def _converter_files():
    return [os.path.join(LLAMA, r) for r in
            ("conversion/base.py", "conversion/vocab.py", "convert_hf_to_gguf.py")
            if os.path.exists(os.path.join(LLAMA, r))]


def probe_chkhsh(hf, py):
    """Ask the converter itself for the fingerprint instead of recomputing it.

    convert_hf_to_gguf.py hashes the tokenisation of a fixed probe string to pick the
    pre-tokenizer regex llama.cpp will use at inference. Reproducing that hash by hand
    risks a subtle mismatch (transformers' `encode` vs the raw tokenizers API, special
    tokens, and so on), and a WRONG pre-tokenizer does not error -- it silently
    produces plausible garbage. So run the converter in verbose mode and read the hash
    it actually computed.
    """
    r = subprocess.run(f"{py} {LLAMA}/convert_hf_to_gguf.py {hf} --outfile /tmp/_probe.gguf "
                       f"--outtype f16 --verbose --vocab-only",
                       shell=True, capture_output=True, text=True)
    for line in (r.stderr + r.stdout).splitlines():
        if "chkhsh:" in line:
            return line.split("chkhsh:")[1].strip()
    return None


def patch_converter(chkhsh):
    """Teach the local llama.cpp converter our fingerprint -> starcoder pre-tokenizer.

    Our pre-tokenizer is byte-for-byte StarCoder2's: Digits(individual_digits=True)
    then ByteLevel with add_prefix_space=False. So this mapping is exact, not a guess
    -- and `verify_tokenizer` proves it afterwards by comparing ids against HF on real
    code rather than trusting the claim.
    """
    for p in _converter_files():
        src = open(p).read()
        if chkhsh in src:
            print(f"  already patched: {os.path.relpath(p, LLAMA)}")
            return True
        lines = src.splitlines(keepends=True)
        # Anchor on `res = None`, the initialisation just above the hash table, and
        # insert at ITS indent. Anchoring on `res = "starcoder"` instead nests the new
        # branch inside the existing one, where it never runs.
        j = next((k for k, ln in enumerate(lines) if ln.strip() == "res = None"), None)
        if j is None:
            continue
        indent = lines[j][: len(lines[j]) - len(lines[j].lstrip())]
        block = (f'{indent}if chkhsh == "{chkhsh}":\n'
                 f'{indent}    # fimcoder: identical pre-tokenizer to StarCoder2\n'
                 f'{indent}    res = "starcoder"\n')
        lines.insert(j + 1, block)
        open(p, "w").write("".join(lines))
        print(f"  patched {os.path.relpath(p, LLAMA)} with {chkhsh[:16]}...")
        return True
    print("  could not find the starcoder branch to patch")
    return False


def gguf():
    """HF safetensors -> GGUF f16, then quantise, then prove the tokenizer matches."""
    hf = f"{OUT}/hf"
    if not os.path.exists(f"{hf}/model.safetensors"):
        print(f"no checkpoint at {hf}; run download first")
        return False
    py = os.path.join(HERE, ".venv/bin/python")
    chk = probe_chkhsh(hf, py)
    if not chk:
        print("could not read chkhsh from the converter")
        return False
    print(f"converter computed chkhsh {chk}")
    if not patch_converter(chk):
        return False

    f16 = f"{OUT}/{TAG}-f16.gguf"
    r = sh(f"{py} {LLAMA}/convert_hf_to_gguf.py {hf} --outfile {f16} "
           f"--outtype f16 --model-name {TAG}")
    if r.returncode != 0 or not os.path.exists(f16):
        print("GGUF conversion failed")
        return False
    for q in ("Q8_0", "Q4_K_M"):
        out = f"{OUT}/{TAG}-{q.lower()}.gguf"
        sh(f"{LLAMA}/build/bin/llama-quantize {f16} {out} {q}")
    for f in sorted(os.listdir(OUT)):
        if f.endswith(".gguf"):
            print(f"  {f:<34}{os.path.getsize(f'{OUT}/{f}') / 1e6:>8.0f} MB")
    return verify_tokenizer(f16)


def verify_tokenizer(gguf_path, n_docs=120):
    """llama.cpp must produce the SAME ids as HF `tokenizers` on real code.

    This is the check that matters: a GGUF whose pre-tokenizer regex differs from the
    one the model trained with will produce plausible-looking garbage rather than an
    obvious error.
    """
    from tokenizers import Tokenizer

    src = f"{OUT}/fimeval.jsonl"
    if not os.path.exists(src):
        print("  no fimeval.jsonl, skipping tokenizer verification")
        return True
    rows = [json.loads(l) for l in open(src)][:n_docs]
    # --no-escape matters: llama-tokenize processes \n, \t, \r in its input by
    # default, so without it the comparison feeds the two tools different text and
    # reports mismatches that are entirely the harness's fault.
    tok = Tokenizer.from_file(f"{OUT}/hf/tokenizer.json")
    bad = 0
    for i, r in enumerate(rows):
        text = (r["context"] + r["prefix"] + r["middle"])[:1500]
        p = f"/tmp/_tk_{i}.txt"
        open(p, "w").write(text)
        out = subprocess.run(
            f"{LLAMA}/build/bin/llama-tokenize -m {gguf_path} -f {p} --ids --no-bos --no-escape",
            shell=True, capture_output=True, text=True)
        try:
            got = json.loads(out.stdout.strip().splitlines()[-1])
        except Exception:
            bad += 1
            continue
        want = tok.encode(text, add_special_tokens=False).ids
        if got != want:
            bad += 1
            if bad == 1:
                k = next((j for j in range(min(len(got), len(want)))
                          if got[j] != want[j]), min(len(got), len(want)))
                print(f"  MISMATCH {r['lang']} at token {k}: "
                      f"llama {got[max(0, k - 3):k + 3]} vs hf {want[max(0, k - 3):k + 3]}")
        os.remove(p)
    print(f"  tokenizer equivalence: {len(rows) - bad}/{len(rows)} documents identical")
    return bad == 0


def onnx(precisions=("fp32", "int4"), verify=True):
    """Export a cached ONNX graph per precision via onnxruntime-genai, then prove it.

    Not a hand-rolled `torch.onnx.export`: transformers 5.x dropped the legacy
    `past_key_values` tuple (`DynamicCache` has no `to_legacy_cache`), so tracing a
    KV-cached graph by hand would mean re-implementing attention. And a graph WITHOUT
    the cache is useless for generation -- decoding 64 tokens after a 1500-token
    prefill would re-run the whole prefill 64 times, ~160 s instead of ~3 s.
    onnxruntime-genai's builder emits the cached graph and a generate() API, and the
    verification below greedy-decodes with both torch and ONNX and demands identical
    token ids.
    """
    hf = f"{OUT}/hf"
    if not os.path.exists(f"{hf}/model.safetensors"):
        print(f"no checkpoint at {hf}; run download first")
        return False
    py = os.path.join(HERE, ".venv/bin/python")
    ok = True
    for p in precisions:
        d = f"{OUT}/onnx-{p}"
        os.makedirs(d, exist_ok=True)
        r = sh(f"{py} -m onnxruntime_genai.models.builder -m {hf} -o {d} "
               f"-p {p} -e cpu -c /tmp/onnx_cache 2>&1 | tail -3")
        if r.returncode != 0 or not os.path.exists(f"{d}/model.onnx"):
            print(f"  {p}: export FAILED")
            ok = False
            continue
        n = sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d))
        print(f"  onnx-{p:<6} {n / 1e6:>7.0f} MB  {sorted(os.listdir(d))}")
        if verify:
            ok = verify_onnx(d) and ok
    return ok


def verify_onnx(onnx_dir, n_new=12):
    """Greedy-decode the same FIM prompt with torch and with ONNX; ids must match.

    int4 is expected to diverge eventually -- it is a different numerical model -- so
    a mismatch there is reported rather than treated as a failure. fp32 diverging
    would mean the graph is wrong.
    """
    import numpy as np
    import onnxruntime_genai as og
    import torch
    from tokenizers import Tokenizer
    from transformers import AutoModelForCausalLM

    torch.set_num_threads(4)
    hf = f"{OUT}/hf"
    tok = Tokenizer.from_file(f"{hf}/tokenizer.json")
    # A prompt whose greedy continuation runs well past any EOS, so the check
    # actually compares n_new tokens instead of stopping after three.
    prompt = ("<|fim_prefix|>func Sum(xs []int) int {\n\ttotal := 0\n\tfor "
              "<|fim_suffix|>\n\treturn total\n}\n<|fim_middle|>")
    ids = tok.encode(prompt, add_special_tokens=False).ids

    with open(f"{hf}/specials.json") as fh:
        eot = json.load(fh)["<|endoftext|>"]
    m = AutoModelForCausalLM.from_pretrained(hf, dtype=torch.float32).eval()
    cur, ref = list(ids), []
    with torch.no_grad():
        for _ in range(n_new):
            nxt = int(m(input_ids=torch.tensor([cur])).logits[0, -1].argmax())
            if nxt == eot:
                break            # onnxruntime-genai stops at EOS and never emits it
            ref.append(nxt)
            cur.append(nxt)

    model = og.Model(onnx_dir)
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=len(ids) + n_new, do_sample=False)
    gen = og.Generator(model, params)
    gen.append_tokens(np.array(ids, dtype=np.int32))
    got = []
    # Append only when the sequence actually GREW. On EOS the generator terminates
    # without adding a token, so reading [-1] unconditionally re-reads the previous
    # one and looks like a divergence -- which is exactly what it looked like.
    prev = len(gen.get_sequence(0))
    while not gen.is_done() and len(got) < len(ref):
        gen.generate_next_token()
        seq = gen.get_sequence(0)
        if len(seq) <= prev:
            break
        prev = len(seq)
        got.append(int(seq[-1]))

    same = ref == got and len(ref) > 0
    tag = os.path.basename(onnx_dir)
    if same:
        print(f"  {tag}: greedy ids identical to torch over {n_new} tokens")
        return True
    k = next((j for j in range(min(len(ref), len(got))) if ref[j] != got[j]), 0)
    print(f"  {tag}: diverges at token {k} (torch {ref[k]} vs onnx {got[k]})"
          + ("  -- expected for int4" if "int4" in tag else "  -- UNEXPECTED for fp32"))
    return "int4" in tag


FIM_PROBE = [
    ("python", "def fib(n: int) -> int:\n    if n < 2:\n        return n\n    return ",
     "\n\nprint(fib(10))\n"),
    ("typescript",
     "export async function getUser(id: number): Promise<User> {\n  const res = await fetch(",
     "\n  return res.json();\n}\n"),
    ("go", "func Sum(xs []int) int {\n\ttotal := 0\n\tfor ",
     "\n\treturn total\n}\n"),
    ("rust", "fn main() {\n    let mut v = vec![1, 2, 3];\n    v.",
     "\n    println!(\"{:?}\", v);\n}\n"),
]


def test(quant="q8_0", n_predict=48, port=8083):
    """Greedy FIM completions on CPU, through llama.cpp, exactly as it will be used.

    Via `llama-server`, not `llama-cli`: this build of llama-cli only runs in
    conversation mode (`-no-cnv` is gone, `-st` echoes the prompt and generates
    nothing), and the server is the path an editor talks to anyway.
    """
    import urllib.request

    g = f"{OUT}/{TAG}-{quant}.gguf"
    if not os.path.exists(g):
        print(f"no {g}; run gguf first")
        return
    srv = subprocess.Popen(
        [f"{LLAMA}/build/bin/llama-server", "-m", g, "--host", "127.0.0.1",
         "--port", str(port), "-c", "4096", "-t", "4", "--no-warmup"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(120):
            try:
                urllib.request.urlopen(f"{base}/health", timeout=2).read()
                break
            except Exception:
                time.sleep(1)
        for lang, pre, suf in FIM_PROBE:
            body = json.dumps({
                "prompt": f"<|fim_prefix|>{pre}<|fim_suffix|>{suf}<|fim_middle|>",
                "n_predict": n_predict, "temperature": 0, "cache_prompt": False,
                "stop": ["<|endoftext|>", "<|file_sep|>"]}).encode()
            req = urllib.request.Request(f"{base}/completion", data=body,
                                        headers={"Content-Type": "application/json"})
            got = json.loads(urllib.request.urlopen(req, timeout=300).read())["content"]
            print(f"\n--- {lang} " + "-" * 40)
            print(pre + "\033[32m" + got.rstrip() + "\033[0m" + suf)
    finally:
        srv.terminate()
        srv.wait(timeout=30)


def config(url="", token="FIMCODER_TOKEN"):
    url = url or f"https://<your-workspace>--fimcoder-serve-server-completions.modal.run"
    ctx = 4096
    print(f"""
================ minuet-ai.nvim (lazy.nvim) ================
{{
  'milanglacier/minuet-ai.nvim',
  config = function()
    require('minuet').setup {{
      provider = 'openai_fim_compatible',
      n_completions = 1,          -- 113M model: one good completion beats three
      context_window = 12000,     -- ~3.4k tokens, inside the model's {ctx}
      provider_options = {{
        openai_fim_compatible = {{
          api_key = '{token}',    -- env var name, not the value
          name = 'fimcoder',
          end_point = '{url}',
          model = '{TAG}',
          optional = {{ max_tokens = 96, top_p = 0.9, temperature = 0.2 }},
          template = {{
            -- our own sentinels; this is why a homemade tokenizer works here
            prompt = function(pref, suff)
              return '<|fim_prefix|>' .. pref .. '<|fim_suffix|>' .. suff .. '<|fim_middle|>'
            end,
            suffix = false,
          }},
        }},
      }},
    }}
  end,
}}

================ minuet, repo-aware variant ================
-- Prepends open buffers as repo context, which is what the 29% repo-FIM
-- share of the training mix was for.
prompt = function(pref, suff)
  local ctx_parts = {{ '<|repo_name|>' .. vim.fn.fnamemodify(vim.fn.getcwd(), ':t') }}
  for _, b in ipairs(vim.api.nvim_list_bufs()) do
    local nm = vim.api.nvim_buf_get_name(b)
    if vim.api.nvim_buf_is_loaded(b) and nm ~= '' and b ~= vim.api.nvim_get_current_buf() then
      local body = table.concat(vim.api.nvim_buf_get_lines(b, 0, 120, false), '\\n')
      table.insert(ctx_parts, '<|file_sep|>' .. vim.fn.fnamemodify(nm, ':.') .. '\\n' .. body)
    end
  end
  table.insert(ctx_parts, '<|file_sep|>' .. vim.fn.expand('%:.') .. '\\n')
  return table.concat(ctx_parts) .. '<|fim_prefix|>' .. pref
         .. '<|fim_suffix|>' .. suff .. '<|fim_middle|>'
end,

================ fully local, no Modal, no credit ================
{LLAMA}/build/bin/llama-server -m {OUT}/{TAG}-q4_k_m.gguf \\
    --host 127.0.0.1 --port 8080 -c {ctx} -t 4
# then point minuet at http://127.0.0.1:8080/v1/completions and drop api_key.
""")


if __name__ == "__main__":
    # Re-exec under .venv: torch, transformers, onnxruntime-genai and gguf all live
    # there, while the system python is 3.9 and only carries the modal client.
    _V = os.path.join(HERE, ".venv/bin/python")
    if os.path.exists(_V) and not os.environ.get("FIMCODER_INVENV"):
        os.environ["FIMCODER_INVENV"] = "1"
        os.execv(_V, [_V, os.path.abspath(__file__), *sys.argv[1:]])

    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("download", "all"):
        download()
    if cmd in ("gguf", "all"):
        gguf()
    if cmd in ("onnx", "all"):
        onnx()
    if cmd in ("test", "all"):
        test()
    if cmd in ("config", "all"):
        config(*sys.argv[2:])
