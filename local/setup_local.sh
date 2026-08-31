#!/usr/bin/env bash
# Run fimcoder-113m on your own machine, on the GPU, and prove it works.
#
#   ./setup_local.sh /path/to/fimcoder-113m-q8_0.gguf
#
# Runtime is picked in this order: an `llama-server` already on PATH, then the
# official CUDA docker image, then it tells you how to build. It then sends the
# exact request shape minuet-ai.nvim sends and prints MEASURED latency.
set -euo pipefail

GGUF="${1:-}"
PORT="${PORT:-8080}"
CTX="${CTX:-4096}"
[ -f "$GGUF" ] || { echo "usage: $0 /path/to/fimcoder-113m-q8_0.gguf"; exit 1; }
GGUF="$(cd "$(dirname "$GGUF")" && pwd)/$(basename "$GGUF")"
have() { command -v "$1" >/dev/null 2>&1; }

echo "model: $GGUF ($(du -h "$GGUF" | cut -f1))"
if have nvidia-smi; then
  echo "gpu:   $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
else
  echo "gpu:   nvidia-smi not found -- this will run on CPU and be ~10x slower"
fi

if have llama-server; then
  echo "runtime: llama-server on PATH"
  # -ngl 99 puts every layer on the GPU. The model is ~122 MB, so an 8 GB card is
  # nowhere close to full -- context and batch are what use VRAM here, not weights.
  llama-server -m "$GGUF" --host 127.0.0.1 --port "$PORT" -c "$CTX" \
    -ngl 99 --no-warmup >/tmp/fimcoder-server.log 2>&1 &
  echo "native $!" > /tmp/fimcoder.rt
elif have docker && docker info >/dev/null 2>&1; then
  echo "runtime: ghcr.io/ggml-org/llama.cpp:server-cuda (docker)"
  docker rm -f fimcoder >/dev/null 2>&1 || true
  docker run -d --rm --gpus all --name fimcoder \
    -v "$(dirname "$GGUF")":/models -p "$PORT":8080 \
    ghcr.io/ggml-org/llama.cpp:server-cuda \
    -m "/models/$(basename "$GGUF")" -c "$CTX" -ngl 99 --host 0.0.0.0 >/dev/null
  echo "docker fimcoder" > /tmp/fimcoder.rt
else
  cat <<'MSG'
No llama-server and no usable docker. Build llama.cpp with CUDA once:

  git clone --depth 1 https://github.com/ggml-org/llama.cpp && cd llama.cpp
  cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
  cmake --build build -j --target llama-server
  sudo install build/bin/llama-server /usr/local/bin/

Then re-run this script. (Or: docker + nvidia-container-toolkit and re-run.)
MSG
  exit 1
fi

printf 'waiting for the server'
for _ in $(seq 1 180); do
  curl -sf -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { echo " up"; break; }
  printf '.'; sleep 1
done
curl -sf -m 5 "http://127.0.0.1:$PORT/health" >/dev/null || {
  echo " FAILED"; tail -25 /tmp/fimcoder-server.log 2>/dev/null
  docker logs fimcoder 2>&1 | tail -25 || true; exit 1; }

U="http://127.0.0.1:$PORT/v1/completions"
fim() {  # fim <prefix> <suffix> <max_tokens>
  python3 - "$1" "$2" "$3" "$U" <<'PY'
import json, sys, time, urllib.request
pre, suf, n, url = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
body = json.dumps({
    "model": "fimcoder-113m",
    "prompt": f"<|fim_prefix|>{pre}<|fim_suffix|>{suf}<|fim_middle|>",
    "max_tokens": n, "temperature": 0,
    "stop": ["<|endoftext|>", "<|file_sep|>"], "stream": False,
}).encode()
req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
t0 = time.time()
r = json.loads(urllib.request.urlopen(req, timeout=120).read())
ms = (time.time() - t0) * 1000
print(f"{ms:.0f}|{r['usage']['prompt_tokens']}|{r['choices'][0]['text']}")
PY
}

echo
echo "=== does it complete code? (greedy, temperature 0)"
declare -a OK
for probe in \
  'rust|fn main() {\n    let mut v = vec![1, 2, 3];\n    v.|\n    println!("{:?}", v);\n}\n' \
  'python|import os\n\ndef read(path):\n    with open(path) as fh:\n        return fh.|\n\nprint(read("a.txt"))\n' \
  'typescript|export function sum(xs: number[]): number {\n  return xs.|\n}\n' \
  'go|func Max(a, b int) int {\n\tif a |\n\t\treturn a\n\t}\n\treturn b\n}\n'
do
  lang="${probe%%|*}"; rest="${probe#*|}"; pre="${rest%%|*}"; suf="${rest#*|}"
  out="$(fim "$(printf '%b' "$pre")" "$(printf '%b' "$suf")" 24)"
  ms="${out%%|*}"; rest2="${out#*|}"; ptok="${rest2%%|*}"; txt="${rest2#*|}"
  printf '  %-11s %5s ms  %3s prompt tokens   ->  %s\n' "$lang" "$ms" "$ptok" "$txt"
  OK+=("$ms")
done

echo
echo "=== latency at a realistic editor context (~2600-token window)"
python3 - "$U" <<'PYLAT'
import json, sys, time, urllib.request
url = sys.argv[1]
pre = ("# padding line to build a realistic window\n" * 260
       + "def total(xs):\n    s = 0\n    for x in xs:\n        s += ")
suf = "\n    return s\n"
prompt = f"<|fim_prefix|>{pre}<|fim_suffix|>{suf}<|fim_middle|>"

def call(n, cache):
    body = json.dumps({"prompt": prompt, "max_tokens": n, "temperature": 0,
                       "cache_prompt": cache, "stream": False}).encode()
    req = urllib.request.Request(url, data=body,
                                headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(req, timeout=300).read())
    u = r["usage"]
    return (time.time() - t0) * 1000, u["prompt_tokens"], u["completion_tokens"]

for n in (16, 48, 96):
    cold, ptok, ctok = call(n, False)     # full prefill, worst case
    call(n, True)                         # prime the prefix cache
    warm, _, _ = call(n, True)            # what you feel typing in one file
    print(f"  budget {n:>3} tok -> emitted {ctok:>2}:  prompt {ptok}"
          f"   cold {cold:>6.0f} ms   warm {warm:>5.0f} ms")
print("")
print("  cold = full prefill, i.e. the first request in a new file.")
print("  warm = llama.cpp reused the cached prefix; that is the common case while")
print("         typing in one buffer, and it is the number you will actually feel.")
print("  'emitted' is what the model chose to write before stopping -- it ends its own")
print("  completions, so a larger max_tokens budget does not cost more time.")
PYLAT
cat <<MSG

=== the server is running on http://127.0.0.1:$PORT
Point minuet at it with the config in minuet.lua (no API key needed for localhost).

stop it with:   $( [ "$(cut -d' ' -f1 /tmp/fimcoder.rt)" = docker ] && echo 'docker rm -f fimcoder' || echo "kill \$(cut -d' ' -f2 /tmp/fimcoder.rt)" )
server log:     /tmp/fimcoder-server.log
MSG
