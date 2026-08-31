#!/usr/bin/env bash
# Everything between "training finished" and "it works in your editor".
# Local and free except the last step (a few minutes of L4 to redeploy the endpoint).
set -euo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/.local/bin:$PATH"
TAG="${FIMCODER_TAG:-fimcoder-113m}"
export FIMCODER_TAG="$TAG"

echo "=== 1/6 download the checkpoint off the Modal volume"
python3 finish.py download

echo "=== 2/6 GGUF f16 / q8_0 / q4_k_m, and prove tokenizer equivalence vs HF"
python3 finish.py gguf

echo "=== 3/6 ONNX fp32 / int4, and prove greedy ids match torch"
python3 finish.py onnx

echo "=== 4/6 bits per byte, fresh 2024+ vs pre-2023, same docs as the Qwen baseline"
.venv/bin/python eval_local.py bpb "out/$TAG/hf" 1024 "$TAG"

echo "=== 5/6 FIM exact-match / edit-sim / first-line, and editor latency"
.venv/bin/python eval_local.py fim "out/$TAG/$TAG-q8_0.gguf" 300   # time-boxed at 25 min
.venv/bin/python eval_local.py latency "out/$TAG/$TAG-q8_0.gguf"

echo "=== 6/6 a local FIM completion you can read, then the editor config"
python3 finish.py test
python3 finish.py config "https://amanmstudies--fimcoder-serve-server-completions.modal.run"

echo
echo "done. artifacts in out/$TAG:"
ls -la "out/$TAG" | sed 's/^/  /'
