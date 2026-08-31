"""
OpenAI-compatible FIM endpoint for fimcoder-300m, shaped for minuet-ai.nvim.

    modal deploy modal_serve.py          # then `modal app stop fimcoder-serve` when done

Cost safety, deliberately:
  * L4, not H100. Decoding a 608 MB model is bandwidth-bound, and an L4 at
    $0.80/hr does 128 tokens in ~250 ms -- fine for autocomplete, 5x cheaper.
  * `min_containers=0` and a 2-minute scaledown window, so an endpoint left
    deployed costs nothing while nobody is typing.
  * `max_containers=1`, so a runaway client cannot fan out onto more GPUs.
  * A bearer token is REQUIRED. A public unauthenticated GPU endpoint is a bad
    default, whatever the blast radius.
"""

import json
import os
import time

import modal

app = modal.App("fimcoder-serve")
vol = modal.Volume.from_name("fimcoder", create_if_missing=True)
VOL = "/vol"
TAG = os.environ.get("FIMCODER_TAG", "fimcoder-113m")

img = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.12.1",
        "transformers==5.16.1",
        "numpy==2.3.4",
        "fastapi[standard]==0.121.1",
    )
    .env({"TOKENIZERS_PARALLELISM": "false", "FIMCODER_TAG": TAG})
    .add_local_python_source("fimlib")
)

# Set with:  modal secret create fimcoder-token FIMCODER_TOKEN=<something-random>
secret = modal.Secret.from_name("fimcoder-token", required_keys=["FIMCODER_TOKEN"])

# fastapi is present in the container but not in the local CLI environment, and this
# module is imported by both. A bare `authorization: str = ""` parameter is read by
# FastAPI as a QUERY parameter, not a header, so it was always empty and every
# authenticated request came back 401.
try:
    from fastapi import Header as _Header

    _AUTH = _Header(default="")
except ImportError:
    _AUTH = ""


@app.cls(image=img, gpu="L4", volumes={VOL: vol}, secrets=[secret], cpu=2.0,
         memory=8192, min_containers=0, max_containers=1, scaledown_window=120,
         timeout=600)
class Server:
    # A plain class attribute, not modal.parameter(): a parameterised class expects
    # the parameter threaded through the web-endpoint URL, so a bare GET silently
    # used the default and the container crash-looped on a path that never existed.
    # Set FIMCODER_TAG before `modal deploy` instead.
    tag = TAG

    @modal.enter()
    def load(self):
        import torch
        from tokenizers import Tokenizer
        from transformers import LlamaForCausalLM

        import fimlib as F

        d = f"{VOL}/runs/{self.tag}/hf"
        if not os.path.exists(f"{d}/tokenizer.json"):
            runs = sorted(p for p in os.listdir(f"{VOL}/runs")
                          if os.path.exists(f"{VOL}/runs/{p}/hf/tokenizer.json"))
            print(f"[serve] {d} missing; available: {runs}")
            if not runs:
                raise FileNotFoundError(f"no exported checkpoint under {VOL}/runs")
            self.tag = runs[-1]
            d = f"{VOL}/runs/{self.tag}/hf"
            print(f"[serve] falling back to {self.tag}")
        self.tok = Tokenizer.from_file(f"{d}/tokenizer.json")
        self.sp = F.Specials(d)
        self.model = LlamaForCausalLM.from_pretrained(
            d, dtype=torch.bfloat16).to("cuda").eval()
        self.model.config.use_cache = True
        self.ctx = self.model.config.max_position_embeddings
        # One warm decode so the first real request is not the one that pays for
        # kernel autotuning.
        list(self._gen([self.sp.pre, self.sp.suf, self.sp.mid], 4, 0.0, []))
        print(f"[serve] {self.tag} loaded, context {self.ctx}")

    def _fit(self, prompt_ids):
        """Keep the tokens nearest the cursor. minuet's default context window is
        16000 characters, well past our 4096 tokens, so the server truncates rather
        than trusting the client."""
        room = self.ctx - 8
        if len(prompt_ids) <= room:
            return prompt_ids
        sp = self.sp
        if sp.mid in prompt_ids and sp.suf in prompt_ids and sp.pre in prompt_ids:
            i, j, k = (prompt_ids.index(sp.pre), prompt_ids.index(sp.suf),
                       prompt_ids.index(sp.mid))
            head, pre, suf = prompt_ids[:i], prompt_ids[i + 1 : j], prompt_ids[j + 1 : k]
            head = head[-(room // 8):]
            budget = room - len(head) - 3
            wp = (budget * 2) // 3
            ws = budget - wp
            if len(suf) < ws:
                wp += ws - len(suf)
            return [*head, sp.pre, *pre[-wp:], sp.suf, *suf[:ws], sp.mid]
        return prompt_ids[-room:]

    def _gen(self, ids, max_new, temperature, stop_ids):
        import torch

        x = torch.tensor([ids], device="cuda")
        past = None
        with torch.no_grad():
            for _ in range(max_new):
                o = self.model(input_ids=x, past_key_values=past, use_cache=True)
                past = o.past_key_values
                logits = o.logits[0, -1].float()
                if temperature and temperature > 0:
                    p = torch.softmax(logits / temperature, -1)
                    nxt = int(torch.multinomial(p, 1))
                else:
                    nxt = int(logits.argmax())
                if nxt == self.sp.eot or nxt in stop_ids:
                    return
                x = torch.tensor([[nxt]], device="cuda")
                yield nxt

    def _prompt_ids(self, body):
        """Accept both shapes minuet can post: a pre-assembled FIM string, or a
        `prompt`/`suffix` pair that we wrap in our own sentinels."""
        sp = self.sp
        e = lambda s: self.tok.encode(s, add_special_tokens=False).ids
        prompt = body.get("prompt") or ""
        suffix = body.get("suffix")
        if isinstance(prompt, list):
            prompt = prompt[0] if prompt else ""
        if isinstance(suffix, str) and suffix:
            return [sp.pre, *e(prompt), sp.suf, *e(suffix), sp.mid]
        if "<|fim_prefix|>" in prompt or "<|fim_middle|>" in prompt:
            return e(prompt)                     # client built the FIM prompt itself
        return [sp.pre, *e(prompt), sp.suf, sp.mid]

    @modal.fastapi_endpoint(method="POST", docs=True)
    def completions(self, body: dict, authorization: str = _AUTH):
        """OpenAI /v1/completions, streaming or not. minuet posts here."""
        from fastapi import HTTPException
        from fastapi.responses import StreamingResponse

        want = os.environ["FIMCODER_TOKEN"]
        if authorization.removeprefix("Bearer ").strip() != want:
            raise HTTPException(status_code=401, detail="bad or missing bearer token")

        ids = self._fit(self._prompt_ids(body))
        max_new = int(body.get("max_tokens") or 96)
        temp = float(body.get("temperature") or 0.0)
        stop = body.get("stop") or []
        stop_ids = set()
        for s in stop if isinstance(stop, list) else [stop]:
            t = self.tok.encode(str(s), add_special_tokens=False).ids
            if len(t) == 1:
                stop_ids.add(t[0])
        created = int(time.time())
        base = {"id": f"cmpl-{created}", "object": "text_completion",
                "created": created, "model": self.tag}

        if not body.get("stream"):
            t0 = time.time()
            toks = list(self._gen(ids, max_new, temp, stop_ids))
            text = self.tok.decode(toks, skip_special_tokens=True)
            for s in stop if isinstance(stop, list) else []:
                if s and s in text:
                    text = text.split(s)[0]
            return {**base, "choices": [{"index": 0, "text": text,
                                         "finish_reason": "stop", "logprobs": None}],
                    "usage": {"prompt_tokens": len(ids), "completion_tokens": len(toks),
                              "total_tokens": len(ids) + len(toks),
                              "ms": round((time.time() - t0) * 1000)}}

        def sse():
            buf = []
            for t in self._gen(ids, max_new, temp, stop_ids):
                buf.append(t)
                piece = self.tok.decode(buf, skip_special_tokens=True)
                if piece:
                    buf = []
                    yield "data: " + json.dumps(
                        {**base, "choices": [{"index": 0, "text": piece,
                                              "finish_reason": None}]}) + "\n\n"
            yield "data: " + json.dumps(
                {**base, "choices": [{"index": 0, "text": "",
                                      "finish_reason": "stop"}]}) + "\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream")

    @modal.fastapi_endpoint(method="GET")
    def health(self):
        return {"ok": True, "model": self.tag, "context": self.ctx}
