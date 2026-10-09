"""Run inside tiny-gpt2's environment to capture its unmodified NumPy operations."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

p = argparse.ArgumentParser()
p.add_argument("reference", type=Path)
p.add_argument("model", type=Path)
p.add_argument("output", type=Path)
args = p.parse_args()
sys.path.insert(0, str(args.reference))
from tinygpt2 import gpt2_ops as ops
from tinygpt2.encoder import get_encoder
from tinygpt2.gpt2_run import gpt2
from tinygpt2.gpt2_tensors import (
    AttentionParams,
    LayerNormParams,
    LinearParams,
    MLPParams,
    ModelParams,
    TransformerBlockParams,
)

# Adapt the pinned checkpoint to the reference's parameter dataclasses. This avoids
# its loader's network request; the numerical operations are imported unchanged.
w = load_file(str(args.model / "model.safetensors"))
c = json.loads((args.model / "config.json").read_text())


def norm(p):
    return LayerNormParams(w[p + ".weight"], w[p + ".bias"])


def linear(p):
    return LinearParams(w[p + ".weight"], w[p + ".bias"])


blocks = []
for i in range(c["n_layer"]):
    p = f"h.{i}"
    blocks.append(
        TransformerBlockParams(
            ln_1=norm(p + ".ln_1"),
            ln_2=norm(p + ".ln_2"),
            mlp=MLPParams(c_fc=linear(p + ".mlp.c_fc"), c_proj=linear(p + ".mlp.c_proj")),
            attn=AttentionParams(
                c_attn=linear(p + ".attn.c_attn"), c_proj=linear(p + ".attn.c_proj")
            ),
        )
    )
params = ModelParams(wte=w["wte.weight"], wpe=w["wpe.weight"], blocks=blocks, ln_f=norm("ln_f"))
encoder = get_encoder("", str(args.reference / "model"))
prompt = "Alan Turing theorized that computers would one day become"
ids = encoder.encode(prompt)
results = {"input_ids": np.array(ids)}
tokens = []
for step in range(10):
    # Capture both residual outputs using the original operation implementations.
    x = params.wte[ids] + params.wpe[range(len(ids))]
    results[f"{step}/embedding"] = x
    for i, block in enumerate(params.blocks):
        a = ops.layer_norm(x, block.ln_1.g, block.ln_1.b)
        x = x + ops.mha(a, block.attn.c_attn, block.attn.c_proj, c["n_head"])
        results[f"{step}/h.{i}.attention"] = x
        m = ops.layer_norm(x, block.ln_2.g, block.ln_2.b)
        x = x + ops.ffn(
            m, block.mlp.c_fc.w, block.mlp.c_fc.b, block.mlp.c_proj.w, block.mlp.c_proj.b
        )
        results[f"{step}/h.{i}.output"] = x
    logits = ops.layer_norm(x, params.ln_f.g, params.ln_f.b) @ params.wte.T
    # Also run the reference's complete gpt2 function, so the trace adapter itself
    # cannot silently become a different reference model.
    direct = gpt2(ids, params, c["n_head"])
    np.testing.assert_array_equal(logits, direct)
    results[f"{step}/logits"] = direct[-1:]
    token = int(direct[-1].argmax())
    tokens.append(token)
    ids.append(token)
results["output_ids"] = np.array(tokens)
results["completion"] = np.array(encoder.decode(tokens))
results["numpy_version"] = np.array(np.__version__)
np.savez(args.output, **results)
print("NumPy reference:", repr(encoder.decode(tokens)), flush=True)
