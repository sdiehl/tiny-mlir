"""GPT-2 expressed as compositions of compiled tensor functions."""

import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file
from tokenizers import Tokenizer

from .expr import gather
from .jit import jit
from .ops import attention, gelu, layer_norm, linear


@jit
def embed(ids, words, positions):
    return gather(words, ids) + positions[: ids.shape[0]]


@jit
def attend(x, gain, bias, qkv_weight, qkv_bias, out_weight, out_bias, *, heads):
    normalized = layer_norm(x, gain, bias)
    return x + attention(normalized, qkv_weight, qkv_bias, out_weight, out_bias, heads)


@jit
def feedforward(x, gain, bias, up_weight, up_bias, down_weight, down_bias):
    normalized = layer_norm(x, gain, bias)
    hidden = gelu(linear(normalized, up_weight, up_bias))
    return x + linear(hidden, down_weight, down_bias)


@jit
def project(x, gain, bias, words):
    return layer_norm(x[-1:], gain, bias) @ words.T


class Model:
    def __init__(self, weights, config):
        self.config = config
        width, heads = config["n_embd"], config["n_head"]
        if heads <= 0 or width % heads:
            raise ValueError("Embedding width must be divisible by head count")
        shapes = {
            "wte.weight": (config["vocab_size"], width),
            "wpe.weight": (config["n_positions"], width),
            "ln_f.weight": (width,),
            "ln_f.bias": (width,),
        }
        for i in range(config["n_layer"]):
            for name, shape in [
                ("ln_1", (width,)),
                ("ln_2", (width,)),
                ("attn.c_attn", (width, 3 * width)),
                ("attn.c_proj", (width, width)),
                ("mlp.c_fc", (width, 4 * width)),
                ("mlp.c_proj", (4 * width, width)),
            ]:
                shapes[f"h.{i}.{name}.weight"] = shape
                shapes[f"h.{i}.{name}.bias"] = (shape[-1],)
        self.weights = {}
        for name, shape in shapes.items():
            value = np.asarray(weights[name])
            if value.shape != shape or value.dtype != np.float32:
                raise ValueError(f"{name} must have shape {shape} and dtype float32")
            self.weights[name] = np.ascontiguousarray(value)

    def __call__(self, tokens, trace=None):
        ids = np.asarray(tokens)
        if (
            ids.ndim != 1
            or ids.dtype.kind not in "iu"
            or not 0 < len(ids) <= self.config["n_positions"]
        ):
            raise ValueError("Expected a nonempty integer token sequence within the context limit")
        if np.any(ids < 0) or np.any(ids >= self.config["vocab_size"]):
            raise ValueError("Token ID outside the vocabulary")
        w = self.weights

        def record(name, value):
            if trace is not None:
                trace[name] = value.copy()
            return value

        x = record("embedding", embed(ids.astype(np.int32), w["wte.weight"], w["wpe.weight"]))
        for i in range(self.config["n_layer"]):
            p = f"h.{i}."
            x = record(
                p + "attention",
                attend(
                    x,
                    w[p + "ln_1.weight"],
                    w[p + "ln_1.bias"],
                    w[p + "attn.c_attn.weight"],
                    w[p + "attn.c_attn.bias"],
                    w[p + "attn.c_proj.weight"],
                    w[p + "attn.c_proj.bias"],
                    heads=self.config["n_head"],
                ),
            )
            x = record(
                p + "output",
                feedforward(
                    x,
                    w[p + "ln_2.weight"],
                    w[p + "ln_2.bias"],
                    w[p + "mlp.c_fc.weight"],
                    w[p + "mlp.c_fc.bias"],
                    w[p + "mlp.c_proj.weight"],
                    w[p + "mlp.c_proj.bias"],
                ),
            )
        return record("logits", project(x, w["ln_f.weight"], w["ln_f.bias"], w["wte.weight"]))


def load_model(directory="model"):
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    if config.get("activation_function") != "gelu_new" or config.get("layer_norm_epsilon") != 1e-5:
        raise ValueError("Expected GPT-2 gelu_new activation and layer_norm_epsilon=1e-5")
    model = Model(load_file(str(directory / "model.safetensors")), config)
    return model, Tokenizer.from_file(str(directory / "tokenizer.json"))


def generate(model, tokenizer, prompt, max_tokens):
    ids = tokenizer.encode(prompt).ids
    if max_tokens < 0 or not ids or len(ids) + max_tokens > model.config["n_positions"]:
        raise ValueError("Prompt and generation must fit the model context")
    for _ in range(max_tokens):
        ids.append(int(np.argmax(model(ids)[0])))
    return tokenizer.decode(ids)
