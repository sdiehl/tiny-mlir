"""GPT-2 inference expressed through our compiled operation library."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .compiler import CPU, Numpy


class Model:
    def __init__(self, weights, config, ops, fused=True):
        self.ops, self.config, self.fused = ops, config, fused
        c, h = config["n_embd"], config["n_head"]
        if c % h:
            raise ValueError("Embedding width must be divisible by head count")
        expected = {
            "wte.weight": (config["vocab_size"], c),
            "wpe.weight": (config["n_positions"], c),
            "ln_f.weight": (c,),
            "ln_f.bias": (c,),
        }
        for i in range(config["n_layer"]):
            prefix = f"h.{i}."
            expected.update(
                {
                    prefix + name: shape
                    for name, shape in {
                        "ln_1.weight": (c,),
                        "ln_1.bias": (c,),
                        "ln_2.weight": (c,),
                        "ln_2.bias": (c,),
                        "attn.c_attn.weight": (c, 3 * c),
                        "attn.c_attn.bias": (3 * c,),
                        "attn.c_proj.weight": (c, c),
                        "attn.c_proj.bias": (c,),
                        "mlp.c_fc.weight": (c, 4 * c),
                        "mlp.c_fc.bias": (4 * c,),
                        "mlp.c_proj.weight": (4 * c, c),
                        "mlp.c_proj.bias": (c,),
                    }.items()
                }
            )
        self.weights = {}
        for name, shape in expected.items():
            value = np.ascontiguousarray(weights[name], dtype=np.float32)
            if value.shape != shape:
                raise ValueError(f"{name}: expected {shape}, got {value.shape}")
            self.weights[name] = ops.array(value)

    def __call__(self, tokens, trace=None):
        raw = np.asarray(tokens)
        if (
            raw.ndim != 1
            or raw.dtype.kind not in "iu"
            or not 0 < len(raw) <= self.config["n_positions"]
        ):
            raise ValueError("Expected a nonempty token sequence within the context limit")
        if np.any(raw < 0) or np.any(raw >= self.config["vocab_size"]):
            raise ValueError("Token ID outside the vocabulary")
        ids = self.ops.array(np.ascontiguousarray(raw, dtype=np.int32))
        w, op = self.weights, self.ops

        def record(name, value):
            if trace is not None:
                trace[name] = op.numpy(value).copy()
            return value

        x = record("embedding", op("embedding", ids, w["wte.weight"], w["wpe.weight"]))
        for i in range(self.config["n_layer"]):
            prefix = f"h.{i}."
            a = op("norm", x, w[prefix + "ln_1.weight"], w[prefix + "ln_1.bias"])
            qkv = op(
                "linear",
                a,
                w[prefix + "attn.c_attn.weight"],
                w[prefix + "attn.c_attn.bias"],
            )
            scores = op("scores", qkv, heads=self.config["n_head"])
            probabilities = op("softmax", scores)
            a = op("context", qkv, probabilities, heads=self.config["n_head"])
            a = op(
                "linear",
                a,
                w[prefix + "attn.c_proj.weight"],
                w[prefix + "attn.c_proj.bias"],
            )
            x = record(prefix + "attention", op("add", x, a))
            m = op("norm", x, w[prefix + "ln_2.weight"], w[prefix + "ln_2.bias"])
            m = op(
                "linear_gelu" if self.fused else "linear",
                m,
                w[prefix + "mlp.c_fc.weight"],
                w[prefix + "mlp.c_fc.bias"],
            )
            if not self.fused:
                m = op("gelu", m)
            m = op(
                "linear",
                m,
                w[prefix + "mlp.c_proj.weight"],
                w[prefix + "mlp.c_proj.bias"],
            )
            x = record(prefix + "output", op("add", x, m))
        x = op("norm", x, w["ln_f.weight"], w["ln_f.bias"])
        return record("logits", op("logits", x, w["wte.weight"]))


def random_model(seed=42):
    """Small deterministic model for tests; these weights do not generate language."""
    config = {"n_embd": 16, "n_head": 2, "n_layer": 2, "n_positions": 32, "vocab_size": 41}
    rng = np.random.default_rng(seed)

    def normal(shape):
        return rng.normal(0, 0.1, shape).astype(np.float32)

    c = config["n_embd"]
    weights = {
        "wte.weight": normal((config["vocab_size"], c)),
        "wpe.weight": normal((config["n_positions"], c)),
        "ln_f.weight": np.ones(c, np.float32),
        "ln_f.bias": np.zeros(c, np.float32),
    }
    for i in range(config["n_layer"]):
        for name in ("ln_1", "ln_2"):
            weights[f"h.{i}.{name}.weight"] = np.ones(c, np.float32)
            weights[f"h.{i}.{name}.bias"] = np.zeros(c, np.float32)
        for name, shape in [
            ("attn.c_attn", (c, 3 * c)),
            ("attn.c_proj", (c, c)),
            ("mlp.c_fc", (c, 4 * c)),
            ("mlp.c_proj", (4 * c, c)),
        ]:
            weights[f"h.{i}.{name}.weight"] = normal(shape)
            weights[f"h.{i}.{name}.bias"] = normal((shape[1],))
    return weights, config


def load_checkpoint(directory):
    from safetensors.numpy import load_file
    from tokenizers import Tokenizer

    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    if config.get("activation_function") != "gelu_new" or config.get("layer_norm_epsilon") != 1e-5:
        raise ValueError("This example implements GPT-2 gelu_new and epsilon=1e-5")
    return (
        load_file(str(directory / "model.safetensors")),
        config,
        Tokenizer.from_file(str(directory / "tokenizer.json")),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=("numpy", "cpu", "cuda"), default="cpu")
    p.add_argument("--model-dir")
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--tokens", type=int, default=4)
    p.add_argument("--cache", default=".mlir-cache")
    p.add_argument("--chip", default="sm_75")
    p.add_argument("--libdevice")
    p.add_argument(
        "--verify",
        action="store_true",
        help="Compare the initial forward pass with NumPy",
    )
    p.add_argument("--unfused", action="store_true")
    args = p.parse_args()
    if args.tokens < 1:
        p.error("--tokens must be positive")
    if args.backend == "cuda":
        from .cuda_runtime import CUDA

        ops = CUDA(args.cache, args.chip, args.libdevice)
    else:
        ops = CPU(args.cache) if args.backend == "cpu" else Numpy()
    try:
        if args.model_dir:
            weights, config, tokenizer = load_checkpoint(args.model_dir)
            ids = tokenizer.encode(args.prompt).ids
        else:
            weights, config = random_model()
            tokenizer, ids = None, [1, 2, 3]
            print("Random-weight smoke test; token IDs are not meaningful text.")
        if not ids or len(ids) + args.tokens > config["n_positions"]:
            raise ValueError("Prompt plus generated tokens must fit the model context")
        model = Model(weights, config, ops, fused=not args.unfused)
        start = time.perf_counter()
        trace = {} if args.verify else None
        logits = ops.numpy(model(ids, trace))
        print(
            f"Initial forward pass (includes cold compilation): {time.perf_counter() - start:.3f}s"
        )
        if args.verify:
            reference_trace = {}
            reference = Model(weights, config, Numpy())
            reference(ids, reference_trace)
            for name, expected in reference_trace.items():
                np.testing.assert_allclose(
                    trace[name], expected, rtol=3e-3, atol=3e-3, err_msg=name
                )
            print("Embedding, block outputs and final logits agree with NumPy (rtol=atol=3e-3).")
        generated = []
        for step in range(args.tokens):
            if step:
                logits = ops.numpy(model(ids))
            token = int(np.argmax(logits[0]))
            ids.append(token)
            generated.append(token)
            if token == config.get("eos_token_id"):
                break
        print(tokenizer.decode(ids) if tokenizer else generated)
    finally:
        ops.close()


if __name__ == "__main__":
    main()
