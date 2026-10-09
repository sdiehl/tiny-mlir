"""GPT-2 expressed as compositions of compiled tensor functions."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
from safetensors.numpy import load_file
from tokenizers import Tokenizer

from .checkpoint import (
    CONFIG_FILENAME,
    DEFAULT_MODEL_DIRECTORY,
    TOKENIZER_FILENAME,
    WEIGHTS_FILENAME,
)
from .expr import gather
from .jit import jit
from .ops import LAYER_NORM_EPSILON, QKV_COMPONENTS, attention, gelu, layer_norm, linear


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


ACTIVATION_FUNCTION = "gelu_new"
FEEDFORWARD_EXPANSION = 4


@dataclass(frozen=True)
class ModelConfig:
    width: int
    heads: int
    layers: int
    context: int
    vocabulary: int

    @classmethod
    def from_dict(cls, config):
        return cls(
            config["n_embd"],
            config["n_head"],
            config["n_layer"],
            config["n_positions"],
            config["vocab_size"],
        )

    def __post_init__(self):
        if min(self.width, self.heads, self.layers, self.context, self.vocabulary) <= 0:
            raise ValueError("Model dimensions must be positive")
        if self.width % self.heads:
            raise ValueError("Embedding width must be divisible by head count")


class Parameters(NamedTuple):
    weight: np.ndarray
    bias: np.ndarray


class Block(NamedTuple):
    attention_norm: Parameters
    attention_in: Parameters
    attention_out: Parameters
    feedforward_norm: Parameters
    feedforward_in: Parameters
    feedforward_out: Parameters


class Model:
    def __init__(self, weights, config):
        self.config = ModelConfig.from_dict(config)
        width = self.config.width

        def array(name, shape):
            value = np.asarray(weights[name])
            if value.shape != shape or value.dtype != np.float32:
                raise ValueError(f"{name} must have shape {shape} and dtype float32")
            return np.ascontiguousarray(value)

        def parameters(prefix, shape):
            return Parameters(
                array(f"{prefix}.weight", shape), array(f"{prefix}.bias", (shape[-1],))
            )

        self.words = array("wte.weight", (self.config.vocabulary, width))
        self.positions = array("wpe.weight", (self.config.context, width))
        self.final_norm = parameters("ln_f", (width,))
        hidden_width = FEEDFORWARD_EXPANSION * width
        self.blocks = tuple(
            Block(
                parameters(f"h.{index}.ln_1", (width,)),
                parameters(f"h.{index}.attn.c_attn", (width, QKV_COMPONENTS * width)),
                parameters(f"h.{index}.attn.c_proj", (width, width)),
                parameters(f"h.{index}.ln_2", (width,)),
                parameters(f"h.{index}.mlp.c_fc", (width, hidden_width)),
                parameters(f"h.{index}.mlp.c_proj", (hidden_width, width)),
            )
            for index in range(self.config.layers)
        )

    def __call__(self, tokens, trace=None):
        ids = np.asarray(tokens)
        if (
            ids.ndim != 1
            or not np.issubdtype(ids.dtype, np.integer)
            or not 0 < len(ids) <= self.config.context
        ):
            raise ValueError("Expected a nonempty integer token sequence within the context limit")
        if np.any(ids < 0) or np.any(ids >= self.config.vocabulary):
            raise ValueError("Token ID outside the vocabulary")

        def record(name, value):
            if trace is not None:
                trace[name] = value.copy()
            return value

        x = record("embedding", embed(ids.astype(np.int32), self.words, self.positions))
        for index, block in enumerate(self.blocks):
            x = record(
                f"h.{index}.attention",
                attend(
                    x,
                    *block.attention_norm,
                    *block.attention_in,
                    *block.attention_out,
                    heads=self.config.heads,
                ),
            )
            x = record(
                f"h.{index}.output",
                feedforward(
                    x,
                    *block.feedforward_norm,
                    *block.feedforward_in,
                    *block.feedforward_out,
                ),
            )
        return record("logits", project(x, *self.final_norm, self.words))


def load_model(directory: str | Path = DEFAULT_MODEL_DIRECTORY) -> tuple[Model, Tokenizer]:
    directory = Path(directory)
    config = json.loads((directory / CONFIG_FILENAME).read_text(encoding="utf-8"))
    if (
        config.get("activation_function") != ACTIVATION_FUNCTION
        or config.get("layer_norm_epsilon") != LAYER_NORM_EPSILON
    ):
        raise ValueError(
            f"Expected GPT-2 {ACTIVATION_FUNCTION} activation and epsilon={LAYER_NORM_EPSILON}"
        )
    model = Model(load_file(str(directory / WEIGHTS_FILENAME)), config)
    return model, Tokenizer.from_file(str(directory / TOKENIZER_FILENAME))


def generate(model: Model, tokenizer: Tokenizer, prompt: str, max_tokens: int) -> str:
    ids = tokenizer.encode(prompt).ids
    if max_tokens < 0 or not ids or len(ids) + max_tokens > model.config.context:
        raise ValueError("Prompt and generation must fit the model context")
    for _ in range(max_tokens):
        ids.append(int(np.argmax(model(ids)[0])))
    return tokenizer.decode(ids)
