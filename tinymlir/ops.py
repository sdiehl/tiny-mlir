"""Transformer mathematics, composed from tensor expressions."""

import numpy as np
from .expr import causal_mask, exp, sqrt, tanh


def gelu(x):
    return 0.5 * x * (1 + tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def softmax(x):
    numerator = exp(x - x.max(axis=-1, keepdims=True))
    return numerator / numerator.sum(axis=-1, keepdims=True)


def layer_norm(x, gain, bias):
    centered = x - x.mean(axis=-1, keepdims=True)
    variance = (centered * centered).mean(axis=-1, keepdims=True)
    return gain * centered / sqrt(variance + 1e-5) + bias


def linear(x, weight, bias):
    return x @ weight + bias


def attention(x, qkv_weight, qkv_bias, out_weight, out_bias, heads):
    packed = linear(x, qkv_weight, qkv_bias)
    length, width = x.shape
    depth = width // heads
    if width % heads:
        raise ValueError('Embedding width must be divisible by head count')
    q, k, v = [packed[:, i*width:(i+1)*width].reshape((length, heads, depth))
               .transpose((1, 0, 2)) for i in range(3)]
    scores = (q @ k.transpose((0, 2, 1))) / np.sqrt(depth)
    probabilities = softmax(scores + causal_mask(length, packed.dtype))
    context = (probabilities @ v).transpose((1, 0, 2)).reshape((length, width))
    return linear(context, out_weight, out_bias)
