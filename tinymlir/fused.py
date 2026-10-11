"""The NumPy reference forward pass traced unchanged into one kernel.

The main path in model.py runs one compiled function per sublayer. This module is the
alternative: the whole model is a single tensor graph, so fusion and the matmul
substitution see every block at once.
"""

import numpy as np

from .jit import jit


@jit
def gpt2(inputs, params, *, n_head):
    """tinygpt2/gpt2_minimal.py as written. NumPy calls trace through the array protocols."""
    x = params.words[inputs] + params.positions[range(len(inputs))]
    seq_len, embedding_dim = x.shape
    head_size = embedding_dim // n_head

    for block in params.blocks:
        ln1_mean = np.mean(x, axis=-1, keepdims=True)
        ln1_variance = np.var(x, axis=-1, keepdims=True)
        ln1_normalized = (x - ln1_mean) / np.sqrt(ln1_variance + 1e-5)
        ln1_output = block.attention_norm.weight * ln1_normalized + block.attention_norm.bias

        qkv_proj = ln1_output @ block.attention_in.weight + block.attention_in.bias
        q_proj, k_proj, v_proj = np.split(qkv_proj, 3, axis=-1)

        q_heads = q_proj.reshape(seq_len, n_head, head_size)
        k_heads = k_proj.reshape(seq_len, n_head, head_size)
        v_heads = v_proj.reshape(seq_len, n_head, head_size)

        q_heads_t = q_heads.transpose(1, 0, 2)
        k_heads_t = k_heads.transpose(1, 0, 2)
        v_heads_t = v_heads.transpose(1, 0, 2)

        attention_scores = (q_heads_t @ k_heads_t.transpose(0, 2, 1)) / np.sqrt(head_size)

        causal_mask = np.triu(np.ones((seq_len, seq_len), dtype=x.dtype) * -np.inf, k=1)
        attention_scores = attention_scores + causal_mask

        exp_scores = np.exp(attention_scores - np.max(attention_scores, axis=-1, keepdims=True))
        attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

        weighted_values = attention_weights @ v_heads_t

        merged_heads = weighted_values.transpose(1, 0, 2).reshape(seq_len, embedding_dim)

        mha_output = merged_heads @ block.attention_out.weight + block.attention_out.bias

        x = x + mha_output

        ln2_mean = np.mean(x, axis=-1, keepdims=True)
        ln2_variance = np.var(x, axis=-1, keepdims=True)
        ln2_normalized = (x - ln2_mean) / np.sqrt(ln2_variance + 1e-5)
        ln2_output = block.feedforward_norm.weight * ln2_normalized + block.feedforward_norm.bias

        fc_output = ln2_output @ block.feedforward_in.weight + block.feedforward_in.bias

        gelu_output = (
            0.5
            * fc_output
            * (1 + np.tanh(np.sqrt(2 / np.pi) * (fc_output + 0.044715 * fc_output**3)))
        )

        ffn_output = gelu_output @ block.feedforward_out.weight + block.feedforward_out.bias

        x = x + ffn_output

    lnf_mean = np.mean(x, axis=-1, keepdims=True)
    lnf_variance = np.var(x, axis=-1, keepdims=True)
    lnf_normalized = (x - lnf_mean) / np.sqrt(lnf_variance + 1e-5)
    x_normalized_final = params.final_norm.weight * lnf_normalized + params.final_norm.bias

    logits = x_normalized_final @ params.words.T

    return logits


def logits(ids, weights, config, target):
    # Pad to a power of two so decoding compiles a handful of kernels rather than one per
    # token. The causal mask keeps real positions from ever seeing the padding.
    padded = np.zeros(min(1 << (len(ids) - 1).bit_length(), config.context), np.int32)
    padded[: len(ids)] = ids
    output = gpt2(padded, weights, n_head=config.heads, target=target)
    return output[len(ids) - 1 : len(ids)]
