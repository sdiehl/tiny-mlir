import numpy as np

from tinymlir import CPU, Model
from tinymlir.model import load_checkpoint

print("Loading model parameters...")
weights, config, tokenizer = load_checkpoint("model")
ops = CPU()
model = Model(weights, config, ops)

example_prompts = [
    ("Alan Turing theorized that computers would one day become", 10),
]

try:
    for prompt, max_tokens in example_prompts:
        print(f"\nPrompt: {prompt}")
        tokens = tokenizer.encode(prompt).ids
        for _ in range(max_tokens):
            logits = ops.numpy(model(tokens))
            token = int(np.argmax(logits[0]))
            tokens.append(token)
            if token == config.get("eos_token_id"):
                break
        print(tokenizer.decode(tokens))
finally:
    ops.close()
