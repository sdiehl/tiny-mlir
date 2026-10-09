from tinymlir.model import generate, load_model

print("Loading model parameters...")
model, tokenizer = load_model()

example_prompts = [
    ("Alan Turing theorized that computers would one day become", 10),
]

for prompt, max_tokens in example_prompts:
    print(f"\nPrompt: {prompt}")
    print(generate(model, tokenizer, prompt, max_tokens))
