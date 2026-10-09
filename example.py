"""Complete a prompt using GPT-2 compiled through MLIR."""

import argparse
from pathlib import Path

from tinymlir.checkpoint import DEFAULT_MODEL_DIRECTORY
from tinymlir.model import generate, load_model

DEFAULT_GENERATION_LENGTH = 10


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", help="Text to complete")
    parser.add_argument("--tokens", type=int, default=DEFAULT_GENERATION_LENGTH)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_DIRECTORY)
    args = parser.parse_args()
    if args.tokens < 0:
        parser.error("--tokens must be nonnegative")
    model, tokenizer = load_model(args.model)
    print(generate(model, tokenizer, args.prompt, args.tokens))


if __name__ == "__main__":
    main()
