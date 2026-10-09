"""Download one pinned GPT-2 checkpoint (~550 MB) and its tokenizer."""

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

REPOSITORY = "openai-community/gpt2"
REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", nargs="?", default="model")
    directory = Path(parser.parse_args().directory)
    for name in ("config.json", "tokenizer.json", "model.safetensors"):
        print(hf_hub_download(REPOSITORY, name, revision=REVISION, local_dir=directory))
    (directory / "revision.txt").write_text(f"{REPOSITORY}\n{REVISION}\n")
