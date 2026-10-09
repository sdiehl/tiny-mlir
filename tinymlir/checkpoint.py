"""Pinned checkpoint metadata shared by downloading and model loading."""

from pathlib import Path

from huggingface_hub import hf_hub_download

REPOSITORY = "openai-community/gpt2"
REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
DEFAULT_MODEL_DIRECTORY = Path("model")
CONFIG_FILENAME = "config.json"
WEIGHTS_FILENAME = "model.safetensors"
TOKENIZER_FILENAME = "tokenizer.json"
REVISION_FILENAME = "revision.txt"
CHECKPOINT_FILES = (CONFIG_FILENAME, TOKENIZER_FILENAME, WEIGHTS_FILENAME)


def download_checkpoint(directory: Path = DEFAULT_MODEL_DIRECTORY) -> None:
    for filename in CHECKPOINT_FILES:
        path = hf_hub_download(REPOSITORY, filename, revision=REVISION, local_dir=directory)
        print(path)
    (directory / REVISION_FILENAME).write_text(f"{REPOSITORY}\n{REVISION}\n", encoding="utf-8")
