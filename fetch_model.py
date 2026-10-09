"""Download the pinned GPT-2 checkpoint and tokenizer."""

import argparse
from pathlib import Path

from tinymlir.checkpoint import DEFAULT_MODEL_DIRECTORY, download_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=DEFAULT_MODEL_DIRECTORY)
    download_checkpoint(parser.parse_args().directory)


if __name__ == "__main__":
    main()
