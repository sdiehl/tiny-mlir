"""Compare every decoding step with a pinned copy of the NumPy reference."""

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

from tinymlir.checkpoint import DEFAULT_MODEL_DIRECTORY, REPOSITORY, REVISION
from tinymlir.jit import CPU, add_target_flags
from tinymlir.model import load_model

REFERENCE_REPOSITORY = "https://github.com/sdiehl/tiny-gpt2.git"
REFERENCE_REVISION = "b8bd1dce2f03bc2ae27f157e073b3e27fd734910"
REFERENCE_RUNNER = Path(__file__).resolve().parent / "tests" / "reference.py"
FIXTURE_PATH = REFERENCE_RUNNER.with_name("gpt2.json")
TOKENIZER_FILES = {"vocab.json": "encoder.json", "merges.txt": "vocab.bpe"}
RELATIVE_TOLERANCE = 3e-4
ABSOLUTE_TOLERANCE = 3e-4


def checkout_reference(directory: Path) -> None:
    """Fetch the exact reference revision without requiring another local repository."""
    commands = (
        ["git", "init", "--quiet", str(directory)],
        [
            "git",
            "-C",
            str(directory),
            "fetch",
            "--quiet",
            "--depth=1",
            REFERENCE_REPOSITORY,
            REFERENCE_REVISION,
        ],
        ["git", "-C", str(directory), "checkout", "--quiet", "--detach", "FETCH_HEAD"],
    )
    for command in commands:
        subprocess.run(command, check=True)


def prepare_tokenizer(directory: Path) -> None:
    directory.mkdir()
    for source_name, target_name in TOKENIZER_FILES.items():
        source = hf_hub_download(REPOSITORY, source_name, revision=REVISION)
        shutil.copyfile(source, directory / target_name)


def verify(
    directory: Path, reference: Path | None = None, target: str = CPU, fused: bool = False
) -> None:
    model, tokenizer = load_model(directory, target, fused)
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as temporary:
        workspace = Path(temporary)
        if reference is None:
            reference = workspace / "tiny-gpt2"
            checkout_reference(reference)
        tokenizer_directory = workspace / "tokenizer"
        prepare_tokenizer(tokenizer_directory)
        capture = workspace / "reference.npz"
        subprocess.run(
            [
                "uv",
                "run",
                "--project",
                str(reference.resolve()),
                "--python",
                "3.12",
                "--locked",
                "--no-dev",
                "python",
                str(REFERENCE_RUNNER),
                str(reference.resolve()),
                str(directory.resolve()),
                str(capture),
                str(tokenizer_directory),
            ],
            check=True,
            env={key: value for key, value in os.environ.items() if key != "VIRTUAL_ENV"},
        )
        with np.load(capture) as expected:
            ids = tokenizer.encode(fixture["prompt"]).ids
            np.testing.assert_array_equal(ids, expected["input_ids"])
            np.testing.assert_array_equal(expected["output_ids"], fixture["output_ids"])
            np.testing.assert_equal(str(expected["completion"]), fixture["completion"])
            generated = []
            worst_difference = 0.0
            for step, token in enumerate(expected["output_ids"]):
                trace = {}
                logits = model(ids, trace)
                for name, value in trace.items():
                    reference_value = expected[f"{step}/{name}"]
                    np.testing.assert_equal(value.dtype, reference_value.dtype, err_msg=name)
                    np.testing.assert_allclose(
                        value,
                        reference_value,
                        rtol=RELATIVE_TOLERANCE,
                        atol=ABSOLUTE_TOLERANCE,
                        err_msg=f"step {step}: {name}",
                    )
                    difference = float(np.max(np.abs(value - reference_value)))
                    worst_difference = max(worst_difference, difference)
                actual = int(logits[0].argmax())
                np.testing.assert_equal(actual, token, err_msg=f"step {step}")
                generated.append(actual)
                ids.append(actual)
                print(
                    f"Step {step + 1}: token {actual} matches. Intermediate checks passed.",
                    flush=True,
                )
            np.testing.assert_equal(tokenizer.decode(generated), fixture["completion"])
            print(
                f"All {len(generated)} tokens match tiny-gpt2 (NumPy {expected['numpy_version']})."
            )
            print(f"Maximum absolute difference across all checkpoints: {worst_difference:.6g}")
            print(repr(tokenizer.decode(generated)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, help="Use a local tiny-gpt2 checkout instead")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_DIRECTORY)
    add_target_flags(parser)
    args = parser.parse_args()
    verify(args.model, args.reference, args.target, args.fused)


if __name__ == "__main__":
    main()
