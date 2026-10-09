"""Compare every decoding step with the actual tiny-gpt2 repository."""

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from tinymlir.model import load_model

PROMPT = "Alan Turing theorized that computers would one day become"
EXPECTED = [262, 749, 3665, 8217, 319, 262, 5440, 13, 198, 198]


def verify(reference, directory):
    model, tokenizer = load_model(directory)
    with tempfile.TemporaryDirectory() as temporary:
        capture = Path(temporary) / "reference.npz"
        subprocess.run(
            [
                "uv",
                "run",
                "--project",
                str(reference),
                "--no-sync",
                "python",
                str(Path(__file__).parent / "tests/reference.py"),
                str(reference.resolve()),
                str(directory.resolve()),
                str(capture),
            ],
            check=True,
            env={key: value for key, value in os.environ.items() if key != "VIRTUAL_ENV"},
        )
        with np.load(capture) as expected:
            ids = tokenizer.encode(PROMPT).ids
            np.testing.assert_array_equal(ids, expected["input_ids"])
            np.testing.assert_array_equal(expected["output_ids"], EXPECTED)
            generated = []
            worst = 0.0
            for step, token in enumerate(expected["output_ids"]):
                trace = {}
                logits = model(ids, trace)
                for name, value in trace.items():
                    ref = expected[f"{step}/{name}"]
                    assert value.dtype == ref.dtype, (name, value.dtype, ref.dtype)
                    np.testing.assert_allclose(
                        value, ref, rtol=3e-4, atol=3e-4, err_msg=f"step {step}: {name}"
                    )
                    worst = max(worst, float(np.max(np.abs(value - ref))))
                actual = int(logits[0].argmax())
                assert actual == token, (step, actual, token)
                generated.append(actual)
                ids.append(actual)
                print(
                    f"Step {step + 1}: token {actual} matches; all intermediate checks passed.",
                    flush=True,
                )
            assert tokenizer.decode(generated) == str(expected["completion"])
            print(f"All 10 tokens match tiny-gpt2 (NumPy {expected['numpy_version']}).")
            print(f"Maximum absolute difference across all checkpoints: {worst:.6g}")
            print(repr(tokenizer.decode(generated)))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--reference", type=Path, default=Path("../tiny-gpt2"))
    p.add_argument("--model", type=Path, default=Path("model"))
    args = p.parse_args()
    verify(args.reference, args.model)
