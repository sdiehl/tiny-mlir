"""The example CLI should pass user arguments through without running on import."""

from unittest.mock import Mock

import example


def test_example_arguments(monkeypatch, capsys, tmp_path):
    model, tokenizer = object(), object()
    load_model = Mock(return_value=(model, tokenizer))
    generate = Mock(return_value="A supplied prompt completed")
    monkeypatch.setattr(example, "load_model", load_model)
    monkeypatch.setattr(example, "generate", generate)
    monkeypatch.setattr(
        "sys.argv", ["example.py", "A supplied prompt", "--tokens", "7", "--model", str(tmp_path)]
    )
    example.main()
    load_model.assert_called_once_with(tmp_path, "cpu")
    generate.assert_called_once_with(model, tokenizer, "A supplied prompt", 7)
    assert capsys.readouterr().out == "A supplied prompt completed\n"
