"""Tests for the command line interface."""

import io
import json
import os

from noise_fingerprint import cli
from noise_fingerprint.evaluate import run_evaluation


def _small_evaluation():
    return run_evaluation(seeds=(11,), rates=(0.10,), n=300)


def test_run_demo(capsys):
    assert cli.run_demo() == 0
    out = capsys.readouterr().out
    assert "Label-noise fingerprint audit" in out
    assert "Diagnosis:" in out


def test_main_demo(capsys):
    assert cli.main(["demo"]) == 0
    assert "Remedy:" in capsys.readouterr().out


def test_run_evaluate_writes_files(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "run_evaluation", _small_evaluation)
    out_dir = str(tmp_path / "results")
    assert cli.run_evaluate(out_dir) == 0
    for name in ("evaluation.json", "evaluation.csv",
                 "evaluation_summary.md", "HEADLINE.md"):
        assert os.path.exists(os.path.join(out_dir, name))
    with io.open(os.path.join(out_dir, "evaluation.json"),
                 encoding="utf-8") as fh:
        results = json.load(fh)
    assert results["aggregate"]["n_datasets"] == 4


def test_main_evaluate(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(cli, "run_evaluation", _small_evaluation)
    out_dir = str(tmp_path / "r")
    assert cli.main(["evaluate", "--out", out_dir]) == 0
    assert "Type accuracy:" in capsys.readouterr().out
