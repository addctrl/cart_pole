"""Testy jednostkowe modułu ``src.objective_score_csv``."""

from pathlib import Path

import pytest

from src import objective_score_csv
from src.objective_score_csv import (
    compute_objective_score,
    recompute_objective_score_csv,
)


def test_compute_objective_score_returns_expected_value() -> None:
    """Zweryfikuj wzor objective_score."""
    score = compute_objective_score(100.0, 20.0, 0.1)

    assert score == pytest.approx(98.0)


def test_recompute_objective_score_csv_creates_output(tmp_path: Path) -> None:
    """Zweryfikuj przeliczenie objective_score i zapis pliku wyjsciowego."""
    source = tmp_path / "input.csv"
    source.write_text(
        "experiment_id,mean_reward,std_reward\n"
        "exp_1,100.0,10.0\n"
        "exp_2,200.0,20.0\n",
        encoding="utf-8",
    )

    output = recompute_objective_score_csv(str(source), stability_penalty=0.1)
    content = Path(output).read_text(encoding="utf-8")

    assert "objective_score" in content
    assert "exp_1,100.0,10.0,99" in content
    assert "exp_2,200.0,20.0,198" in content


def test_recompute_objective_score_csv_handles_empty_values(tmp_path: Path) -> None:
    """Zweryfikuj pusty objective_score dla brakujacych metryk."""
    source = tmp_path / "input_empty.csv"
    source.write_text(
        "experiment_id,mean_reward,std_reward,objective_score\n"
        "exp_1,100.0,10.0,1\n"
        "exp_2,200.0,,2\n",
        encoding="utf-8",
    )

    output = recompute_objective_score_csv(
        str(source),
        stability_penalty=0.1,
        output_path=str(tmp_path / "out.csv"),
    )
    content = Path(output).read_text(encoding="utf-8")

    assert "exp_1,100.0,10.0,99" in content
    assert "exp_2,200.0,," in content


def test_recompute_objective_score_csv_requires_reward_columns(tmp_path: Path) -> None:
    """Zweryfikuj blad dla CSV bez wymaganych kolumn metryk."""
    source = tmp_path / "invalid.csv"
    source.write_text(
        "experiment_id,training_time_s\n"
        "exp_1,12.3\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="mean_reward"):
        recompute_objective_score_csv(str(source), stability_penalty=0.1)


def test_main_processes_multiple_csv_paths(tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    """Zweryfikuj sciezke CLI dla wielu CSV i opcji in-place."""
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    first.write_text(
        "experiment_id,mean_reward,std_reward\nexp_1,10.0,2.0\n",
        encoding="utf-8",
    )
    second.write_text(
        "experiment_id,mean_reward,std_reward\nexp_2,20.0,3.0\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "objective_score_csv",
            "--csv-paths",
            str(first),
            str(second),
            "--stability-penalty",
            "0.2",
            "--in-place",
        ],
    )

    objective_score_csv.main()
    out = capsys.readouterr().out

    assert "Przeliczono objective_score" in out
    assert "exp_1,10.0,2.0,9.6" in first.read_text(encoding="utf-8")
    assert "exp_2,20.0,3.0,19.4" in second.read_text(encoding="utf-8")
