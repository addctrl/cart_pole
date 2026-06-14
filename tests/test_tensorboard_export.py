"""Testy jednostkowe modułu ``src.tensorboard_export``."""

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from src import tensorboard_export
from src.tensorboard_export import (
    discover_event_files,
    extract_scalar_rows,
    write_long_csv,
    write_pivot_csv,
)


class FakeScalarEvent:
    """Minimalna atrapa eventu skalarnego TensorBoard."""

    def __init__(self, step: int, value: float) -> None:
        """Zainicjalizuj event skalaru.

        Parameters
        ----------
        step : int
            Krok treningowy.
        value : float
            Wartosc skalaru.
        """
        self.step = step
        self.value = value


class FakeAccumulator:
    """Atrapa EventAccumulator do testow ekstrakcji."""

    def __init__(self, path: str) -> None:
        """Zapamietaj sciezke pliku eventow.

        Parameters
        ----------
        path : str
            Sciezka do pliku eventow TensorBoard.
        """
        self.path = path

    def Reload(self) -> None:
        """Atrapowa metoda reload bez efektu ubocznego."""

    def Tags(self) -> dict[str, list[str]]:
        """Zwroc fake tagi skalarne dla danego pliku eventow.

        Returns
        -------
        dict[str, list[str]]
            Slownik tagow zgodny z API EventAccumulator.
        """
        return {"scalars": ["rollout/ep_rew_mean", "train/loss"]}

    def Scalars(self, tag: str) -> list[FakeScalarEvent]:
        """Zwroc fake serie wartosci dla wskazanego tagu.

        Parameters
        ----------
        tag : str
            Nazwa tagu skalarnego.

        Returns
        -------
        list[FakeScalarEvent]
            Lista eventow dla tagu.
        """
        data: dict[str, list[tuple[int, float]]] = {
            "rollout/ep_rew_mean": [(10, 12.5), (20, 15.0)],
            "train/loss": [(10, 0.9), (20, 0.7)],
        }
        return [FakeScalarEvent(step=step, value=value) for step, value in data[tag]]


def test_discover_event_files_returns_all_event_paths(tmp_path: Path) -> None:
    """Zweryfikuj wykrycie wszystkich plikow eventow."""
    run1 = tmp_path / "run1"
    run2 = tmp_path / "run2"
    run1.mkdir()
    run2.mkdir()
    (run1 / "events.out.tfevents.1").write_text("x", encoding="utf-8")
    (run2 / "events.out.tfevents.2").write_text("x", encoding="utf-8")

    files = discover_event_files(str(tmp_path))

    assert len(files) == 2
    assert all("events.out.tfevents" in str(path) for path in files)


def test_extract_scalar_rows_filters_tags(tmp_path: Path) -> None:
    """Zweryfikuj ekstrakcje i filtrowanie wybranych tagow."""
    run = tmp_path / "runA"
    run.mkdir()
    (run / "events.out.tfevents.1").write_text("x", encoding="utf-8")

    with patch("src.tensorboard_export.EventAccumulator", FakeAccumulator):
        rows = extract_scalar_rows(
            logdir=str(tmp_path),
            tags={"rollout/ep_rew_mean"},
        )

    assert len(rows) == 2
    assert all(row["tag"] == "rollout/ep_rew_mean" for row in rows)


def test_write_long_and_pivot_csv(tmp_path: Path) -> None:
    """Zweryfikuj zapis long i pivot CSV dla skalarow."""
    rows: list[dict[str, Any]] = [
        {"run": "runA", "step": 10, "tag": "rollout/ep_rew_mean", "value": 1.0},
        {"run": "runA", "step": 10, "tag": "train/loss", "value": 0.5},
        {"run": "runA", "step": 20, "tag": "rollout/ep_rew_mean", "value": 2.0},
    ]

    long_path = tmp_path / "long.csv"
    pivot_path = tmp_path / "pivot.csv"

    write_long_csv(rows, str(long_path))
    pivot_count = write_pivot_csv(rows, str(pivot_path))

    long_content = long_path.read_text(encoding="utf-8")
    pivot_content = pivot_path.read_text(encoding="utf-8")

    assert "run,step,tag,value" in long_content
    assert "run,step,rollout/ep_rew_mean,train/loss" in pivot_content
    assert pivot_count == 2


def test_main_prints_message_when_no_rows(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Zweryfikuj sciezke CLI, gdy brak danych do eksportu."""
    output = tmp_path / "pivot.csv"
    monkeypatch.setattr(
        "sys.argv",
        [
            "tensorboard_export",
            "--logdir",
            str(tmp_path),
            "--output-csv",
            str(output),
        ],
    )
    monkeypatch.setattr(tensorboard_export, "extract_scalar_rows", lambda *_: [])

    tensorboard_export.main()
    out = capsys.readouterr().out

    assert "Nie znaleziono zadnych danych skalarnych" in out


def test_main_writes_pivot_and_long_and_prints_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Zweryfikuj pelna sciezke CLI z zapisami pivot i long."""
    pivot = tmp_path / "pivot.csv"
    long = tmp_path / "long.csv"
    sample_rows: list[dict[str, Any]] = [
        {"run": "runB", "step": 1, "tag": "rollout/ep_rew_mean", "value": 3.0},
    ]

    monkeypatch.setattr(
        "sys.argv",
        [
            "tensorboard_export",
            "--logdir",
            str(tmp_path),
            "--output-csv",
            str(pivot),
            "--long-output-csv",
            str(long),
            "--tags",
            "rollout/ep_rew_mean",
        ],
    )

    def fake_extract(logdir: str, tags: set[str] | None) -> list[dict[str, Any]]:
        assert logdir == str(tmp_path)
        assert tags == {"rollout/ep_rew_mean"}
        return sample_rows

    monkeypatch.setattr(tensorboard_export, "extract_scalar_rows", fake_extract)

    tensorboard_export.main()
    out = capsys.readouterr().out

    assert "Zapisano" in out
    assert pivot.exists()
    assert long.exists()
