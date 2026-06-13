"""Testy jednostkowe modułu ``src.evaluate``."""

from unittest.mock import MagicMock, patch

import pytest

from src.evaluate import evaluate_model, main


def test_evaluate_model_loads_model_and_renders() -> None:
    """Zweryfikuj ładowanie modelu i renderowanie wskazanej liczby epizodów."""
    mock_model = MagicMock()
    mock_env = MagicMock()
    mock_env.reset.side_effect = [
        ("obs-1", {}),
        ("obs-3", {}),
    ]
    mock_env.step.side_effect = [
        ("obs-2", 1.0, False, False, {}),
        ("obs-3", 1.0, True, False, {}),
        ("obs-4", 1.0, False, False, {}),
        ("obs-5", 1.0, False, True, {}),
    ]
    mock_model.predict.side_effect = [
        (0, None),
        (1, None),
        (0, None),
        (1, None),
    ]

    with (
        patch("src.evaluate.Path.is_file", return_value=True),
        patch("src.evaluate.PPO.load", return_value=mock_model) as mock_load,
        patch("src.evaluate.gym.make", return_value=mock_env) as mock_make,
    ):
        evaluate_model("models/exp_001.zip", "CartPole-v1", 2)

    mock_load.assert_called_once_with("models/exp_001.zip")
    mock_make.assert_called_once_with("CartPole-v1", render_mode="human")
    assert mock_model.predict.call_count == 4
    mock_env.close.assert_called_once_with()


def test_evaluate_model_invalid_model_path() -> None:
    """Zweryfikuj błąd dla nieistniejącej ścieżki modelu."""
    with pytest.raises(FileNotFoundError, match="Plik modelu nie istnieje"):
        evaluate_model("models/missing.zip", "CartPole-v1", 1)


def test_main_calls_evaluate_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Zweryfikuj przekazanie argumentów CLI do evaluate_model.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture do tymczasowej podmiany ``sys.argv``.
    """
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate",
            "--model-path",
            "models/exp_001.zip",
            "--env-id",
            "CartPole-v1",
            "--episodes",
            "5",
        ],
    )

    with patch("src.evaluate.evaluate_model") as mock_evaluate_model:
        main()

    mock_evaluate_model.assert_called_once_with(
        "models/exp_001.zip",
        "CartPole-v1",
        5,
    )
