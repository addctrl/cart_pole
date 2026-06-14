"""Testy jednostkowe modułu ``src.training``."""

from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

from src.training import get_cooldown_seconds, main, run_all_experiments, run_experiment

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SMALL_NET_ARCH = "[64, 64]"
_LARGE_NET_ARCH = "[1024, 1024, 1024]"


@pytest.fixture
def small_config() -> dict[str, Any]:
    """Zwróć konfigurację eksperymentu z małą siecią neuronową ([64, 64]).

    Returns
    -------
    dict[str, Any]
        Słownik konfiguracji gotowy do przekazania do ``run_experiment``.
    """
    return {
        "experiment_id": "exp_001",
        "env_id": "CartPole-v1",
        "net_arch": _SMALL_NET_ARCH,
        "learning_rate": 0.0003,
        "batch_size": 256,
        "gamma": 0.99,
        "n_steps": 2048,
        "ent_coef": 0.01,
        "total_timesteps": 100000,
        "mean_reward": None,
        "std_reward": None,
        "training_time_s": None,
    }


@pytest.fixture
def large_config() -> dict[str, Any]:
    """Zwróć konfigurację eksperymentu z dużą siecią neuronową ([1024, 1024, 1024]).

    Returns
    -------
    dict[str, Any]
        Słownik konfiguracji gotowy do przekazania do ``run_experiment``.
    """
    return {
        "experiment_id": "exp_003",
        "env_id": "CartPole-v1",
        "net_arch": _LARGE_NET_ARCH,
        "learning_rate": 0.0003,
        "batch_size": 256,
        "gamma": 0.99,
        "n_steps": 2048,
        "ent_coef": 0.01,
        "total_timesteps": 200000,
        "mean_reward": None,
        "std_reward": None,
        "training_time_s": None,
    }


# ---------------------------------------------------------------------------
# Testy: get_cooldown_seconds
# ---------------------------------------------------------------------------


def test_get_cooldown_small_net() -> None:
    """Zweryfikuj cooldown 60s dla małej sieci [64, 64]."""
    assert get_cooldown_seconds([64, 64]) == 60


def test_get_cooldown_large_net() -> None:
    """Zweryfikuj cooldown 120s dla dużej sieci [1024, 1024, 1024]."""
    assert get_cooldown_seconds([1024, 1024, 1024]) == 120


def test_get_cooldown_boundary() -> None:
    """Zweryfikuj cooldown 60s na granicy sumy == 1000 (warunek > 1000)."""
    assert get_cooldown_seconds([500, 500]) == 60


# ---------------------------------------------------------------------------
# Testy: run_experiment
# ---------------------------------------------------------------------------


def test_run_experiment_returns_metrics(small_config: dict[str, Any]) -> None:
    """Zweryfikuj zwrócenie słownika z wymaganymi kluczami metryk.

    Parameters
    ----------
    small_config : dict[str, Any]
        Fixture — konfiguracja eksperymentu z małą siecią.
    """
    mock_env = MagicMock()
    mock_model = MagicMock()

    with (
        patch("src.training.gym.make", return_value=mock_env),
        patch("src.training.Monitor", side_effect=lambda env: env),
        patch("src.training.PPO", return_value=mock_model),
        patch("src.training.evaluate_policy", return_value=(450.0, 25.0)),
    ):
        result = run_experiment(small_config)

    assert "mean_reward" in result
    assert "std_reward" in result
    assert "training_time_s" in result
    assert result["mean_reward"] == pytest.approx(450.0)
    assert result["std_reward"] == pytest.approx(25.0)


def test_run_experiment_saves_model(small_config: dict[str, Any]) -> None:
    """Zweryfikuj wywołanie model.save() z poprawną ścieżką.

    Parameters
    ----------
    small_config : dict[str, Any]
        Fixture — konfiguracja eksperymentu z małą siecią.
    """
    mock_env = MagicMock()
    mock_model = MagicMock()

    with (
        patch("src.training.gym.make", return_value=mock_env),
        patch("src.training.Monitor", side_effect=lambda env: env),
        patch("src.training.PPO", return_value=mock_model),
        patch("src.training.evaluate_policy", return_value=(450.0, 25.0)),
    ):
        run_experiment(small_config)

    mock_model.save.assert_called_once_with("models/exp_001")


def test_run_experiment_uses_device_auto(small_config: dict[str, Any]) -> None:
    """Zweryfikuj inicjalizację PPO z device='auto' (DKB-004).

    Parameters
    ----------
    small_config : dict[str, Any]
        Fixture — konfiguracja eksperymentu z małą siecią.
    """
    mock_env = MagicMock()
    mock_model = MagicMock()
    mock_ppo_class = MagicMock(return_value=mock_model)

    with (
        patch("src.training.gym.make", return_value=mock_env),
        patch("src.training.Monitor", side_effect=lambda env: env),
        patch("src.training.PPO", mock_ppo_class),
        patch("src.training.evaluate_policy", return_value=(450.0, 25.0)),
    ):
        run_experiment(small_config)

    _, kwargs = mock_ppo_class.call_args
    assert kwargs.get("device") == "auto"


def test_run_experiment_tb_log_name(small_config: dict[str, Any]) -> None:
    """Zweryfikuj wywołanie model.learn() z tb_log_name=experiment_id (DKB-005).

    Parameters
    ----------
    small_config : dict[str, Any]
        Fixture — konfiguracja eksperymentu z małą siecią.
    """
    mock_env = MagicMock()
    mock_model = MagicMock()

    with (
        patch("src.training.gym.make", return_value=mock_env),
        patch("src.training.Monitor", side_effect=lambda env: env),
        patch("src.training.PPO", return_value=mock_model),
        patch("src.training.evaluate_policy", return_value=(450.0, 25.0)),
    ):
        run_experiment(small_config)

    _, kwargs = mock_model.learn.call_args
    assert kwargs.get("tb_log_name") == "exp_001"


# ---------------------------------------------------------------------------
# Testy: run_all_experiments
# ---------------------------------------------------------------------------


def test_run_all_experiments_sleeps(small_config: dict[str, Any]) -> None:
    """Zweryfikuj wywołanie time.sleep() z właściwą wartością cooldownu.

    Parameters
    ----------
    small_config : dict[str, Any]
        Fixture — konfiguracja eksperymentu z małą siecią (cooldown 60s).
    """
    mock_metrics = {"mean_reward": 450.0, "std_reward": 25.0, "training_time_s": 30.0}

    with (
        patch("src.training.load_experiments", return_value=[small_config]),
        patch("src.training.run_experiment", return_value=mock_metrics),
        patch("src.training.save_results"),
        patch("src.training.time.sleep") as mock_sleep,
    ):
        run_all_experiments("dummy.csv")

    mock_sleep.assert_called_once_with(60)


def test_run_all_experiments_skips_completed(
    small_config: dict[str, Any],
) -> None:
    """Zweryfikuj pominięcie eksperymentu z wypełnionym mean_reward.

    Parameters
    ----------
    small_config : dict[str, Any]
        Fixture — konfiguracja eksperymentu z małą siecią.
    """
    completed_config = dict(small_config)
    completed_config["mean_reward"] = 450.0

    pending_config = dict(small_config)
    pending_config["experiment_id"] = "exp_002"

    mock_metrics = {"mean_reward": 430.0, "std_reward": 20.0, "training_time_s": 25.0}

    with (
        patch(
            "src.training.load_experiments",
            return_value=[completed_config, pending_config],
        ),
        patch("src.training.run_experiment", return_value=mock_metrics) as mock_run,
        patch("src.training.save_results"),
        patch("src.training.time.sleep"),
    ):
        run_all_experiments("dummy.csv")

    assert mock_run.call_count == 1
    call_config = mock_run.call_args[0][0]
    assert call_config["experiment_id"] == "exp_002"


def test_run_all_experiments_saves_results(small_config: dict[str, Any]) -> None:
    """Zweryfikuj wywołanie save_results raz na każdy uruchomiony eksperyment.

    Parameters
    ----------
    small_config : dict[str, Any]
        Fixture — konfiguracja eksperymentu z małą siecią.
    """
    config_b = dict(small_config)
    config_b["experiment_id"] = "exp_002"

    mock_metrics = {"mean_reward": 450.0, "std_reward": 25.0, "training_time_s": 30.0}

    with (
        patch(
            "src.training.load_experiments",
            return_value=[small_config, config_b],
        ),
        patch("src.training.run_experiment", return_value=mock_metrics),
        patch("src.training.save_results") as mock_save,
        patch("src.training.time.sleep"),
    ):
        run_all_experiments("dummy.csv")

    assert mock_save.call_count == 2
    assert mock_save.call_args_list == [
        call("dummy.csv", "exp_001", mock_metrics),
        call("dummy.csv", "exp_002", mock_metrics),
    ]


def test_main_calls_run_all_experiments(monkeypatch: pytest.MonkeyPatch) -> None:
    """Zweryfikuj przekazanie argumentu ``--csv`` do run_all_experiments.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture do tymczasowej podmiany ``sys.argv``.
    """
    monkeypatch.setattr(
        "sys.argv",
        [
            "training",
            "--csv",
            "data/experiments.csv",
        ],
    )

    with patch("src.training.run_all_experiments") as mock_run_all_experiments:
        main()

    mock_run_all_experiments.assert_called_once_with("data/experiments.csv")
