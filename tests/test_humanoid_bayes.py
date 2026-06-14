"""Testy jednostkowe modułu ``src.humanoid_bayes``."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.humanoid_bayes import (
    DEFAULT_EVAL_EPISODES,
    DEFAULT_OPTUNA_STORAGE,
    DEFAULT_PRUNER_WARMUP_STEPS,
    DEFAULT_REPORT_INTERVAL_TIMESTEPS,
    DEFAULT_RESULTS_CSV,
    DEFAULT_STABILITY_PENALTY,
    DEFAULT_STARTUP_TRIALS,
    DEFAULT_TIMESTEPS,
    DEFAULT_TRIALS,
    HUMANOID_ENV_ID,
    HumanoidTrialPruned,
    _require_optuna,
    append_humanoid_result,
    build_budget_warning,
    build_humanoid_config,
    compute_humanoid_objective_score,
    main,
    run_humanoid_study,
    run_prunable_humanoid_experiment,
    sample_humanoid_hyperparameters,
)


class FakeTrial:
    """Minimalna atrapa triala Optuny do testów jednostkowych."""

    def __init__(self) -> None:
        """Zainicjalizuj atrapy zwracające deterministyczne wartości."""
        self.number = 7

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        *,
        log: bool = False,
    ) -> float:
        """Zwróć deterministyczną wartość zgodną z nazwą parametru.

        Parameters
        ----------
        name : str
            Nazwa hiperparametru.
        low : float
            Dolna granica zakresu.
        high : float
            Górna granica zakresu.
        log : bool, default=False
            Flaga logarytmicznego próbkowania.

        Returns
        -------
        float
            Deterministyczna wartość testowa.
        """
        assert high >= low
        assert isinstance(log, bool)
        return {
            "learning_rate": 0.0003,
            "ent_coef": 0.01,
            "target_kl": 0.02,
        }[name]

    def suggest_categorical(self, name: str, choices: list[Any]) -> Any:
        """Zwróć deterministyczny wybór ze zbioru kategorycznego.

        Parameters
        ----------
        name : str
            Nazwa hiperparametru.
        choices : list[Any]
            Kandydaci do wyboru.

        Returns
        -------
        Any
            Wybrana wartość testowa.
        """
        selected = {
            "n_steps": 2048,
            "batch_size": 256,
            "gamma": 0.99,
            "gae_lambda": 0.966,
            "clip_range": 0.15,
            "n_epochs": 15,
            "vf_coef": 0.5,
            "normalize_advantage": True,
        }[name]
        assert selected in choices
        return selected

    def report(self, value: float, step: int) -> None:
        """Przyjmij wynik pośredni bez dodatkowej logiki.

        Parameters
        ----------
        value : float
            Wartość metryki pośredniej.
        step : int
            Krok treningowy raportu.
        """
        assert step >= 0
        assert isinstance(value, float)

    def should_prune(self) -> bool:
        """Nigdy nie przerywaj próby w podstawowej atrapie."""
        return False


class FakePruningTrial(FakeTrial):
    """Atrapa triala, która wymusza pruning po pierwszym raporcie."""

    def __init__(self) -> None:
        """Zainicjalizuj flagę wykrycia raportu pośredniego."""
        super().__init__()
        self.has_report = False

    def report(self, value: float, step: int) -> None:
        """Zapamiętaj fakt wysłania raportu pośredniego."""
        super().report(value, step)
        self.has_report = True

    def should_prune(self) -> bool:
        """Przerwij próbę po pierwszym raporcie pośrednim."""
        return self.has_report


def test_require_optuna_returns_module_when_available() -> None:
    """Zweryfikuj zwrócenie modułu Optuna przy poprawnej instalacji."""

    class FakeOptunaModule:
        """Atrapa modułu Optuna."""

    with patch(
        "src.humanoid_bayes.importlib.import_module",
        return_value=FakeOptunaModule(),
    ):
        module = _require_optuna()

    assert isinstance(module, FakeOptunaModule)


def test_require_optuna_raises_clear_error_when_missing() -> None:
    """Zweryfikuj czytelny komunikat przy braku optymalizacji Humanoid."""
    with patch(
        "src.humanoid_bayes.importlib.import_module",
        side_effect=ModuleNotFoundError("optuna missing"),
    ):
        with pytest.raises(ModuleNotFoundError, match="requirements-humanoid.txt"):
            _require_optuna()


def test_build_humanoid_config_uses_fixed_variant() -> None:
    """Zweryfikuj stałe środowisko i architekturę wariantu Humanoid."""
    config = build_humanoid_config(
        trial_number=3,
        learning_rate=0.0003,
        batch_size=256,
        gamma=0.99,
        n_steps=2048,
        ent_coef=0.01,
        gae_lambda=0.966,
        clip_range=0.15,
        target_kl=0.02,
        n_epochs=15,
        vf_coef=0.5,
        normalize_advantage=True,
        total_timesteps=600000,
    )

    assert config["experiment_id"] == "hum_256x256_trial_003"
    assert config["env_id"] == HUMANOID_ENV_ID
    assert config["net_arch"] == "[256, 256]"
    assert config["clip_range"] == pytest.approx(0.15)
    assert config["normalize_advantage"] is True
    assert config["total_timesteps"] == 600000


def test_sample_humanoid_hyperparameters_preserves_divisibility() -> None:
    """Zweryfikuj zgodność batch_size z ograniczeniem DKB-003."""
    params = sample_humanoid_hyperparameters(FakeTrial())

    assert params["n_steps"] == 2048
    assert params["batch_size"] == 256
    assert params["clip_range"] == pytest.approx(0.15)
    assert params["target_kl"] == pytest.approx(0.02)
    assert params["normalize_advantage"] is True
    assert int(params["n_steps"]) % int(params["batch_size"]) == 0


def test_compute_humanoid_objective_score_penalizes_variance() -> None:
    """Zweryfikuj scoring stabilnościowy zgodny z karą wariancyjną."""
    score = compute_humanoid_objective_score(100.0, 15.0, 0.1)

    assert score == pytest.approx(98.5)


def test_append_humanoid_result_creates_csv(tmp_path: Path) -> None:
    """Zweryfikuj utworzenie i zapis jednego wiersza wynikowego.

    Parameters
    ----------
    tmp_path : Path
        Katalog tymczasowy pytesta.
    """
    csv_path = tmp_path / "humanoid.csv"
    config = build_humanoid_config(
        trial_number=1,
        learning_rate=0.0003,
        batch_size=256,
        gamma=0.99,
        n_steps=2048,
        ent_coef=0.01,
        gae_lambda=0.966,
        clip_range=0.15,
        target_kl=0.02,
        n_epochs=15,
        vf_coef=0.5,
        normalize_advantage=True,
        total_timesteps=300000,
    )
    metrics = {"mean_reward": 123.4, "std_reward": 5.6, "training_time_s": 78.9}

    append_humanoid_result(str(csv_path), config, metrics)

    content = csv_path.read_text(encoding="utf-8")
    assert "experiment_id,env_id,net_arch" in content
    assert "hum_256x256_trial_001" in content
    assert "123.4" in content
    assert "normalize_advantage" in content


def test_build_budget_warning_for_smoke_budget() -> None:
    """Zweryfikuj ostrzeżenie dla zbyt małego budżetu Humanoida."""
    warning = build_budget_warning(10, 300000)

    assert warning is not None
    assert "smoke test" in warning


def test_build_budget_warning_returns_none_for_reasonable_budget() -> None:
    """Zweryfikuj brak ostrzeżenia dla większego budżetu strojenia."""
    assert build_budget_warning(30, 1000000) is None


def test_run_prunable_humanoid_experiment_returns_metrics() -> None:
    """Zweryfikuj etapowy trening Humanoida bez przerwania próby."""
    config = build_humanoid_config(
        trial_number=1,
        learning_rate=0.0003,
        batch_size=256,
        gamma=0.99,
        n_steps=2048,
        ent_coef=0.01,
        gae_lambda=0.966,
        clip_range=0.15,
        target_kl=0.02,
        n_epochs=15,
        vf_coef=0.5,
        normalize_advantage=True,
        total_timesteps=200000,
    )
    mock_env = MagicMock()

    with (
        patch("src.humanoid_bayes.gym.make", return_value=mock_env),
        patch("src.humanoid_bayes.Monitor", side_effect=lambda env: env),
        patch("src.humanoid_bayes.DummyVecEnv", return_value=mock_env),
        patch("src.humanoid_bayes.VecNormalize", side_effect=lambda env, **kwargs: env),
        patch("src.humanoid_bayes.PPO") as mock_ppo,
        patch(
            "src.humanoid_bayes.evaluate_policy",
            side_effect=[(10.0, 1.0), (20.0, 2.0)],
        ),
        patch("src.humanoid_bayes.time.time", side_effect=[100.0, 130.0, 160.0]),
    ):
        mock_model = mock_ppo.return_value
        metrics = run_prunable_humanoid_experiment(
            config=config,
            trial=FakeTrial(),
            report_interval_timesteps=100000,
            eval_episodes=20,
            stability_penalty=0.1,
        )

    assert metrics["mean_reward"] == pytest.approx(20.0)
    assert metrics["std_reward"] == pytest.approx(2.0)
    assert metrics["objective_score"] == pytest.approx(19.8)
    assert metrics["trained_timesteps"] == pytest.approx(200000.0)
    assert mock_ppo.call_count == 1
    assert mock_model.learn.call_count == 2
    mock_model.save.assert_called_once_with("models/hum_256x256_trial_001")
    mock_env.save.assert_called_once_with("models/hum_256x256_trial_001_vecnormalize.pkl")


def test_run_prunable_humanoid_experiment_raises_pruned() -> None:
    """Zweryfikuj przerwanie próby po raporcie pośrednim do Optuny."""
    config = build_humanoid_config(
        trial_number=2,
        learning_rate=0.0003,
        batch_size=256,
        gamma=0.99,
        n_steps=2048,
        ent_coef=0.01,
        gae_lambda=0.966,
        clip_range=0.15,
        target_kl=0.02,
        n_epochs=15,
        vf_coef=0.5,
        normalize_advantage=True,
        total_timesteps=200000,
    )
    mock_env = MagicMock()

    with (
        patch("src.humanoid_bayes.gym.make", return_value=mock_env),
        patch("src.humanoid_bayes.Monitor", side_effect=lambda env: env),
        patch("src.humanoid_bayes.DummyVecEnv", return_value=mock_env),
        patch("src.humanoid_bayes.VecNormalize", side_effect=lambda env, **kwargs: env),
        patch("src.humanoid_bayes.PPO") as mock_ppo,
        patch("src.humanoid_bayes.evaluate_policy", return_value=(10.0, 1.0)),
        patch("src.humanoid_bayes.time.time", side_effect=[100.0, 125.0]),
    ):
        with pytest.raises(HumanoidTrialPruned) as exc_info:
            run_prunable_humanoid_experiment(
                config=config,
                trial=FakePruningTrial(),
                report_interval_timesteps=100000,
                eval_episodes=20,
                stability_penalty=0.1,
            )

    assert exc_info.value.metrics["mean_reward"] == pytest.approx(10.0)
    assert exc_info.value.metrics["objective_score"] == pytest.approx(9.9)
    assert exc_info.value.metrics["trained_timesteps"] == pytest.approx(100000.0)
    mock_ppo.return_value.save.assert_not_called()


def test_run_humanoid_study_returns_best_summary() -> None:
    """Zweryfikuj integrację study z dedykowanym objective bez realnego treningu."""

    class FakeStudy:
        """Atrapa obiektu study Optuny."""

        def __init__(self) -> None:
            """Ustaw wartości końcowe studium."""
            self.best_value = 321.0
            self.best_params = {"learning_rate": 0.0003}

        def optimize(self, objective: Any, n_trials: int) -> None:
            """Wywołaj objective jeden raz dla deterministycznej próby."""
            assert n_trials == 2
            objective(FakeTrial())

    class FakeSamplers:
        """Przestrzeń nazw z atrapą TPESampler."""

        @staticmethod
        def TPESampler(seed: int) -> object:
            """Zwróć obiekt sentynelowy dla seedowanego samplera."""
            assert seed == 42
            return object()

    class FakeOptuna:
        """Atrapa modułu Optuna."""

        samplers = FakeSamplers()
        TrialPruned = RuntimeError

        class pruners:
            """Przestrzeń nazw z atrapą prunerów."""

            @staticmethod
            def MedianPruner(n_startup_trials: int, n_warmup_steps: int) -> object:
                """Zwróć obiekt sentynelowy dla MedianPruner."""
                assert n_startup_trials == 5
                assert n_warmup_steps == 100000
                return object()

        @staticmethod
        def create_study(
            direction: str,
            study_name: str,
            sampler: object,
            pruner: object,
            storage: str,
            load_if_exists: bool,
        ) -> FakeStudy:
            """Zwróć atrapę studium i zweryfikuj parametry wejściowe."""
            assert direction == "maximize"
            assert study_name == "humanoid_test"
            assert sampler is not None
            assert pruner is not None
            assert storage == "sqlite:///data/humanoid_optuna.db"
            assert load_if_exists is True
            return FakeStudy()

    with (
        patch("src.humanoid_bayes._require_optuna", return_value=FakeOptuna()),
        patch(
            "src.humanoid_bayes.run_prunable_humanoid_experiment",
            return_value={
                "mean_reward": 321.0,
                "std_reward": 12.3,
                "objective_score": 319.77,
                "training_time_s": 45.6,
                "trained_timesteps": 300000.0,
            },
        ),
        patch("src.humanoid_bayes.append_humanoid_result"),
        patch("src.humanoid_bayes.time.sleep"),
    ):
        summary = run_humanoid_study(
            trials=2,
            total_timesteps=300000,
            results_csv="data/humanoid.csv",
            study_name="humanoid_test",
            startup_trials=5,
            pruner_warmup_steps=100000,
            report_interval_timesteps=100000,
            eval_episodes=20,
            stability_penalty=0.1,
            optuna_storage="sqlite:///data/humanoid_optuna.db",
        )

    assert summary["best_value"] == pytest.approx(321.0)
    assert summary["best_params"] == {"learning_rate": 0.0003}


def test_run_humanoid_study_marks_pruned_trials() -> None:
    """Zweryfikuj zapis statusu `pruned` i propagację wyjątku Optuny."""

    class FakeStudy:
        """Atrapa study wymuszająca pruning w objective."""

        best_value = 0.0
        best_params: dict[str, float] = {}

        def optimize(self, objective: Any, n_trials: int) -> None:
            """Uruchom objective i oczekuj przerwania próby."""
            assert n_trials == 1
            with pytest.raises(RuntimeError):
                objective(FakeTrial())

    class FakeSamplers:
        """Przestrzeń nazw z atrapą TPESampler."""

        @staticmethod
        def TPESampler(seed: int) -> object:
            """Zwróć obiekt sentynelowy dla seedowanego samplera."""
            assert seed == 42
            return object()

    class FakeOptuna:
        """Atrapa modułu Optuna z wyjątkiem pruningowym."""

        samplers = FakeSamplers()
        TrialPruned = RuntimeError

        class pruners:
            """Przestrzeń nazw z atrapą prunerów."""

            @staticmethod
            def MedianPruner(n_startup_trials: int, n_warmup_steps: int) -> object:
                """Zwróć obiekt sentynelowy dla MedianPruner."""
                assert n_startup_trials == 5
                assert n_warmup_steps == 100000
                return object()

        @staticmethod
        def create_study(
            direction: str,
            study_name: str,
            sampler: object,
            pruner: object,
            storage: str,
            load_if_exists: bool,
        ) -> FakeStudy:
            """Zwróć atrapę study dla scenariusza pruningowego."""
            assert direction == "maximize"
            assert study_name == "humanoid_pruned"
            assert sampler is not None
            assert pruner is not None
            assert storage == "sqlite:///data/humanoid_optuna.db"
            assert load_if_exists is True
            return FakeStudy()

    with (
        patch("src.humanoid_bayes._require_optuna", return_value=FakeOptuna()),
        patch(
            "src.humanoid_bayes.run_prunable_humanoid_experiment",
            side_effect=HumanoidTrialPruned(
                {
                    "mean_reward": 11.0,
                    "std_reward": 2.0,
                    "objective_score": 10.8,
                    "training_time_s": 33.0,
                    "trained_timesteps": 100000.0,
                }
            ),
        ),
        patch("src.humanoid_bayes.append_humanoid_result") as mock_append,
        patch("src.humanoid_bayes.time.sleep"),
    ):
        summary = run_humanoid_study(
            trials=1,
            total_timesteps=300000,
            results_csv="data/humanoid.csv",
            study_name="humanoid_pruned",
            startup_trials=5,
            pruner_warmup_steps=100000,
            report_interval_timesteps=100000,
            eval_episodes=20,
            stability_penalty=0.1,
            optuna_storage="sqlite:///data/humanoid_optuna.db",
        )

    assert summary["best_value"] == pytest.approx(0.0)
    assert summary["best_params"] == {}
    _, _, appended_metrics = mock_append.call_args.args
    assert appended_metrics["status"] == "pruned"


def test_main_calls_run_humanoid_study(monkeypatch: pytest.MonkeyPatch) -> None:
    """Zweryfikuj delegację argumentów CLI do run_humanoid_study.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture do chwilowej podmiany ``sys.argv``.
    """
    monkeypatch.setattr(
        "sys.argv",
        [
            "humanoid_bayes",
            "--trials",
            "4",
            "--timesteps",
            "500000",
            "--results-csv",
            DEFAULT_RESULTS_CSV,
            "--study-name",
            "humanoid_cli",
            "--startup-trials",
            "6",
            "--pruner-warmup-steps",
            "200000",
            "--report-interval-timesteps",
            "50000",
            "--eval-episodes",
            "30",
            "--stability-penalty",
            "0.2",
            "--optuna-storage",
            "sqlite:///data/custom_humanoid_optuna.db",
        ],
    )

    with patch("src.humanoid_bayes.run_humanoid_study") as mock_run:
        mock_run.return_value = {"best_value": 1.0, "best_params": {}}
        main()

    mock_run.assert_called_once_with(
        trials=4,
        total_timesteps=500000,
        results_csv=DEFAULT_RESULTS_CSV,
        study_name="humanoid_cli",
        startup_trials=6,
        pruner_warmup_steps=200000,
        report_interval_timesteps=50000,
        eval_episodes=30,
        stability_penalty=0.2,
        optuna_storage="sqlite:///data/custom_humanoid_optuna.db",
    )


def test_main_uses_recommended_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Zweryfikuj domyślne parametry CLI po włączeniu pruningowego workflow."""
    monkeypatch.setattr("sys.argv", ["humanoid_bayes"])

    with patch("src.humanoid_bayes.run_humanoid_study") as mock_run:
        mock_run.return_value = {"best_value": 1.0, "best_params": {}}
        main()

    mock_run.assert_called_once_with(
        trials=DEFAULT_TRIALS,
        total_timesteps=DEFAULT_TIMESTEPS,
        results_csv=DEFAULT_RESULTS_CSV,
        study_name="humanoid_256x256_bayes",
        startup_trials=DEFAULT_STARTUP_TRIALS,
        pruner_warmup_steps=DEFAULT_PRUNER_WARMUP_STEPS,
        report_interval_timesteps=DEFAULT_REPORT_INTERVAL_TIMESTEPS,
        eval_episodes=DEFAULT_EVAL_EPISODES,
        stability_penalty=DEFAULT_STABILITY_PENALTY,
        optuna_storage=DEFAULT_OPTUNA_STORAGE,
    )
