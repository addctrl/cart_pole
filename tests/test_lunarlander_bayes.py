"""Testy jednostkowe modulu ``src.lunarlander_bayes``."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.lunarlander_bayes import (
    DEFAULT_EVAL_EPISODES,
    DEFAULT_OPTUNA_STORAGE,
    DEFAULT_PRUNER_WARMUP_STEPS,
    DEFAULT_REPORT_INTERVAL_TIMESTEPS,
    DEFAULT_RESULTS_CSV,
    DEFAULT_STABILITY_PENALTY,
    DEFAULT_STARTUP_TRIALS,
    DEFAULT_STUDY_NAME,
    DEFAULT_TIMESTEPS,
    DEFAULT_TRIALS,
    LUNARLANDER_ENV_ID,
    LunarLanderTrialPruned,
    _require_optuna,
    append_lunarlander_result,
    build_budget_warning,
    build_lunarlander_config,
    compute_lunarlander_objective_score,
    main,
    resolve_lunarlander_net_arch,
    run_lunarlander_study,
    run_prunable_lunarlander_experiment,
    sample_lunarlander_hyperparameters,
)


class FakeTrial:
    """Minimalna atrapa triala Optuny do testow jednostkowych."""

    def __init__(self) -> None:
        """Zainicjalizuj atrapy zwracajace deterministyczne wartosci."""
        self.number = 7

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        *,
        log: bool = False,
    ) -> float:
        """Zwroc deterministyczna wartosc zgodna z nazwa parametru."""
        assert high >= low
        assert isinstance(log, bool)
        return {
            "target_kl": 0.02,
        }[name]

    def suggest_categorical(self, name: str, choices: list[Any]) -> Any:
        """Zwroc deterministyczny wybor ze zbioru kategorycznego."""
        selected = {
            "net_arch_label": "64x64",
            "learning_rate": 0.0003,
            "n_steps": 2048,
            "batch_size": 256,
            "gamma": 0.99,
            "ent_coef": 0.01,
            "gae_lambda": 0.97,
            "clip_range": 0.2,
            "n_epochs": 10,
            "vf_coef": 0.5,
            "normalize_advantage": True,
        }[name]
        assert selected in choices
        return selected

    def report(self, value: float, step: int) -> None:
        """Przyjmij wynik posredni bez dodatkowej logiki."""
        assert step >= 0
        assert isinstance(value, float)

    def should_prune(self) -> bool:
        """Nigdy nie przerywaj proby w podstawowej atrapie."""
        return False


class FakePruningTrial(FakeTrial):
    """Atrapa triala, ktora wymusza pruning po pierwszym raporcie."""

    def __init__(self) -> None:
        """Zainicjalizuj flage wykrycia raportu posredniego."""
        super().__init__()
        self.has_report = False

    def report(self, value: float, step: int) -> None:
        """Zapamietaj fakt wyslania raportu posredniego."""
        super().report(value, step)
        self.has_report = True

    def should_prune(self) -> bool:
        """Przerwij probe po pierwszym raporcie posrednim."""
        return self.has_report


def test_require_optuna_returns_module_when_available() -> None:
    """Zweryfikuj zwrocenie modulu Optuna przy poprawnej instalacji."""

    class FakeOptunaModule:
        """Atrapa modulu Optuna."""

    with patch(
        "src.lunarlander_bayes.importlib.import_module",
        return_value=FakeOptunaModule(),
    ):
        module = _require_optuna()

    assert isinstance(module, FakeOptunaModule)


def test_require_optuna_raises_clear_error_when_missing() -> None:
    """Zweryfikuj czytelny komunikat przy braku Optuny."""
    with patch(
        "src.lunarlander_bayes.importlib.import_module",
        side_effect=ModuleNotFoundError("optuna missing"),
    ):
        with pytest.raises(ModuleNotFoundError, match="requirements-humanoid.txt"):
            _require_optuna()


def test_resolve_lunarlander_net_arch_returns_expected_layers() -> None:
    """Zweryfikuj mapowanie etykiety architektury na warstwy."""
    assert resolve_lunarlander_net_arch("64x64") == [64, 64]
    assert resolve_lunarlander_net_arch("128x128") == [128, 128]


def test_resolve_lunarlander_net_arch_raises_for_unknown_label() -> None:
    """Zweryfikuj blad dla nieznanej etykiety architektury."""
    with pytest.raises(ValueError, match="Nieznana architektura"):
        resolve_lunarlander_net_arch("256x256")


def test_build_lunarlander_config_uses_requested_variant() -> None:
    """Zweryfikuj srodowisko i architekture wariantu LunarLandera."""
    config = build_lunarlander_config(
        trial_number=3,
        net_arch_label="128x128",
        learning_rate=0.0003,
        batch_size=256,
        gamma=0.99,
        n_steps=2048,
        ent_coef=0.01,
        gae_lambda=0.97,
        clip_range=0.2,
        target_kl=0.02,
        n_epochs=10,
        vf_coef=0.5,
        normalize_advantage=True,
        total_timesteps=300000,
    )

    assert config["experiment_id"] == "ll_pre5_128x128_trial_003"
    assert config["env_id"] == LUNARLANDER_ENV_ID
    assert config["net_arch"] == "[128, 128]"
    assert config["net_arch_label"] == "128x128"
    assert config["target_kl"] == pytest.approx(0.02)
    assert config["normalize_advantage"] is True


def test_sample_lunarlander_hyperparameters_preserves_divisibility() -> None:
    """Zweryfikuj zgodnosc batch_size z ograniczeniem DKB-003."""
    params = sample_lunarlander_hyperparameters(FakeTrial())

    assert params["net_arch_label"] == "64x64"
    assert params["n_steps"] == 2048
    assert params["batch_size"] == 256
    assert params["clip_range"] == pytest.approx(0.2)
    assert params["target_kl"] == pytest.approx(0.02)
    assert params["normalize_advantage"] is True
    assert int(params["n_steps"]) % int(params["batch_size"]) == 0


def test_compute_lunarlander_objective_score_penalizes_variance() -> None:
    """Zweryfikuj scoring stabilnosciowy zgodny z kara wariancyjna."""
    score = compute_lunarlander_objective_score(120.0, 15.0, 0.1)

    assert score == pytest.approx(118.5)


def test_append_lunarlander_result_creates_csv(tmp_path: Path) -> None:
    """Zweryfikuj utworzenie i zapis jednego wiersza wynikowego."""
    csv_path = tmp_path / "lunarlander.csv"
    config = build_lunarlander_config(
        trial_number=1,
        net_arch_label="64x64",
        learning_rate=0.0003,
        batch_size=256,
        gamma=0.99,
        n_steps=2048,
        ent_coef=0.01,
        gae_lambda=0.97,
        clip_range=0.2,
        target_kl=0.02,
        n_epochs=10,
        vf_coef=0.5,
        normalize_advantage=True,
        total_timesteps=300000,
    )
    metrics = {"mean_reward": 123.4, "std_reward": 5.6, "training_time_s": 78.9}

    append_lunarlander_result(str(csv_path), config, metrics)

    content = csv_path.read_text(encoding="utf-8")
    assert "experiment_id,env_id,net_arch,net_arch_label" in content
    assert "ll_pre5_64x64_trial_001" in content
    assert "123.4" in content
    assert "normalize_advantage" in content


def test_build_budget_warning_for_smoke_budget() -> None:
    """Zweryfikuj ostrzezenie dla zbyt malego budzetu pre5."""
    warning = build_budget_warning(10, 200000)

    assert warning is not None
    assert "smoke test" in warning


def test_build_budget_warning_returns_none_for_reasonable_budget() -> None:
    """Zweryfikuj brak ostrzezenia dla wiekszego budzetu strojenia."""
    assert build_budget_warning(40, 300000) is None


def test_run_prunable_lunarlander_experiment_returns_metrics() -> None:
    """Zweryfikuj etapowy trening LunarLandera bez przerwania proby."""
    config = build_lunarlander_config(
        trial_number=1,
        net_arch_label="64x64",
        learning_rate=0.0003,
        batch_size=256,
        gamma=0.99,
        n_steps=2048,
        ent_coef=0.01,
        gae_lambda=0.97,
        clip_range=0.2,
        target_kl=0.02,
        n_epochs=10,
        vf_coef=0.5,
        normalize_advantage=True,
        total_timesteps=100000,
    )
    mock_env = MagicMock()

    with (
        patch("src.lunarlander_bayes.gym.make", return_value=mock_env),
        patch("src.lunarlander_bayes.Monitor", side_effect=lambda env: env),
        patch("src.lunarlander_bayes.PPO") as mock_ppo,
        patch(
            "src.lunarlander_bayes.evaluate_policy",
            side_effect=[(10.0, 1.0), (20.0, 2.0)],
        ),
        patch("src.lunarlander_bayes.time.time", side_effect=[100.0, 130.0, 160.0]),
    ):
        mock_model = mock_ppo.return_value
        metrics = run_prunable_lunarlander_experiment(
            config=config,
            trial=FakeTrial(),
            report_interval_timesteps=50000,
            eval_episodes=20,
            stability_penalty=0.1,
        )

    assert metrics["mean_reward"] == pytest.approx(20.0)
    assert metrics["std_reward"] == pytest.approx(2.0)
    assert metrics["objective_score"] == pytest.approx(19.8)
    assert metrics["trained_timesteps"] == pytest.approx(100000.0)
    assert mock_ppo.call_count == 1
    assert mock_model.learn.call_count == 2
    mock_model.save.assert_called_once_with("models/ll_pre5_64x64_trial_001")


def test_run_prunable_lunarlander_experiment_raises_pruned() -> None:
    """Zweryfikuj przerwanie proby po raporcie posrednim do Optuny."""
    config = build_lunarlander_config(
        trial_number=2,
        net_arch_label="128x128",
        learning_rate=0.0003,
        batch_size=256,
        gamma=0.99,
        n_steps=2048,
        ent_coef=0.01,
        gae_lambda=0.97,
        clip_range=0.2,
        target_kl=0.02,
        n_epochs=10,
        vf_coef=0.5,
        normalize_advantage=True,
        total_timesteps=100000,
    )
    mock_env = MagicMock()

    with (
        patch("src.lunarlander_bayes.gym.make", return_value=mock_env),
        patch("src.lunarlander_bayes.Monitor", side_effect=lambda env: env),
        patch("src.lunarlander_bayes.PPO") as mock_ppo,
        patch("src.lunarlander_bayes.evaluate_policy", return_value=(10.0, 1.0)),
        patch("src.lunarlander_bayes.time.time", side_effect=[100.0, 125.0]),
    ):
        with pytest.raises(LunarLanderTrialPruned) as exc_info:
            run_prunable_lunarlander_experiment(
                config=config,
                trial=FakePruningTrial(),
                report_interval_timesteps=50000,
                eval_episodes=20,
                stability_penalty=0.1,
            )

    assert exc_info.value.metrics["mean_reward"] == pytest.approx(10.0)
    assert exc_info.value.metrics["objective_score"] == pytest.approx(9.9)
    assert exc_info.value.metrics["trained_timesteps"] == pytest.approx(50000.0)
    mock_ppo.return_value.save.assert_not_called()


def test_run_lunarlander_study_returns_best_summary() -> None:
    """Zweryfikuj integracje study z dedykowanym objective bez realnego treningu."""

    class FakeStudy:
        """Atrapa obiektu study Optuny."""

        def __init__(self) -> None:
            """Ustaw wartosci koncowe studium."""
            self.best_value = 321.0
            self.best_params = {"net_arch_label": "64x64", "learning_rate": 0.0003}

        def optimize(self, objective: Any, n_trials: int) -> None:
            """Wywolaj objective jeden raz dla deterministycznej proby."""
            assert n_trials == 2
            objective(FakeTrial())

    class FakeSamplers:
        """Przestrzen nazw z atrapa TPESampler."""

        @staticmethod
        def TPESampler(seed: int) -> object:
            """Zwroc obiekt sentynelowy dla seedowanego samplera."""
            assert seed == 42
            return object()

    class FakeOptuna:
        """Atrapa modulu Optuna."""

        samplers = FakeSamplers()
        TrialPruned = RuntimeError

        class pruners:
            """Przestrzen nazw z atrapa prunerow."""

            @staticmethod
            def MedianPruner(n_startup_trials: int, n_warmup_steps: int) -> object:
                """Zwroc obiekt sentynelowy dla MedianPruner."""
                assert n_startup_trials == 8
                assert n_warmup_steps == 50000
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
            """Zwroc atrape studium i zweryfikuj parametry wejsciowe."""
            assert direction == "maximize"
            assert study_name == "lunarlander_test"
            assert sampler is not None
            assert pruner is not None
            assert storage == "sqlite:///data/lunarlander_optuna.db"
            assert load_if_exists is True
            return FakeStudy()

    with (
        patch("src.lunarlander_bayes._require_optuna", return_value=FakeOptuna()),
        patch(
            "src.lunarlander_bayes.run_prunable_lunarlander_experiment",
            return_value={
                "mean_reward": 321.0,
                "std_reward": 12.3,
                "objective_score": 319.77,
                "training_time_s": 45.6,
                "trained_timesteps": 300000.0,
            },
        ),
        patch("src.lunarlander_bayes.append_lunarlander_result"),
        patch("src.lunarlander_bayes.time.sleep"),
    ):
        summary = run_lunarlander_study(
            trials=2,
            total_timesteps=300000,
            results_csv="data/lunarlander.csv",
            study_name="lunarlander_test",
            startup_trials=8,
            pruner_warmup_steps=50000,
            report_interval_timesteps=50000,
            eval_episodes=20,
            stability_penalty=0.1,
            optuna_storage="sqlite:///data/lunarlander_optuna.db",
        )

    assert summary["best_value"] == pytest.approx(321.0)
    assert summary["best_params"] == {
        "net_arch_label": "64x64",
        "learning_rate": 0.0003,
    }


def test_run_lunarlander_study_marks_pruned_trials() -> None:
    """Zweryfikuj zapis statusu `pruned` i propagacje wyjatku Optuny."""

    class FakeStudy:
        """Atrapa study wymuszajaca pruning w objective."""

        best_value = 0.0
        best_params: dict[str, float] = {}

        def optimize(self, objective: Any, n_trials: int) -> None:
            """Uruchom objective i oczekuj przerwania proby."""
            assert n_trials == 1
            with pytest.raises(RuntimeError):
                objective(FakeTrial())

    class FakeSamplers:
        """Przestrzen nazw z atrapa TPESampler."""

        @staticmethod
        def TPESampler(seed: int) -> object:
            """Zwroc obiekt sentynelowy dla seedowanego samplera."""
            assert seed == 42
            return object()

    class FakeOptuna:
        """Atrapa modulu Optuna z wyjatkiem pruningowym."""

        samplers = FakeSamplers()
        TrialPruned = RuntimeError

        class pruners:
            """Przestrzen nazw z atrapa prunerow."""

            @staticmethod
            def MedianPruner(n_startup_trials: int, n_warmup_steps: int) -> object:
                """Zwroc obiekt sentynelowy dla MedianPruner."""
                assert n_startup_trials == 8
                assert n_warmup_steps == 50000
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
            """Zwroc atrape study dla scenariusza pruningowego."""
            assert direction == "maximize"
            assert study_name == "lunarlander_pruned"
            assert sampler is not None
            assert pruner is not None
            assert storage == "sqlite:///data/lunarlander_optuna.db"
            assert load_if_exists is True
            return FakeStudy()

    with (
        patch("src.lunarlander_bayes._require_optuna", return_value=FakeOptuna()),
        patch(
            "src.lunarlander_bayes.run_prunable_lunarlander_experiment",
            side_effect=LunarLanderTrialPruned(
                {
                    "mean_reward": 11.0,
                    "std_reward": 2.0,
                    "objective_score": 10.8,
                    "training_time_s": 33.0,
                    "trained_timesteps": 50000.0,
                }
            ),
        ),
        patch("src.lunarlander_bayes.append_lunarlander_result") as mock_append,
        patch("src.lunarlander_bayes.time.sleep"),
    ):
        summary = run_lunarlander_study(
            trials=1,
            total_timesteps=300000,
            results_csv="data/lunarlander.csv",
            study_name="lunarlander_pruned",
            startup_trials=8,
            pruner_warmup_steps=50000,
            report_interval_timesteps=50000,
            eval_episodes=20,
            stability_penalty=0.1,
            optuna_storage="sqlite:///data/lunarlander_optuna.db",
        )

    assert summary["best_value"] == pytest.approx(0.0)
    assert summary["best_params"] == {}
    _, _, appended_metrics = mock_append.call_args.args
    assert appended_metrics["status"] == "pruned"


def test_main_calls_run_lunarlander_study(monkeypatch: pytest.MonkeyPatch) -> None:
    """Zweryfikuj delegacje argumentow CLI do run_lunarlander_study."""
    monkeypatch.setattr(
        "sys.argv",
        [
            "lunarlander_bayes",
            "--trials",
            "4",
            "--timesteps",
            "500000",
            "--results-csv",
            DEFAULT_RESULTS_CSV,
            "--study-name",
            "lunarlander_cli",
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
            "sqlite:///data/custom_lunarlander_optuna.db",
        ],
    )

    with patch("src.lunarlander_bayes.run_lunarlander_study") as mock_run:
        mock_run.return_value = {"best_value": 1.0, "best_params": {}}
        main()

    mock_run.assert_called_once_with(
        trials=4,
        total_timesteps=500000,
        results_csv=DEFAULT_RESULTS_CSV,
        study_name="lunarlander_cli",
        startup_trials=6,
        pruner_warmup_steps=200000,
        report_interval_timesteps=50000,
        eval_episodes=30,
        stability_penalty=0.2,
        optuna_storage="sqlite:///data/custom_lunarlander_optuna.db",
    )


def test_main_uses_recommended_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Zweryfikuj domyslne parametry CLI dla workflow pre5."""
    monkeypatch.setattr("sys.argv", ["lunarlander_bayes"])

    with patch("src.lunarlander_bayes.run_lunarlander_study") as mock_run:
        mock_run.return_value = {"best_value": 1.0, "best_params": {}}
        main()

    mock_run.assert_called_once_with(
        trials=DEFAULT_TRIALS,
        total_timesteps=DEFAULT_TIMESTEPS,
        results_csv=DEFAULT_RESULTS_CSV,
        study_name=DEFAULT_STUDY_NAME,
        startup_trials=DEFAULT_STARTUP_TRIALS,
        pruner_warmup_steps=DEFAULT_PRUNER_WARMUP_STEPS,
        report_interval_timesteps=DEFAULT_REPORT_INTERVAL_TIMESTEPS,
        eval_episodes=DEFAULT_EVAL_EPISODES,
        stability_penalty=DEFAULT_STABILITY_PENALTY,
        optuna_storage=DEFAULT_OPTUNA_STORAGE,
    )


def test_main_prints_warning_for_smoke_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Zweryfikuj wydruk ostrzezenia przy zbyt malym budzecie CLI."""
    monkeypatch.setattr(
        "sys.argv",
        [
            "lunarlander_bayes",
            "--trials",
            "10",
            "--timesteps",
            "200000",
        ],
    )

    with (
        patch("src.lunarlander_bayes.run_lunarlander_study") as mock_run,
        patch("builtins.print") as mock_print,
    ):
        mock_run.return_value = {"best_value": 1.0, "best_params": {}}
        main()

    assert mock_print.call_args_list[0].args[0].startswith("[WARN]")