"""Dedykowany runner optymalizacji bayesowskiej PPO dla Humanoid."""

from __future__ import annotations

import argparse
import csv
import importlib
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, cast

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.training import MODELS_DIR, TENSORBOARD_LOG_DIR, get_cooldown_seconds

HUMANOID_ENV_ID = "Humanoid-v5"
HUMANOID_NET_ARCH = [512, 512]
DEFAULT_RESULTS_CSV = "data/humanoid_bayes_results.csv"
DEFAULT_TRIALS = 50
DEFAULT_TIMESTEPS = 1_000_000
DEFAULT_PRUNER_WARMUP_STEPS = 100_000
DEFAULT_STARTUP_TRIALS = 5
DEFAULT_REPORT_INTERVAL_TIMESTEPS = 100_000
DEFAULT_EVAL_EPISODES = 20
DEFAULT_STABILITY_PENALTY = 0.1
DEFAULT_OPTUNA_STORAGE = "sqlite:///data/humanoid_optuna.db"
ResultValue = bool | float | str


class HumanoidTrialPruned(RuntimeError):
    """Wyjątek pomocniczy przenoszący metryki przerwanej próby."""

    def __init__(self, metrics: dict[str, float]) -> None:
        """Zainicjalizuj wyjątek z ostatnimi znanymi metrykami próby.

        Parameters
        ----------
        metrics : dict[str, float]
            Ostatnie metryki uzyskane przed przerwaniem próby.
        """
        super().__init__("Humanoid trial pruned")
        self.metrics = metrics


class TrialProtocol(Protocol):
    """Minimalny kontrakt triala wymagany przez moduł.

    Definiuje wyłącznie te metody Optuny, które są wykorzystywane
    do próbkowania hiperparametrów PPO.
    """

    number: int

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        *,
        log: bool = False,
    ) -> float:
        """Zwróć wartość zmiennoprzecinkową ze zdefiniowanego zakresu."""

    def suggest_categorical(self, name: str, choices: list[Any]) -> Any:
        """Zwróć wartość kategoryczną ze zdefiniowanego zbioru."""

    def report(self, value: float, step: int) -> None:
        """Zgłoś wynik pośredni do mechanizmu pruning."""

    def should_prune(self) -> bool:
        """Zwróć informację, czy próba powinna zostać przerwana."""


def _require_optuna() -> Any:
    """Załaduj Optunę leniwie, aby nie obciążać bazowego pipeline'u.

    Returns
    -------
    Any
        Moduł ``optuna``.

    Raises
    ------
    ModuleNotFoundError
        Jeśli opcjonalne zależności Humanoida nie są zainstalowane.
    """
    try:
        optuna = importlib.import_module("optuna")
    except ModuleNotFoundError as exc:  # pragma: no cover - zależne od środowiska
        raise ModuleNotFoundError(
            "Brakuje opcjonalnej zależności 'optuna'. "
            "Zainstaluj: pip install -r requirements-humanoid.txt"
        ) from exc

    return optuna


def build_humanoid_config(
    trial_number: int,
    learning_rate: float,
    batch_size: int,
    gamma: float,
    n_steps: int,
    ent_coef: float,
    gae_lambda: float,
    clip_range: float,
    target_kl: float,
    n_epochs: int,
    vf_coef: float,
    normalize_advantage: bool,
    total_timesteps: int,
) -> dict[str, object]:
    """Zbuduj konfigurację pojedynczej próby dla Humanoid.

    Architektura sieci jest stała i wynosi ``[512, 512]``. Dzięki temu
    eksperyment 5 bada wyłącznie hiperparametry, a nie pojemność modelu.

    Parameters
    ----------
    trial_number : int
        Numer próby zwracany przez Optunę.
    learning_rate : float
        Współczynnik uczenia PPO.
    batch_size : int
        Wielkość batcha PPO.
    gamma : float
        Współczynnik dyskontowania PPO.
    n_steps : int
        Rozmiar rollout buffer PPO.
    ent_coef : float
        Współczynnik entropii PPO.
    gae_lambda : float
        Parametr GAE kontrolujący kompromis bias-variance.
    clip_range : float
        Zakres clipowania aktualizacji PPO.
    target_kl : float
        Docelowa dywergencja KL ograniczająca zbyt agresywne aktualizacje.
    n_epochs : int
        Liczba epok optymalizacji na jednym rollout bufferze.
    vf_coef : float
        Waga składnika value loss w funkcji celu PPO.
    normalize_advantage : bool
        Flaga normalizacji advantage przed aktualizacją polityki.
    total_timesteps : int
        Budżet kroków treningowych dla jednej próby.

    Returns
    -------
    dict[str, object]
        Słownik zgodny z wejściem ``src.training.run_experiment``.
    """
    return {
        "experiment_id": f"hum_512x512_trial_{trial_number:03d}",
        "env_id": HUMANOID_ENV_ID,
        "net_arch": str(HUMANOID_NET_ARCH),
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "gamma": gamma,
        "n_steps": n_steps,
        "ent_coef": ent_coef,
        "gae_lambda": gae_lambda,
        "clip_range": clip_range,
        "target_kl": target_kl,
        "n_epochs": n_epochs,
        "vf_coef": vf_coef,
        "normalize_advantage": normalize_advantage,
        "total_timesteps": total_timesteps,
        "mean_reward": None,
        "std_reward": None,
        "training_time_s": None,
    }


def sample_humanoid_hyperparameters(trial: TrialProtocol) -> dict[str, float | int]:
    """Wylosuj bezpieczny zestaw hiperparametrów PPO dla Humanoida.

    Dobór przestrzeni wyszukiwania respektuje zależność ``n_steps`` i
    ``batch_size`` (DKB-003) oraz ograniczenia CPU na MacBook Air M4.

    Parameters
    ----------
    trial : TrialProtocol
        Obiekt triala Optuny.

    Returns
    -------
    dict[str, float | int]
        Wylosowane hiperparametry PPO.
    """
    n_steps = int(trial.suggest_categorical("n_steps", [1024, 2048, 4096]))
    valid_batch_sizes = [size for size in (64, 128, 256, 512) if n_steps % size == 0]

    return {
        "learning_rate": float(
            trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        ),
        "batch_size": int(
            trial.suggest_categorical("batch_size", valid_batch_sizes)
        ),
        "gamma": float(trial.suggest_categorical("gamma", [0.96, 0.97, 0.99])),
        "n_steps": n_steps,
        "ent_coef": float(trial.suggest_float("ent_coef", 1e-4, 1e-2, log=True)),
        "gae_lambda": float(
            trial.suggest_categorical("gae_lambda", [0.95, 0.966, 0.99])
        ),
        "clip_range": float(
            trial.suggest_categorical("clip_range", [0.1, 0.15, 0.2, 0.3])
        ),
        "target_kl": float(trial.suggest_float("target_kl", 0.003, 0.03)),
        "n_epochs": int(trial.suggest_categorical("n_epochs", [10, 15, 20])),
        "vf_coef": float(trial.suggest_categorical("vf_coef", [0.35, 0.5, 0.75])),
        "normalize_advantage": bool(
            trial.suggest_categorical("normalize_advantage", [True, False])
        ),
    }


def compute_humanoid_objective_score(
    mean_reward: float,
    std_reward: float,
    stability_penalty: float,
) -> float:
    """Policz wynik optymalizacji łączący jakość i stabilność polityki.

    Parameters
    ----------
    mean_reward : float
        Średnia nagroda z ewaluacji.
    std_reward : float
        Odchylenie standardowe nagrody z ewaluacji.
    stability_penalty : float
        Współczynnik kary za niestabilność polityki.

    Returns
    -------
    float
        Wynik celu optymalizacji: średnia nagroda pomniejszona o karę wariancyjną.
    """
    return mean_reward - (stability_penalty * std_reward)


def append_humanoid_result(
    csv_path: str,
    config: dict[str, object],
    metrics: Mapping[str, ResultValue],
) -> None:
    """Dopisz wynik pojedynczej próby Optuny do dedykowanego CSV.

    Parameters
    ----------
    csv_path : str
        Ścieżka do pliku CSV wyników dla Humanoida.
    config : dict[str, object]
        Konfiguracja pojedynczej próby.
    metrics : dict[str, float]
        Metryki zwrócone przez ``run_experiment``.
    """
    output_path = Path(csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "experiment_id",
        "env_id",
        "net_arch",
        "learning_rate",
        "batch_size",
        "gamma",
        "n_steps",
        "ent_coef",
        "gae_lambda",
        "clip_range",
        "target_kl",
        "n_epochs",
        "vf_coef",
        "normalize_advantage",
        "total_timesteps",
        "trained_timesteps",
        "status",
        "objective_score",
        "mean_reward",
        "std_reward",
        "training_time_s",
    ]

    row = {**config, **metrics}
    file_exists = output_path.exists()

    with output_path.open("a", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _build_humanoid_model(
    config: dict[str, object],
) -> tuple[PPO, VecNormalize]:
    """Utwórz model PPO i znormalizowane środowisko dla Humanoida.

    Parameters
    ----------
    config : dict[str, object]
        Konfiguracja pojedynczej próby.

    Returns
    -------
    tuple[PPO, VecNormalize]
        Zainicjalizowany model PPO oraz środowisko z VecNormalize.
    """
    base_env = DummyVecEnv([lambda: Monitor(gym.make(str(config["env_id"])))])
    env = VecNormalize(
        base_env,
        norm_obs=True,
        norm_reward=True,
        gamma=float(cast(float | int | str, config["gamma"])),
    )

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=float(cast(float | int | str, config["learning_rate"])),
        batch_size=int(cast(int | str, config["batch_size"])),
        gamma=float(cast(float | int | str, config["gamma"])),
        n_steps=int(cast(int | str, config["n_steps"])),
        gae_lambda=float(cast(float | int | str, config["gae_lambda"])),
        clip_range=float(cast(float | int | str, config["clip_range"])),
        ent_coef=float(cast(float | int | str, config["ent_coef"])),
        vf_coef=float(cast(float | int | str, config["vf_coef"])),
        n_epochs=int(cast(int | str, config["n_epochs"])),
        normalize_advantage=bool(cast(bool, config["normalize_advantage"])),
        target_kl=float(cast(float | int | str, config["target_kl"])),
        policy_kwargs={"net_arch": HUMANOID_NET_ARCH},
        tensorboard_log=TENSORBOARD_LOG_DIR,
        verbose=0,
        device="auto",
    )
    return model, env


def _save_vecnormalize_state(env: VecNormalize, config: dict[str, object]) -> None:
    """Zapisz statystyki normalizacji dla bieżącej próby Humanoida.

    Parameters
    ----------
    env : VecNormalize
        Środowisko z normalizacją obserwacji i nagród.
    config : dict[str, object]
        Konfiguracja pojedynczej próby.
    """
    vecnormalize_path = Path(MODELS_DIR) / f"{config['experiment_id']}_vecnormalize.pkl"
    env.save(str(vecnormalize_path))


def run_prunable_humanoid_experiment(
    config: dict[str, object],
    trial: TrialProtocol,
    report_interval_timesteps: int,
    eval_episodes: int,
    stability_penalty: float,
) -> dict[str, float]:
    """Uruchom próbę Humanoida z raportowaniem pośrednim do Optuny.

    Trening jest dzielony na etapy, aby pruner Optuny mógł realnie odciąć
    słabe konfiguracje przed wyczerpaniem pełnego budżetu kroków.

    Parameters
    ----------
    config : dict[str, object]
        Konfiguracja pojedynczej próby.
    trial : TrialProtocol
        Trial Optuny używany do raportowania wyników pośrednich.
    report_interval_timesteps : int
        Liczba kroków między kolejnymi checkpointami pruningowymi.
    eval_episodes : int
        Liczba epizodów używana do każdej ewaluacji pośredniej.
    stability_penalty : float
        Kara za niestabilność polityki używana w funkcji celu.

    Returns
    -------
    dict[str, float]
        Końcowe metryki treningu, w tym liczba faktycznie wykonanych kroków.

    Raises
    ------
    HumanoidTrialPruned
        Jeśli trial został oznaczony do przerwania przez pruner.
    """
    Path(MODELS_DIR).mkdir(exist_ok=True)

    model, env = _build_humanoid_model(config)
    total_timesteps = int(cast(int | str, config["total_timesteps"]))
    trained_timesteps = 0
    start_time = time.time()
    latest_metrics: dict[str, float] = {
        "mean_reward": 0.0,
        "std_reward": 0.0,
        "objective_score": 0.0,
        "training_time_s": 0.0,
        "trained_timesteps": 0.0,
    }

    try:
        print(
            f"[TRIAL] {config['experiment_id']} start "
            f"total_timesteps={total_timesteps} "
            f"report_interval={report_interval_timesteps} "
            f"eval_episodes={eval_episodes}"
        )
        while trained_timesteps < total_timesteps:
            learn_timesteps = min(
                report_interval_timesteps,
                total_timesteps - trained_timesteps,
            )
            model.learn(
                total_timesteps=learn_timesteps,
                tb_log_name=str(config["experiment_id"]),
                reset_num_timesteps=trained_timesteps == 0,
            )
            trained_timesteps += learn_timesteps

            mean_reward, std_reward = cast(
                tuple[float, float],
                evaluate_policy(
                    model,
                    env,
                    n_eval_episodes=eval_episodes,
                    deterministic=True,
                ),
            )
            objective_score = compute_humanoid_objective_score(
                mean_reward=float(mean_reward),
                std_reward=float(std_reward),
                stability_penalty=stability_penalty,
            )
            latest_metrics = {
                "mean_reward": float(mean_reward),
                "std_reward": float(std_reward),
                "objective_score": float(objective_score),
                "training_time_s": round(time.time() - start_time, 2),
                "trained_timesteps": float(trained_timesteps),
            }
            progress_percent = (trained_timesteps / total_timesteps) * 100
            print(
                f"[TRIAL] {config['experiment_id']} "
                f"progress={trained_timesteps}/{total_timesteps} "
                f"({progress_percent:.1f}%) "
                f"mean={mean_reward:.2f} std={std_reward:.2f} "
                f"objective={objective_score:.2f}"
            )
            trial.report(float(objective_score), step=trained_timesteps)
            if trial.should_prune():
                raise HumanoidTrialPruned(latest_metrics)

        model.save(f"{MODELS_DIR}/{config['experiment_id']}")
        _save_vecnormalize_state(env, config)
        return latest_metrics
    finally:
        env.close()  # type: ignore[no-untyped-call]


def build_budget_warning(trials: int, total_timesteps: int) -> str | None:
    """Zwróć ostrzeżenie dla zbyt małego budżetu wyszukiwania Humanoida.

    Parameters
    ----------
    trials : int
        Liczba prób Optuny.
    total_timesteps : int
        Budżet kroków na jedną próbę.

    Returns
    -------
    str | None
        Tekst ostrzeżenia albo ``None``, jeśli budżet nie wygląda podejrzanie.
    """
    if trials < 11 and total_timesteps < 1_000_000:
        return (
            "[WARN] Budżet wygląda jak smoke test, nie jak sensowne strojenie "
            "Humanoida: TPE zaczyna realnie wykorzystywać historię po fazie "
            "startup, a 300k kroków zwykle nie wystarcza dla MuJoCo Humanoid."
        )
    return None


def run_humanoid_study(
    trials: int,
    total_timesteps: int,
    results_csv: str,
    study_name: str,
    startup_trials: int,
    pruner_warmup_steps: int,
    report_interval_timesteps: int,
    eval_episodes: int,
    stability_penalty: float,
    optuna_storage: str,
) -> dict[str, object]:
    """Uruchom optymalizację bayesowską PPO dla środowiska Humanoid.

    Każda próba używa tej samej architektury ``[512, 512]``. Zmieniane są
    wyłącznie hiperparametry PPO, a wyniki są dopisywane do osobnego CSV.

    Parameters
    ----------
    trials : int
        Liczba prób Optuny.
    total_timesteps : int
        Liczba kroków treningowych na próbę.
    results_csv : str
        Ścieżka do CSV z wynikami wszystkich prób.
    study_name : str
        Nazwa studium Optuny.
    startup_trials : int
        Liczba prób rozruchowych prunera i samplera TPE.
    pruner_warmup_steps : int
        Liczba kroków treningowych ignorowanych przez pruner.
    report_interval_timesteps : int
        Liczba kroków pomiędzy raportami pośrednimi do Optuny.
    eval_episodes : int
        Liczba epizodów w pojedynczej ewaluacji triala.
    stability_penalty : float
        Kara za niestabilność polityki w funkcji celu Optuny.
    optuna_storage : str
        URI storage Optuny (np. SQLite) do wznawiania i analizy studium.

    Returns
    -------
    dict[str, object]
        Podsumowanie najlepszego wyniku: ``best_value`` i ``best_params``.
    """
    optuna = _require_optuna()
    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=startup_trials,
        n_warmup_steps=pruner_warmup_steps,
    )
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        sampler=sampler,
        pruner=pruner,
        storage=optuna_storage,
        load_if_exists=True,
    )

    def objective(trial: TrialProtocol) -> float:
        params = sample_humanoid_hyperparameters(trial)
        config = build_humanoid_config(
            trial_number=trial.number,
            learning_rate=float(cast(float, params["learning_rate"])),
            batch_size=int(cast(int, params["batch_size"])),
            gamma=float(cast(float, params["gamma"])),
            n_steps=int(cast(int, params["n_steps"])),
            ent_coef=float(cast(float, params["ent_coef"])),
            gae_lambda=float(cast(float, params["gae_lambda"])),
            clip_range=float(cast(float, params["clip_range"])),
            target_kl=float(cast(float, params["target_kl"])),
            n_epochs=int(cast(int, params["n_epochs"])),
            vf_coef=float(cast(float, params["vf_coef"])),
            normalize_advantage=bool(cast(bool, params["normalize_advantage"])),
            total_timesteps=total_timesteps,
        )
        try:
            metrics = run_prunable_humanoid_experiment(
                config=config,
                trial=trial,
                report_interval_timesteps=report_interval_timesteps,
                eval_episodes=eval_episodes,
                stability_penalty=stability_penalty,
            )
        except HumanoidTrialPruned as exc:
            append_humanoid_result(
                results_csv,
                config,
                {**exc.metrics, "status": "pruned"},
            )
            cooldown_s = get_cooldown_seconds(HUMANOID_NET_ARCH)
            print(
                f"[TRIAL] {config['experiment_id']} pruned "
                f"objective={exc.metrics['objective_score']:.2f} "
                f"cooldown={cooldown_s}s"
            )
            time.sleep(cooldown_s)
            raise optuna.TrialPruned() from exc

        append_humanoid_result(
            results_csv,
            config,
            {**metrics, "status": "completed"},
        )
        cooldown_s = get_cooldown_seconds(HUMANOID_NET_ARCH)
        print(
            f"[TRIAL] {config['experiment_id']} completed "
            f"objective={metrics['objective_score']:.2f} "
            f"cooldown={cooldown_s}s"
        )
        time.sleep(cooldown_s)
        return metrics["objective_score"]

    study.optimize(objective, n_trials=trials)

    return {
        "best_value": float(study.best_value),
        "best_params": dict(study.best_params),
    }


def main() -> None:
    """Uruchom CLI optymalizacji bayesowskiej dla Humanoida."""
    parser = argparse.ArgumentParser(
        description=(
            "Optymalizacja bayesowska PPO dla Humanoid-v5 "
            "na stałej architekturze [512, 512]."
        )
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=DEFAULT_TRIALS,
        help="Liczba prób optymalizacji bayesowskiej.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=DEFAULT_TIMESTEPS,
        help="Liczba kroków treningowych na jedną próbę.",
    )
    parser.add_argument(
        "--results-csv",
        default=DEFAULT_RESULTS_CSV,
        help="Ścieżka do CSV, do którego będą dopisywane wyniki prób.",
    )
    parser.add_argument(
        "--study-name",
        default="humanoid_512x512_bayes",
        help="Nazwa studium Optuny.",
    )
    parser.add_argument(
        "--startup-trials",
        type=int,
        default=DEFAULT_STARTUP_TRIALS,
        help="Liczba prób startup dla TPE i prunera MedianPruner.",
    )
    parser.add_argument(
        "--pruner-warmup-steps",
        type=int,
        default=DEFAULT_PRUNER_WARMUP_STEPS,
        help="Liczba kroków ignorowanych przez pruner przed pierwszym cięciem.",
    )
    parser.add_argument(
        "--report-interval-timesteps",
        type=int,
        default=DEFAULT_REPORT_INTERVAL_TIMESTEPS,
        help="Liczba kroków między raportami pośrednimi do Optuny.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=DEFAULT_EVAL_EPISODES,
        help="Liczba epizodów ewaluacyjnych po każdym etapie triala.",
    )
    parser.add_argument(
        "--stability-penalty",
        type=float,
        default=DEFAULT_STABILITY_PENALTY,
        help="Kara za odchylenie standardowe w funkcji celu Optuny.",
    )
    parser.add_argument(
        "--optuna-storage",
        default=DEFAULT_OPTUNA_STORAGE,
        help=(
            "URI storage Optuny (np. sqlite:///data/humanoid_optuna.db) "
            "używane do resume i dashboardu."
        ),
    )
    args = parser.parse_args()

    warning = build_budget_warning(args.trials, args.timesteps)
    if warning is not None:
        print(warning)

    summary = run_humanoid_study(
        trials=args.trials,
        total_timesteps=args.timesteps,
        results_csv=args.results_csv,
        study_name=args.study_name,
        startup_trials=args.startup_trials,
        pruner_warmup_steps=args.pruner_warmup_steps,
        report_interval_timesteps=args.report_interval_timesteps,
        eval_episodes=args.eval_episodes,
        stability_penalty=args.stability_penalty,
        optuna_storage=args.optuna_storage,
    )
    print(
        "[BEST] "
        f"value={summary['best_value']:.3f} "
        f"params={summary['best_params']}"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
