"""Dedykowany runner optymalizacji bayesowskiej PPO dla LunarLandera."""

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

from src.training import MODELS_DIR, TENSORBOARD_LOG_DIR, get_cooldown_seconds

LUNARLANDER_ENV_ID = "LunarLander-v3"
LUNARLANDER_NET_ARCH_OPTIONS: dict[str, list[int]] = {
    "64x64": [64, 64],
    "128x128": [128, 128],
}
DEFAULT_RESULTS_CSV = "data/lunarlander_bayes_results.csv"
DEFAULT_STUDY_NAME = "lunarlander_pre5_bayes"
DEFAULT_TRIALS = 40
DEFAULT_TIMESTEPS = 300_000
DEFAULT_PRUNER_WARMUP_STEPS = 50_000
DEFAULT_STARTUP_TRIALS = 8
DEFAULT_REPORT_INTERVAL_TIMESTEPS = 50_000
DEFAULT_EVAL_EPISODES = 20
DEFAULT_STABILITY_PENALTY = 0.1
DEFAULT_OPTUNA_STORAGE = "sqlite:///data/lunarlander_optuna.db"
ResultValue = bool | float | str


class LunarLanderTrialPruned(RuntimeError):
    """Wyjatek pomocniczy przenoszacy metryki przerwanej proby."""

    def __init__(self, metrics: dict[str, float]) -> None:
        """Zainicjalizuj wyjatek z ostatnimi znanymi metrykami proby.

        Parameters
        ----------
        metrics : dict[str, float]
            Ostatnie metryki uzyskane przed przerwaniem proby.
        """
        super().__init__("LunarLander trial pruned")
        self.metrics = metrics


class TrialProtocol(Protocol):
    """Minimalny kontrakt triala wymagany przez modul."""

    number: int

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        *,
        log: bool = False,
    ) -> float:
        """Zwroc wartosc zmiennoprzecinkowa ze zdefiniowanego zakresu."""

    def suggest_categorical(self, name: str, choices: list[Any]) -> Any:
        """Zwroc wartosc kategoryczna ze zdefiniowanego zbioru."""

    def report(self, value: float, step: int) -> None:
        """Zglos wynik posredni do mechanizmu pruning."""

    def should_prune(self) -> bool:
        """Zwroc informacje, czy proba powinna zostac przerwana."""


def _require_optuna() -> Any:
    """Zaladuj Optune leniwie, aby nie obciazac bazowego pipeline'u.

    Returns
    -------
    Any
        Modul ``optuna``.

    Raises
    ------
    ModuleNotFoundError
        Jesli opcjonalna zaleznosc ``optuna`` nie jest zainstalowana.
    """
    try:
        optuna = importlib.import_module("optuna")
    except ModuleNotFoundError as exc:  # pragma: no cover - zalezne od srodowiska
        raise ModuleNotFoundError(
            "Brakuje opcjonalnej zaleznosci 'optuna'. "
            "Zainstaluj: pip install -r requirements-humanoid.txt"
        ) from exc

    return optuna


def resolve_lunarlander_net_arch(net_arch_label: str) -> list[int]:
    """Przeksztalc etykiete architektury na liste warstw ukrytych.

    Parameters
    ----------
    net_arch_label : str
        Etykieta architektury z przestrzeni wyszukiwania.

    Returns
    -------
    list[int]
        Lista rozmiarow warstw ukrytych.

    Raises
    ------
    ValueError
        Jesli etykieta architektury nie jest wspierana.
    """
    try:
        return LUNARLANDER_NET_ARCH_OPTIONS[net_arch_label]
    except KeyError as exc:
        raise ValueError(f"Nieznana architektura LunarLandera: {net_arch_label}") from exc


def build_lunarlander_config(
    trial_number: int,
    net_arch_label: str,
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
    """Zbuduj konfiguracje pojedynczej proby dla LunarLandera.

    Parameters
    ----------
    trial_number : int
        Numer proby zwracany przez Optune.
    net_arch_label : str
        Etykieta architektury sieci porownywanej w pre5.
    learning_rate : float
        Wspolczynnik uczenia PPO.
    batch_size : int
        Wielkosc batcha PPO.
    gamma : float
        Wspolczynnik dyskontowania PPO.
    n_steps : int
        Rozmiar rollout buffer PPO.
    ent_coef : float
        Wspolczynnik entropii PPO.
    gae_lambda : float
        Parametr GAE kontrolujacy kompromis bias-variance.
    clip_range : float
        Zakres clipowania aktualizacji PPO.
    target_kl : float
        Docelowa dywergencja KL ograniczajaca zbyt agresywne aktualizacje.
    n_epochs : int
        Liczba epok optymalizacji na jednym rollout bufferze.
    vf_coef : float
        Waga skladnika value loss w funkcji celu PPO.
    normalize_advantage : bool
        Flaga normalizacji advantage przed aktualizacja polityki.
    total_timesteps : int
        Budzet krokow treningowych dla jednej proby.

    Returns
    -------
    dict[str, object]
        Slownik zgodny z wejsciem ``PPO`` i loggerem wynikow pre5.
    """
    net_arch = resolve_lunarlander_net_arch(net_arch_label)
    return {
        "experiment_id": f"ll_pre5_{net_arch_label}_trial_{trial_number:03d}",
        "env_id": LUNARLANDER_ENV_ID,
        "net_arch": str(net_arch),
        "net_arch_label": net_arch_label,
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


def sample_lunarlander_hyperparameters(
    trial: TrialProtocol,
) -> dict[str, bool | float | int | str]:
    """Wylosuj hiperparametry PPO dla porownania LunarLandera w pre5.

    Przestrzen wyszukiwania jest wzorowana na praktyce z artykulu o
    optymalizacji PPO: wiekszosc osi jest zdyskretyzowana, aby szybciej
    porownac konfiguracje na ograniczonym budzecie CPU.

    Parameters
    ----------
    trial : TrialProtocol
        Obiekt triala Optuny.

    Returns
    -------
    dict[str, bool | float | int | str]
        Wylosowane hiperparametry PPO wraz z etykieta architektury.
    """
    net_arch_label = str(
        trial.suggest_categorical(
            "net_arch_label",
            list(LUNARLANDER_NET_ARCH_OPTIONS),
        )
    )
    n_steps = int(trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096]))
    valid_batch_sizes = [size for size in (64, 128, 256, 512) if n_steps % size == 0]

    return {
        "net_arch_label": net_arch_label,
        "learning_rate": float(
            trial.suggest_categorical(
                "learning_rate",
                [3e-5, 1e-4, 3e-4, 1e-3],
            )
        ),
        "batch_size": int(
            trial.suggest_categorical("batch_size", valid_batch_sizes)
        ),
        "gamma": float(
            trial.suggest_categorical("gamma", [0.95, 0.99, 0.995, 0.999])
        ),
        "n_steps": n_steps,
        "ent_coef": float(
            trial.suggest_categorical("ent_coef", [0.0, 1e-4, 1e-3, 1e-2])
        ),
        "gae_lambda": float(
            trial.suggest_categorical("gae_lambda", [0.95, 0.97, 0.99])
        ),
        "clip_range": float(
            trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3])
        ),
        "target_kl": float(trial.suggest_float("target_kl", 0.003, 0.03)),
        "n_epochs": int(trial.suggest_categorical("n_epochs", [5, 10, 15, 20])),
        "vf_coef": float(
            trial.suggest_categorical("vf_coef", [0.35, 0.5, 0.75, 1.0])
        ),
        "normalize_advantage": bool(
            trial.suggest_categorical("normalize_advantage", [True, False])
        ),
    }


def compute_lunarlander_objective_score(
    mean_reward: float,
    std_reward: float,
    stability_penalty: float,
) -> float:
    """Policz wynik optymalizacji laczacy jakosc i stabilnosc polityki.

    Parameters
    ----------
    mean_reward : float
        Srednia nagroda z ewaluacji.
    std_reward : float
        Odchylenie standardowe nagrody z ewaluacji.
    stability_penalty : float
        Wspolczynnik kary za niestabilnosc polityki.

    Returns
    -------
    float
        Wynik celu optymalizacji: srednia nagroda pomniejszona o kare wariancyjna.
    """
    return mean_reward - (stability_penalty * std_reward)


def append_lunarlander_result(
    csv_path: str,
    config: dict[str, object],
    metrics: Mapping[str, ResultValue],
) -> None:
    """Dopisz wynik pojedynczej proby Optuny do dedykowanego CSV.

    Parameters
    ----------
    csv_path : str
        Sciezka do pliku CSV wynikow dla LunarLandera.
    config : dict[str, object]
        Konfiguracja pojedynczej proby.
    metrics : Mapping[str, ResultValue]
        Metryki zwrocone przez runner pre5.
    """
    output_path = Path(csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "experiment_id",
        "env_id",
        "net_arch",
        "net_arch_label",
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


def _build_lunarlander_model(
    config: dict[str, object],
) -> tuple[PPO, gym.Env[Any, Any]]:
    """Utworz model PPO i srodowisko monitorowane dla LunarLandera.

    Parameters
    ----------
    config : dict[str, object]
        Konfiguracja pojedynczej proby.

    Returns
    -------
    tuple[PPO, gym.Env[Any, Any]]
        Zainicjalizowany model PPO oraz srodowisko Gymnasium.
    """
    env: gym.Env[Any, Any] = Monitor(gym.make(str(config["env_id"])))

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
        policy_kwargs={
            "net_arch": resolve_lunarlander_net_arch(
                str(cast(str, config["net_arch_label"]))
            )
        },
        tensorboard_log=TENSORBOARD_LOG_DIR,
        verbose=0,
        device="auto",
    )
    return model, env


def run_prunable_lunarlander_experiment(
    config: dict[str, object],
    trial: TrialProtocol,
    report_interval_timesteps: int,
    eval_episodes: int,
    stability_penalty: float,
) -> dict[str, float]:
    """Uruchom probe LunarLandera z raportowaniem posrednim do Optuny.

    Parameters
    ----------
    config : dict[str, object]
        Konfiguracja pojedynczej proby.
    trial : TrialProtocol
        Trial Optuny uzywany do raportowania wynikow posrednich.
    report_interval_timesteps : int
        Liczba krokow miedzy kolejnymi checkpointami pruningowymi.
    eval_episodes : int
        Liczba epizodow uzywana do kazdej ewaluacji posredniej.
    stability_penalty : float
        Kara za niestabilnosc polityki uzywana w funkcji celu.

    Returns
    -------
    dict[str, float]
        Koncowe metryki treningu, w tym liczba faktycznie wykonanych krokow.

    Raises
    ------
    LunarLanderTrialPruned
        Jesli trial zostal oznaczony do przerwania przez pruner.
    """
    Path(MODELS_DIR).mkdir(exist_ok=True)

    model, env = _build_lunarlander_model(config)
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
            objective_score = compute_lunarlander_objective_score(
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
            trial.report(float(objective_score), step=trained_timesteps)
            if trial.should_prune():
                raise LunarLanderTrialPruned(latest_metrics)

        model.save(f"{MODELS_DIR}/{config['experiment_id']}")
        return latest_metrics
    finally:
        env.close()  # type: ignore[no-untyped-call]


def build_budget_warning(trials: int, total_timesteps: int) -> str | None:
    """Zwroc ostrzezenie dla zbyt malego budzetu wyszukiwania pre5.

    Parameters
    ----------
    trials : int
        Liczba prob Optuny.
    total_timesteps : int
        Budzet krokow na jedna probe.

    Returns
    -------
    str | None
        Tekst ostrzezenia albo ``None``, jesli budzet nie wyglada podejrzanie.
    """
    if trials < 20 and total_timesteps < 300_000:
        return (
            "[WARN] Budzet wyglada jak smoke test, nie jak sensowne porownanie pre5 "
            "dla LunarLandera: optimizer potrzebuje kilkunastu-kilkudziesieciu prob, "
            "a 300k krokow pomaga odsiac konfiguracje uczace sie zbyt wolno."
        )
    return None


def run_lunarlander_study(
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
    """Uruchom optymalizacje bayesowska PPO dla LunarLandera.

    Parameters
    ----------
    trials : int
        Liczba prob Optuny.
    total_timesteps : int
        Liczba krokow treningowych na probe.
    results_csv : str
        Sciezka do CSV z wynikami wszystkich prob.
    study_name : str
        Nazwa studium Optuny.
    startup_trials : int
        Liczba prob rozruchowych prunera i samplera TPE.
    pruner_warmup_steps : int
        Liczba krokow treningowych ignorowanych przez pruner.
    report_interval_timesteps : int
        Liczba krokow pomiedzy raportami posrednimi do Optuny.
    eval_episodes : int
        Liczba epizodow w pojedynczej ewaluacji triala.
    stability_penalty : float
        Kara za niestabilnosc polityki w funkcji celu Optuny.
    optuna_storage : str
        URI storage Optuny do wznawiania i analizy studium.

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
        params = sample_lunarlander_hyperparameters(trial)
        config = build_lunarlander_config(
            trial_number=trial.number,
            net_arch_label=str(cast(str, params["net_arch_label"])),
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
            metrics = run_prunable_lunarlander_experiment(
                config=config,
                trial=trial,
                report_interval_timesteps=report_interval_timesteps,
                eval_episodes=eval_episodes,
                stability_penalty=stability_penalty,
            )
        except LunarLanderTrialPruned as exc:
            append_lunarlander_result(
                results_csv,
                config,
                {**exc.metrics, "status": "pruned"},
            )
            time.sleep(
                get_cooldown_seconds(
                    resolve_lunarlander_net_arch(str(config["net_arch_label"]))
                )
            )
            raise optuna.TrialPruned() from exc

        append_lunarlander_result(
            results_csv,
            config,
            {**metrics, "status": "completed"},
        )
        time.sleep(
            get_cooldown_seconds(
                resolve_lunarlander_net_arch(str(config["net_arch_label"]))
            )
        )
        return metrics["objective_score"]

    study.optimize(objective, n_trials=trials)

    return {
        "best_value": float(study.best_value),
        "best_params": dict(study.best_params),
    }


def main() -> None:
    """Uruchom CLI optymalizacji bayesowskiej dla LunarLandera."""
    parser = argparse.ArgumentParser(
        description=(
            "Optymalizacja bayesowska PPO dla LunarLander-v3 "
            "na architekturach [64, 64] i [128, 128]."
        )
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=DEFAULT_TRIALS,
        help="Liczba prob optymalizacji bayesowskiej.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=DEFAULT_TIMESTEPS,
        help="Liczba krokow treningowych na jedna probe.",
    )
    parser.add_argument(
        "--results-csv",
        default=DEFAULT_RESULTS_CSV,
        help="Sciezka do CSV, do ktorego beda dopisywane wyniki prob.",
    )
    parser.add_argument(
        "--study-name",
        default=DEFAULT_STUDY_NAME,
        help="Nazwa studium Optuny.",
    )
    parser.add_argument(
        "--startup-trials",
        type=int,
        default=DEFAULT_STARTUP_TRIALS,
        help="Liczba prob startup dla TPE i prunera MedianPruner.",
    )
    parser.add_argument(
        "--pruner-warmup-steps",
        type=int,
        default=DEFAULT_PRUNER_WARMUP_STEPS,
        help="Liczba krokow ignorowanych przez pruner przed pierwszym cieciem.",
    )
    parser.add_argument(
        "--report-interval-timesteps",
        type=int,
        default=DEFAULT_REPORT_INTERVAL_TIMESTEPS,
        help="Liczba krokow miedzy raportami posrednimi do Optuny.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=DEFAULT_EVAL_EPISODES,
        help="Liczba epizodow ewaluacyjnych po kazdym etapie triala.",
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
            "URI storage Optuny (np. sqlite:///data/lunarlander_optuna.db) "
            "uzywane do resume i analizy studium."
        ),
    )
    args = parser.parse_args()

    warning = build_budget_warning(args.trials, args.timesteps)
    if warning is not None:
        print(warning)

    summary = run_lunarlander_study(
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