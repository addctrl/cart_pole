"""Moduł treningowy — pętla PPO iterująca po konfiguracji CSV."""

import time
from pathlib import Path
from typing import cast

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from src.config import load_experiments, parse_net_arch, save_results

MODELS_DIR = "models"
TENSORBOARD_LOG_DIR = "./logs/tensorboard/"


def get_cooldown_seconds(net_arch: list[int]) -> int:
    """Zwróć czas cooldownu w sekundach zależny od rozmiaru sieci.

    Chroni chłodzenie pasywne MacBook Air M4 przed thermal throttlingiem
    (DKB-001). Dla dużych sieci stosowany jest dłuższy cooldown.

    Parameters
    ----------
    net_arch : list[int]
        Lista rozmiarów warstw ukrytych sieci neuronowej.

    Returns
    -------
    int
        Czas cooldownu: 120 sekund jeśli suma parametrów sieci
        przekracza 1000, 60 sekund w pozostałych przypadkach.
    """
    if sum(net_arch) > 1000:
        return 120
    return 60


def run_experiment(config: dict[str, object]) -> dict[str, float]:
    """Uruchom pojedynczy eksperyment treningowy PPO.

    Tworzy środowisko Gymnasium, trenuje agenta PPO z podaną
    konfiguracją, ewaluuje model i zapisuje wagi na dysk.

    Parameters
    ----------
    config : dict[str, object]
        Słownik konfiguracji eksperymentu wczytany z CSV. Wymagane
        klucze: ``env_id``, ``net_arch``, ``learning_rate``,
        ``batch_size``, ``gamma``, ``n_steps``, ``ent_coef``,
        ``total_timesteps``, ``experiment_id``.

    Returns
    -------
    dict[str, float]
        Słownik z metrykami: ``mean_reward``, ``std_reward``,
        ``training_time_s``.
    """
    Path(MODELS_DIR).mkdir(exist_ok=True)

    env = gym.make(str(config["env_id"]))

    net_arch: list[int] = parse_net_arch(str(config["net_arch"]))
    policy_kwargs: dict[str, list[int]] = {"net_arch": net_arch}

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=float(str(config["learning_rate"])),
        batch_size=int(str(config["batch_size"])),
        gamma=float(str(config["gamma"])),
        n_steps=int(str(config["n_steps"])),
        ent_coef=float(str(config["ent_coef"])),
        policy_kwargs=policy_kwargs,
        tensorboard_log=TENSORBOARD_LOG_DIR,
        verbose=0,
        device="auto",
    )

    start = time.time()
    model.learn(
        total_timesteps=int(str(config["total_timesteps"])),
        tb_log_name=str(config["experiment_id"]),
    )
    training_time_s = time.time() - start

    # evaluate_policy zwraca Union[tuple[float, float], tuple[list[float], list[int]]].
    # Przy return_episode_rewards=False (domyślnie) wynikiem są skalary float.
    mean_reward, std_reward = cast(
        tuple[float, float],
        evaluate_policy(
            model,
            env,
            n_eval_episodes=10,
            deterministic=True,
        ),
    )

    model.save(f"{MODELS_DIR}/{config['experiment_id']}")
    env.close()  # type: ignore[no-untyped-call]

    return {
        "mean_reward": float(mean_reward),
        "std_reward": float(std_reward),
        "training_time_s": round(training_time_s, 2),
    }


def run_all_experiments(csv_path: str) -> None:
    """Uruchom wszystkie zaplanowane eksperymenty z pliku CSV.

    Iteruje po liście eksperymentów, pomijając już ukończone
    (obsługa restartu po awarii). Po każdym treningu zapisuje
    wyniki i wymusza cooldown chroniący sprzęt (DKB-001).

    Parameters
    ----------
    csv_path : str
        Ścieżka do pliku CSV z macierzą eksperymentów.
    """
    experiments = load_experiments(csv_path)

    for config in experiments:
        if config["mean_reward"] is not None:
            continue

        print(f"[START] {config['experiment_id']} ...")
        metrics = run_experiment(config)
        save_results(csv_path, str(config["experiment_id"]), metrics)
        print(
            f"[DONE] {config['experiment_id']} "
            f"| mean_reward={metrics['mean_reward']:.1f}"
        )

        cooldown = get_cooldown_seconds(parse_net_arch(str(config["net_arch"])))
        time.sleep(cooldown)
