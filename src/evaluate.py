"""Moduł ewaluacyjny — ładowanie modelu i renderowanie epizodów."""

import argparse
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO


def evaluate_model(model_path: str, env_id: str, episodes: int) -> None:
    """Załaduj model PPO i uruchom ewaluację z renderowaniem.

    Parameters
    ----------
    model_path : str
        Ścieżka do pliku modelu zapisanego przez ``stable-baselines3``.
    env_id : str
        Identyfikator środowiska Gymnasium, np. ``CartPole-v1``.
    episodes : int
        Liczba epizodów do uruchomienia w trybie renderowania.

    Raises
    ------
    FileNotFoundError
        Jeśli plik modelu nie istnieje.
    """
    model_file = Path(model_path)
    if not model_file.is_file():
        raise FileNotFoundError(f"Plik modelu nie istnieje: {model_path}")

    model = PPO.load(model_path)
    env = gym.make(env_id, render_mode="human")

    try:
        for _ in range(episodes):
            observation, _ = env.reset()
            terminated = False
            truncated = False

            while not (terminated or truncated):
                action, _ = model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, _ = env.step(action)
    finally:
        env.close()  # type: ignore[no-untyped-call]


def main() -> None:
    """Uruchom CLI skryptu ewaluacyjnego."""
    parser = argparse.ArgumentParser(
        description="Ładowanie modelu PPO i renderowanie ewaluacji w Gymnasium."
    )
    parser.add_argument("--model-path", required=True, help="Ścieżka do pliku modelu")
    parser.add_argument("--env-id", required=True, help="Identyfikator środowiska")
    parser.add_argument("--episodes", required=True, type=int, help="Liczba epizodów")
    args = parser.parse_args()

    evaluate_model(args.model_path, args.env_id, args.episodes)


if __name__ == "__main__":  # pragma: no cover
    main()
