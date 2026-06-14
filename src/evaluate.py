"""Moduł ewaluacyjny — ładowanie modelu i renderowanie epizodów."""

import argparse
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def _build_evaluation_env(model_path: Path, env_id: str) -> tuple[object, bool]:
    """Zbuduj środowisko ewaluacyjne i opcjonalnie dołącz VecNormalize.

    Parameters
    ----------
    model_path : Path
        Ścieżka do pliku modelu.
    env_id : str
        Identyfikator środowiska Gymnasium.

    Returns
    -------
    tuple[object, bool]
        Środowisko ewaluacyjne oraz flaga wskazująca, czy jest to VecEnv.
    """
    vecnormalize_path = model_path.with_name(f"{model_path.stem}_vecnormalize.pkl")
    if vecnormalize_path.exists():
        vec_env = DummyVecEnv([lambda: gym.make(env_id, render_mode="human")])
        vec_env = VecNormalize.load(str(vecnormalize_path), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        return vec_env, True

    return gym.make(env_id, render_mode="human"), False


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
    env, is_vec_env = _build_evaluation_env(model_file, env_id)

    try:
        for _ in range(episodes):
            if is_vec_env:
                observation = env.reset()
                terminated = False

                while not terminated:
                    action, _ = model.predict(observation, deterministic=True)
                    observation, _, done, _ = env.step(action)
                    terminated = bool(done[0])
            else:
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
