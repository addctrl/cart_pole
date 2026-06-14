"""Ewaluacja produkcyjnego modelu Humanoida z auto-sciezkami artefaktow."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.evaluate import evaluate_model

DEFAULT_MODEL_PATH = Path("models/humanoid_prod/latest_model.zip")
DEFAULT_ENV_ID = "Humanoid-v5"
DEFAULT_EPISODES = 5


def main() -> None:
    """Uruchom ewaluacje produkcyjnego modelu Humanoida.

    Parameters
    ----------
    None
        Funkcja korzysta z argumentow CLI.

    Returns
    -------
    None
        Funkcja nie zwraca wartosci.

    Raises
    ------
    FileNotFoundError
        Gdy domyslny lub podany model nie istnieje.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Ewaluuje produkcyjny model Humanoida z katalogu "
            "models/humanoid_prod."
        )
    )
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_PATH),
        help=(
            "Sciezka do modelu PPO. Domyslnie: "
            "models/humanoid_prod/latest_model.zip"
        ),
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=DEFAULT_EPISODES,
        help=f"Liczba epizodow ewaluacji (domyslnie: {DEFAULT_EPISODES}).",
    )
    args = parser.parse_args()

    evaluate_model(
        model_path=args.model_path,
        env_id=DEFAULT_ENV_ID,
        episodes=args.episodes,
    )


if __name__ == "__main__":
    main()
