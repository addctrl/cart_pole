"""Produkcyjny trening Humanoida z auto-resume i spójnym checkpointem."""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

BEST_PARAMS: dict[str, float | int | bool] = {
    "n_steps": 1024,
    "learning_rate": 2.446001779697762e-05,
    "batch_size": 512,
    "gamma": 0.99,
    "ent_coef": 0.00014344593563113446,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "target_kl": 0.02683369034384713,
    "n_epochs": 15,
    "vf_coef": 0.75,
    "normalize_advantage": False,
}

ENV_ID = "Humanoid-v5"
NET_ARCH = [512, 512]
TOTAL_TIMESTEPS = 30_000_000
SAVE_FREQ = 1_000_000

MODELS_DIR = Path("models/humanoid_prod")
TENSORBOARD_LOG_DIR = "logs/tensorboard/"

MODEL_LATEST_PATH = MODELS_DIR / "latest_model"
VECNORM_LATEST_PATH = MODELS_DIR / "latest_vecnormalize.pkl"


class SyncedCheckpointCallback(BaseCallback):
    """Zapisuj model i statystyki normalizacji w tym samym kroku.

    Parameters
    ----------
    save_freq : int
        Interwał zapisu checkpointu w krokach środowiska.
    vec_env : VecNormalize
        Środowisko z normalizacją, które trzeba zapisać razem z modelem.
    verbose : int, default=0
        Poziom logowania callbacku.
    """

    def __init__(self, save_freq: int, vec_env: VecNormalize, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.save_freq = save_freq
        self.vec_env = vec_env

    def _on_step(self) -> bool:
        """Obsłuż pojedynczy krok treningu i opcjonalny checkpoint.

        Returns
        -------
        bool
            ``True`` oznacza kontynuację treningu.
        """
        if self.n_calls % self.save_freq == 0:
            self.model.save(str(MODEL_LATEST_PATH))
            self.vec_env.save(str(VECNORM_LATEST_PATH))
            if self.verbose > 0:
                print(
                    f"[{self.num_timesteps}] Auto-zapis: model + VecNormalize zapisane."
                )
        return True


def _build_base_env() -> DummyVecEnv:
    """Zbuduj bazowe środowisko wektorowe dla Humanoida.

    Returns
    -------
    DummyVecEnv
        Jednośrodowiskowy wrapper wektorowy z monitorem.
    """
    return DummyVecEnv([lambda: Monitor(gym.make(ENV_ID))])


def _build_new_model() -> tuple[PPO, VecNormalize]:
    """Utwórz nowy model PPO i świeżą normalizację środowiska.

    Returns
    -------
    tuple[PPO, VecNormalize]
        Model PPO oraz środowisko ``VecNormalize`` gotowe do treningu.
    """
    env = VecNormalize(
        _build_base_env(),
        norm_obs=True,
        norm_reward=True,
        gamma=float(BEST_PARAMS["gamma"]),
    )

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=float(BEST_PARAMS["learning_rate"]),
        batch_size=int(BEST_PARAMS["batch_size"]),
        gamma=float(BEST_PARAMS["gamma"]),
        n_steps=int(BEST_PARAMS["n_steps"]),
        ent_coef=float(BEST_PARAMS["ent_coef"]),
        gae_lambda=float(BEST_PARAMS["gae_lambda"]),
        clip_range=float(BEST_PARAMS["clip_range"]),
        target_kl=float(BEST_PARAMS["target_kl"]),
        n_epochs=int(BEST_PARAMS["n_epochs"]),
        vf_coef=float(BEST_PARAMS["vf_coef"]),
        normalize_advantage=bool(BEST_PARAMS["normalize_advantage"]),
        policy_kwargs={"net_arch": NET_ARCH},
        tensorboard_log=TENSORBOARD_LOG_DIR,
        verbose=1,
        device="auto",
    )
    return model, env


def _load_resumed_model() -> tuple[PPO, VecNormalize]:
    """Wczytaj zapisany stan modelu i normalizacji.

    Returns
    -------
    tuple[PPO, VecNormalize]
        Model PPO i środowisko gotowe do wznowienia treningu.
    """
    env = VecNormalize.load(str(VECNORM_LATEST_PATH), _build_base_env())
    env.training = True
    env.norm_reward = True
    model = PPO.load(str(MODEL_LATEST_PATH), env=env)
    return model, env


def _has_resume_checkpoint() -> bool:
    """Sprawdź, czy dostępny jest kompletny checkpoint auto-resume.

    Returns
    -------
    bool
        ``True``, gdy istnieje model i odpowiadający mu plik VecNormalize.
    """
    return MODEL_LATEST_PATH.with_suffix(".zip").exists() and VECNORM_LATEST_PATH.exists()


def main() -> None:
    """Uruchom produkcyjny trening Humanoida na 30 milionów kroków."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if _has_resume_checkpoint():
        print("Znaleziono checkpoint. Wznawiam trening.")
        model, env = _load_resumed_model()
    else:
        print("Brak checkpointu. Start zimny.")
        model, env = _build_new_model()

    already_trained = int(model.num_timesteps)
    if already_trained >= TOTAL_TIMESTEPS:
        print(
            "Budżet 30M kroków jest już osiągnięty "
            f"({already_trained} kroków). Bez dodatkowego treningu."
        )
    else:
        remaining = TOTAL_TIMESTEPS - already_trained
        print(
            f"Start uczenia. Dotychczas: {already_trained} kroków, "
            f"pozostało: {remaining}."
        )
        callback = SyncedCheckpointCallback(
            save_freq=SAVE_FREQ,
            vec_env=env,
            verbose=1,
        )
        model.learn(
            total_timesteps=remaining,
            callback=callback,
            tb_log_name="humanoid_512x512_prod_30m",
            reset_num_timesteps=False,
        )

    model.save(str(MODEL_LATEST_PATH))
    env.save(str(VECNORM_LATEST_PATH))
    print("Zapis końcowy wykonany: model + VecNormalize.")


if __name__ == "__main__":
    main()
