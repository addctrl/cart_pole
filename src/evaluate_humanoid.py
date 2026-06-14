import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# 1. Ścieżki - muszą być identyczne jak w treningu
model_path = "models/humanoid_prod/latest_model"
stats_path = "models/humanoid_prod/latest_vecnormalize.pkl"

# 2. Tworzymy środowisko - IDENTYCZNIE jak w _build_base_env()
env = DummyVecEnv([lambda: Monitor(gym.make("Humanoid-v5", render_mode="human"))])

# 3. Wczytujemy normalizację (KLUCZOWE!)
# VecNormalize.load musi opakować to samo bazowe środowisko
env = VecNormalize.load(stats_path, env)

# 4. Wyłączamy treningowe modyfikacje normalizacji
env.training = False 
env.norm_reward = False

# 5. Wczytujemy model
model = PPO.load(model_path, env=env)

# 6. Pętla ewaluacji
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    # W VecEnv reset jest automatyczny, nie musisz ręcznie resetować
env.close()