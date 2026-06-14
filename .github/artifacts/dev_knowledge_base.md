# Baza Wiedzy Deweloperskiej

> Rejestr znanych problemów, rozwiązań i specyfiki technicznej projektu.
> **Przed debugowaniem — sprawdź ten plik.** Po rozwiązaniu nowego problemu — dopisz wpis.

---

## DKB-001: Specyfika MacBook Air M4 — chłodzenie pasywne

| Pole | Wartość |
|---|---|
| **Data** | 2026-05-09 |
| **Kategoria** | Sprzęt |
| **Priorytet** | Krytyczny |

### Problem

MacBook Air M4 posiada chłodzenie pasywne (brak wentylatora). Przy ciągłym obciążeniu CPU (trening RL) temperatura rośnie do progu thermal throttlingu (~100°C), co powoduje:
1. Redukcję taktowania CPU (spadek wydajności o 30-50%).
2. Niespójne pomiary czasu treningu między eksperymentami.
3. Potencjalne skrócenie żywotności komponentów.

### Rozwiązanie

Wymuszony cooldown via `time.sleep()` między eksperymentami:
- **60 sekund** — domyślny cooldown.
- **120 sekund** — dla architektury `[1024, 1024, 1024]`.

Implementacja w `src/training.py`:

```python
import time

def get_cooldown_seconds(net_arch: list[int]) -> int:
    """Zwraca czas cooldownu w sekundach."""
    total_params = sum(net_arch)
    if total_params > 1000:
        return 120
    return 60

# W pętli treningowej:
time.sleep(get_cooldown_seconds(config["net_arch"]))
```

### Dodatkowe uwagi

- Trening na zasilaczu (nie na baterii) zapewnia stabilniejsze taktowanie.
- Monitorowanie temperatury: `sudo powermetrics --samplers smc -i 1000 -n 1` (wymaga uprawnień root).

---

## DKB-002: Renderowanie Gymnasium na macOS (Apple Silicon)

| Pole | Wartość |
|---|---|
| **Data** | 2026-05-09 |
| **Kategoria** | Gymnasium / Renderowanie |
| **Priorytet** | Wysoki |

### Problem

`gymnasium` z `render_mode="human"` wymaga backendu graficznego. Na macOS z Apple Silicon mogą wystąpić:
1. **Brak okna renderowania** — pygame nie inicjalizuje się poprawnie.
2. **Crash z `NSInternalInconsistencyException`** — konflikt z wątkami macOS.
3. **Czarny ekran** — brak poprawnego backendu SDL.

### Rozwiązanie

1. **Zainstaluj `pygame`** w `requirements.txt`:
   ```
   pygame>=2.5.0
   ```

2. **Nie uruchamiaj renderowania w wątkach pobocznych.** Rendering musi działać w głównym wątku.

3. **Jeśli czarny ekran na M4:**
   ```bash
   export SDL_VIDEODRIVER=cocoa
   ```
   Dodaj do instrukcji uruchomienia lub skryptu.

4. **W testach nigdy nie renderuj.** Mockuj `gym.make()` z `render_mode=None`.

### Weryfikacja

```bash
python -c "import pygame; pygame.init(); print(pygame.display.get_driver())"
```

Oczekiwany output: `cocoa`.

---

## DKB-003: Relacja `n_steps` vs `batch_size` w PPO

| Pole | Wartość |
|---|---|
| **Data** | 2026-05-09 |
| **Kategoria** | Stable-Baselines3 / Hiperparametry |
| **Priorytet** | Wysoki |

### Problem

PPO w `stable-baselines3` wymaga, aby `n_steps * n_envs` było **podzielne przez `batch_size`**. Naruszenie tego warunku powoduje:

```
ValueError: `n_steps * n_envs` must be a multiple of `batch_size`
```

W projekcie `n_envs=1` (jedno środowisko), więc warunek upraszcza się do:

```
n_steps % batch_size == 0
```

### Rozwiązanie

1. **Walidacja w `src/config.py`** — przy wczytywaniu CSV sprawdź warunek `n_steps % batch_size == 0`. Jeśli niespełniony — rzuć `ValueError` z czytelnym komunikatem.

2. **Bezpieczne kombinacje (dla `n_envs=1`):**

   | `n_steps` | Dopuszczalne `batch_size` |
   |---|---|
   | 2048 | 32, 64, 128, 256, 512, 1024, 2048 |
   | 1024 | 32, 64, 128, 256, 512, 1024 |
   | 512 | 32, 64, 128, 256, 512 |

3. **W pliku `data/experiments.csv`** — dobieraj wartości tak, aby warunek był zawsze spełniony.

### Uwaga dot. pamięci RAM

`n_steps` określa rozmiar bufora rollout. Dla dużych `n_steps` (np. 4096) i dużej sieci (`[1024, 1024, 1024]`):
- Bufor zużywa ~`n_steps * obs_dim * sizeof(float32)` pamięci.
- Dla CartPole (`obs_dim=4`): 4096 * 4 * 4 = ~64 KB — pomijalnie małe.
- Dla LunarLander (`obs_dim=8`): nadal pomijalnie małe.
- **Pamięć nie jest bottleneckiem** dla tych środowisk. Bottleneck to CPU (thermal throttling).

---

## DKB-004: `device="auto"` na macOS (Apple Silicon)

| Pole | Wartość |
|---|---|
| **Data** | 2026-05-09 |
| **Kategoria** | Stable-Baselines3 / Konfiguracja |
| **Priorytet** | Średni |

### Problem

`stable-baselines3` nie wspiera MPS (Metal Performance Shaders) jako backendu PyTorch. Ustawienie `device="mps"` lub `device="cuda"` spowoduje błąd.

### Rozwiązanie

Zawsze używaj `device="auto"`. Na macOS sb3 automatycznie wybierze `cpu`.

```python
model = PPO("MlpPolicy", env, device="auto")
```

**Nie próbuj wymuszać GPU.** PyTorch na M4 wspiera MPS, ale sb3 nie. To ograniczenie biblioteki, nie sprzętu.

---

## DKB-005: TensorBoard — nazewnictwo logów

| Pole | Wartość |
|---|---|
| **Data** | 2026-05-09 |
| **Kategoria** | TensorBoard / Logowanie |
| **Priorytet** | Średni |

### Problem

Domyślnie sb3 tworzy logi TensorBoard z losowymi suffixami. Utrudnia to identyfikację eksperymentów.

### Rozwiązanie

Użyj parametru `tb_log_name` w `model.learn()`:

```python
model = PPO(
    "MlpPolicy",
    env,
    tensorboard_log="./logs/tensorboard/",
)
model.learn(
    total_timesteps=100_000,
    tb_log_name=f"exp_{experiment_id}",
)
```

Nazewnictwo: `exp_<experiment_id>` — zgodne z `experiment_id` z CSV.

### Przeglądanie logów

```bash
tensorboard --logdir=./logs/tensorboard/ --port=6006
```

Otworzy dashboard pod `http://localhost:6006`.

---

## DKB-006: Gymnasium `v1` vs `v0` — API reset()

| Pole | Wartość |
|---|---|
| **Data** | 2026-05-09 |
| **Kategoria** | Gymnasium / API |
| **Priorytet** | Średni |

### Problem

API Gymnasium zmieniło się między v0 a v1. `reset()` zwraca teraz tuple:

```python
# Stare API (v0):
obs = env.reset()

# Nowe API (v1):
obs, info = env.reset()
```

Analogicznie `step()`:

```python
# Stare API (v0):
obs, reward, done, info = env.step(action)

# Nowe API (v1):
obs, reward, terminated, truncated, info = env.step(action)
```

### Rozwiązanie

Używaj wyłącznie nowego API (v1). Środowiska w projekcie: `CartPole-v1`, `LunarLander-v2` — oba używają nowego API.

**W `evaluate.py`:**

```python
obs, info = env.reset()
for _ in range(max_steps):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
```

---

## DKB-007: LunarLander wymaga Box2D i wersji `v3`

| Pole | Wartość |
|---|---|
| **Data** | 2026-06-13 |
| **Kategoria** | Gymnasium / LunarLander |
| **Priorytet** | Wysoki |

### Problem

`LunarLander-v2` został zastąpiony przez `LunarLander-v3` w aktualnym `gymnasium`. Dodatkowo środowisko nie jest dostępne bez zależności Box2D i narzędzia `swig`.

### Rozwiązanie

1. Instaluj zależności z `requirements.txt`, które zawierają `gymnasium[box2d]` i `swig`.
2. W CSV i komendach CLI używaj `LunarLander-v3`.
3. Przy istniejącym venv wykonaj ponowny `pip install -r requirements.txt`.

---

## DKB-008: Humanoid wymaga MuJoCo i izolacji zależności

| Pole | Wartość |
|---|---|
| **Data** | 2026-06-13 |
| **Kategoria** | Gymnasium / MuJoCo / Optuna |
| **Priorytet** | Wysoki |

### Problem

Środowisko `Humanoid-v5` nie jest częścią bazowego stosu CartPole/LunarLander. Wymaga dodatkowych zależności MuJoCo, a optymalizacja bayesowska wymaga biblioteki `optuna`. Dołączenie tych pakietów do ścieżki bazowej zwiększa ryzyko instalacyjnych regresji i niepotrzebnie poszerza zależności podstawowego projektu.

### Rozwiązanie

1. Instaluj wariant Humanoid wyłącznie przez:

    ```bash
    pip install -r requirements-humanoid.txt
    ```

2. Uruchamiaj optymalizację dedykowanym modułem:

    ```bash
    python -m src.humanoid_bayes --trials 10 --timesteps 300000
    ```

3. Nie dopinaj semantyki Optuny do `data/experiments.csv`; wyniki zapisuj do osobnego CSV.

Próba uruchomienia `LunarLander-v2` kończy się deprecjacją środowiska, a `LunarLander-v3` bez dodatkowych zależności kończy się błędem:

```text
DependencyNotInstalled: Box2D is not installed
```

### Rozwiązanie

1. Używaj `LunarLander-v3` zamiast `LunarLander-v2`.
2. W środowisku Python zainstaluj zależności Box2D:

```bash
pip install swig
pip install "gymnasium[box2d]"
```

3. W repozytorium utrzymuj `gymnasium[box2d]==1.1.1` oraz `swig` w `requirements.txt`.

### Konsekwencje

- Kod treningowy nie wymaga zmian logicznych, bo `env_id` jest parametrem wejściowym.
- Zmienia się tylko konfiguracja eksperymentów i zależności środowiskowe.
