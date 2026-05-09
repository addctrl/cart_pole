# Architektura i Backlog Zadań

> Plik definiuje architekturę systemu oraz pełny backlog realizacji.
> Agenci aktualizują status zadań po każdym zamkniętym epiku.

---

## 1. Architektura systemu

### 1.1 Diagram przepływu danych

```
experiments.csv ──► config.py ──► training.py ──► models/*.zip
                                      │               │
                                      ▼               ▼
                                 tensorboard/    experiments.csv
                                   (logi)        (+ wyniki)
                                      
models/*.zip ──► evaluate.py ──► render_mode="human"
```

### 1.2 Moduły

#### `src/config.py` — Moduł konfiguracji

**Odpowiedzialność:** Odczyt i zapis danych z/do pliku CSV.

| Funkcja | Opis |
|---|---|
| `load_experiments(path: str) -> list[dict]` | Wczytuje macierz eksperymentów z CSV. Zwraca listę słowników. |
| `save_results(path: str, experiment_id: str, metrics: dict) -> None` | Dopisuje metryki wynikowe do wiersza w CSV. |
| `parse_net_arch(raw: str) -> list[int]` | Parsuje string `"[64, 64]"` na listę intów. |

#### `src/training.py` — Moduł treningowy

**Odpowiedzialność:** Pętla treningowa iterująca po konfiguracji CSV.

| Funkcja / Klasa | Opis |
|---|---|
| `run_experiment(config: dict) -> dict` | Uruchamia pojedynczy trening PPO. Zwraca metryki. |
| `run_all_experiments(csv_path: str) -> None` | Iteruje po CSV, uruchamia `run_experiment()`, zapisuje wyniki, implementuje cooldown. |
| `get_cooldown_seconds(net_arch: list[int]) -> int` | Zwraca czas cooldownu w zależności od rozmiaru sieci. |

#### `src/evaluate.py` — Moduł ewaluacyjny

**Odpowiedzialność:** Ładowanie wytrenowanego modelu i wizualizacja gry.

| Funkcja | Opis |
|---|---|
| `evaluate_model(model_path: str, env_id: str, episodes: int) -> None` | Ładuje model, uruchamia `n` epizodów z `render_mode="human"`. |
| `main() -> None` | Entry point ze wsparciem argumentów CLI (`argparse`). |

---

## 2. Backlog realizacji

### Epik 0: Setup środowiska

- [x] Stworzenie repozytorium
- [x] Przygotowanie planu realizacji zadania
- [x] Przygotowanie folderu `.github/` oraz struktury pod pracę z agentami AI
- [x] Zaprojektowanie bazy promptów agentów
- [x] Przygotowanie wsadu do plików kontekstowych (PRD, ADR, backlog, baza wiedzy, AGENTS.md)

### Epik 1: Konfiguracja środowiska deweloperskiego

- [x] **T-1.1:** Utworzenie `requirements.txt` z pinowanymi wersjami zależności
  - Kryteria akceptacji: `pip install -r requirements.txt` w czystym venv kończy się sukcesem.
  - Zależności: `gymnasium`, `stable-baselines3`, `tensorboard`, `pygame`, `pytest`, `pytest-cov`, `ruff`, `mypy`, `pdoc`.
- [x] **T-1.2:** Utworzenie `pyproject.toml` z konfiguracją Ruff, Mypy, Pytest
  - Kryteria akceptacji: `ruff check .` i `mypy src/` działają bez błędów konfiguracyjnych.
- [x] **T-1.3:** Utworzenie struktury katalogów (`src/`, `tests/`, `data/`, `logs/`, `models/`, `docs/`)
  - Kryteria akceptacji: Katalogi istnieją, pliki `__init__.py` w `src/` i `tests/`.
- [x] **T-1.4:** Utworzenie pliku `data/experiments.csv` z macierzą eksperymentów
  - Kryteria akceptacji: CSV zawiera kolumny: `experiment_id`, `env_id`, `net_arch`, `learning_rate`, `batch_size`, `gamma`, `n_steps`, `ent_coef`, `total_timesteps`. Kolumny wynikowe (puste): `mean_reward`, `std_reward`, `training_time_s`.
- [x] **T-1.5:** Konfiguracja `.github/workflows/gatekeeper.yml`
  - Kryteria akceptacji: Pipeline uruchamia się na PR do `main`. Etapy: Ruff, Mypy, Pytest.
- [x] **T-1.6:** Skrypt `scripts/run_checks.sh` — lokalny odpowiednik pipeline'u CI/CD
  - Kryteria akceptacji: Plik wykonywalny, exit 0 przy sukcesie, exit 1 przy błędzie.

### Epik 2: Implementacja modułu konfiguracji

- [x] **T-2.1:** Implementacja `src/config.py`
  - `load_experiments()` — wczytanie CSV do `list[dict]`.
  - `save_results()` — dopisanie metryk do CSV.
  - `parse_net_arch()` — parsowanie stringa architektury.
  - Kryteria akceptacji: Pełna dokumentacja NumPy, type hints, brak błędów Ruff/Mypy.
- [x] **T-2.2:** Testy jednostkowe `tests/test_config.py`
  - Scenariusze: poprawny odczyt CSV, błędny plik, pusty plik, parsowanie architektury, zapis wyników.
  - Kryteria akceptacji: 100% coverage modułu `config.py`.

### Epik 3: Implementacja pętli treningowej

- [x] **T-3.1:** Implementacja `src/training.py`
  - `run_experiment()` — pojedynczy trening PPO z konfiguracją z CSV.
  - `run_all_experiments()` — pętla po CSV z cooldownem i zapisem wyników.
  - `get_cooldown_seconds()` — logika cooldownu (60s domyślnie, 120s dla dużych sieci).
  - Kryteria akceptacji: Pełna dokumentacja NumPy, type hints, logowanie TensorBoard, zapis modelu.
- [x] **T-3.2:** Testy jednostkowe `tests/test_training.py`
  - Scenariusze: mockowany trening, weryfikacja cooldownu, weryfikacja zapisu modelu, weryfikacja logów TensorBoard.
  - Kryteria akceptacji: 100% coverage modułu `training.py`. Brak fizycznego treningu w testach.
- [x] **T-3.3:** Integracja z TensorBoard
  - Kryteria akceptacji: `tensorboard --logdir=./logs/tensorboard/` wyświetla wykresy `ep_rew_mean` i `loss` per eksperyment.

### Epik 4: Skrypt ewaluacyjny

- [ ] **T-4.1:** Implementacja `src/evaluate.py`
  - `evaluate_model()` — ładowanie modelu, uruchomienie epizodów z renderowaniem.
  - `main()` — CLI via `argparse` (argumenty: `--model-path`, `--env-id`, `--episodes`).
  - Kryteria akceptacji: `python -m src.evaluate --model-path models/exp_001.zip --env-id CartPole-v1 --episodes 5` renderuje grę.
- [ ] **T-4.2:** Testy jednostkowe `tests/test_evaluate.py`
  - Scenariusze: mockowane ładowanie modelu, mockowane renderowanie, nieprawidłowa ścieżka modelu.
  - Kryteria akceptacji: 100% coverage modułu `evaluate.py`. Brak renderowania w testach.

### Epik 5: Uruchomienie eksperymentów

- [ ] **T-5.1:** Uruchomienie pełnej pętli treningowej na CartPole-v1
  - Kryteria akceptacji: Wszystkie wiersze w CSV mają wypełnione kolumny wynikowe. Logi TensorBoard istnieją. Wagi modeli zapisane.
- [ ] **T-5.2:** Analiza wyników w TensorBoard
  - Kryteria akceptacji: Wykresy porównawcze `ep_rew_mean` dla 3 architektur sieci. Screenshoty w dokumentacji.
- [ ] **T-5.3:** Uruchomienie ewaluacji najlepszego modelu na LunarLander-v2
  - Kryteria akceptacji: Demo live działa z `render_mode="human"`.

### Epik 6: Finalizacja i dokumentacja

- [ ] **T-6.1:** Aktualizacja `README.md` — finalna instrukcja uruchomienia
- [ ] **T-6.2:** Generowanie dokumentacji HTML via `pdoc`
- [ ] **T-6.3:** Eksport historii komunikacji z AI do repozytorium
- [ ] **T-6.4:** Przygotowanie prezentacji zaliczeniowej

---

## 3. Zależności między epikami

```
Epik 0 ──► Epik 1 ──► Epik 2 ──► Epik 3 ──► Epik 4 ──► Epik 5 ──► Epik 6
                                      │
                                      └──► T-3.3 (TensorBoard)
```

- Epik 2 i 3 można częściowo równoleglić (config.py jest niezależny od training.py).
- Epik 4 wymaga ukończenia Epiku 3 (potrzebne są zapisane modele do ewaluacji).
- Epik 5 wymaga ukończenia Epików 2, 3 i 4.

---

## 4. Sprint: Epik 2 + 3 — Szczegółowy plan realizacji

> **Data:** 2026-05-09 | **Decydent:** Architekt | **Wykonawca:** Developer (1-developer.agent.md)

### Kontekst decyzji

Epiki 2 i 3 realizowane w jednej iteracji ze względu na silną kohezję modułów — `training.py` jest bezpośrednim konsumentem `config.py`. Realizacja sekwencyjna w ramach jednego branch'a eliminuje overhead zarządzania zależnościami między branch'ami.

### Rama iteracji

| Element | Wartość |
|---|---|
| **Branch** | `feat/config-and-training` |
| **Kolejność plików** | `src/config.py` → `tests/test_config.py` → `src/training.py` → `tests/test_training.py` |
| **Warunek zamknięcia** | `scripts/run_checks.sh` kończy się exit 0 |

---

### T-2.1: `src/config.py`

#### Cel

Moduł odpowiada za **wyłącznie** operacje I/O na pliku CSV. Żadnej logiki domenowej.

#### Specyfikacja funkcji

**`parse_net_arch(raw: str) -> list[int]`**

- Parsuje string `"[64, 64]"` na `[64, 64]` za pomocą `ast.literal_eval()`.
- Weryfikuje, że wynik jest `list[int]` (każdy element `> 0`).
- `ValueError` jeśli format niepoprawny lub elementy nie są dodatnimi intami.

**`load_experiments(path: str) -> list[dict]`**

- Wczytuje CSV przez `csv.DictReader`.
- `FileNotFoundError` jeśli plik nie istnieje.
- Puste CSV (tylko nagłówek) → zwraca `[]`.
- Dla każdego wiersza konwertuje typy:
  - `learning_rate`, `gamma`, `ent_coef` → `float`
  - `batch_size`, `n_steps`, `total_timesteps` → `int`
  - `net_arch` → zostawia jako `str` (parsuje `parse_net_arch` osobno)
  - `mean_reward`, `std_reward`, `training_time_s` → `float` jeśli niepuste, `None` jeśli puste
- Walidacja po konwersji: `n_steps % batch_size != 0` → `ValueError` z komunikatem wskazującym `experiment_id` i wartości (DKB-003).

**`save_results(path: str, experiment_id: str, metrics: dict) -> None`**

- Wczytuje cały CSV do pamięci.
- Znajduje wiersz po `experiment_id`.
- `ValueError` jeśli `experiment_id` nie istnieje w pliku.
- Aktualizuje pola: `mean_reward`, `std_reward`, `training_time_s` z wartości w `metrics`.
- Zapisuje z powrotem cały plik (nadpisuje) — atomowa operacja dla małych plików.
- Używa `csv.DictWriter` z `extrasaction="ignore"`.

#### Wymagania kodu

- Import wyłącznie z biblioteki standardowej: `ast`, `csv`, `pathlib.Path`.
- Wszystkie typy danych wyrażone przez `type hints`.
- Docstringi NumPy w języku polskim — sekcje: opis, `Parameters`, `Returns`, `Raises`.
- Zero wartości domyślnych dla parametrów (wszystkie obligatoryjne).

---

### T-2.2: `tests/test_config.py`

#### Wymagane scenariusze

| ID testu | Opis | Mechanizm |
|---|---|---|
| `test_load_experiments_valid` | Poprawny CSV → lista dicts z poprawnymi typami | `tmp_path` fixture |
| `test_load_experiments_file_not_found` | Nieistniejący plik → `FileNotFoundError` | `pytest.raises` |
| `test_load_experiments_empty_csv` | CSV z samym nagłówkiem → `[]` | `tmp_path` |
| `test_load_experiments_invalid_batch_size` | `n_steps=512, batch_size=300` → `ValueError` | `tmp_path`, `pytest.raises` |
| `test_parse_net_arch_valid_two_layers` | `"[64, 64]"` → `[64, 64]` | bezpośrednie wywołanie |
| `test_parse_net_arch_valid_three_layers` | `"[1024, 1024, 1024]"` → `[1024, 1024, 1024]` | bezpośrednie wywołanie |
| `test_parse_net_arch_invalid_string` | `"invalid"` → `ValueError` | `pytest.raises` |
| `test_parse_net_arch_invalid_elements` | `"[64, -1]"` → `ValueError` | `pytest.raises` |
| `test_save_results_valid` | Aktualizuje właściwy wiersz, pozostałe nienaruszone | `tmp_path` |
| `test_save_results_experiment_not_found` | Nieistniejące `experiment_id` → `ValueError` | `tmp_path`, `pytest.raises` |

#### Wymagania kodu testowego

- Fixture `tmp_path` (wbudowana w pytest) — zero plików stałych w repozytorium.
- `conftest.py` jeśli potrzebne współdzielone fixtures (decyzja dewelopera).
- Type hints, docstringi NumPy, Ruff-clean — identyczny standard co kod produkcyjny.

---

### T-3.1: `src/training.py`

#### Cel

Moduł implementuje pętlę treningową. Jedyna odpowiedzialność: przyjmuje config, produkuje model i metryki.

#### Specyfikacja funkcji

**`get_cooldown_seconds(net_arch: list[int]) -> int`**

- `sum(net_arch) > 1000` → `return 120`
- w przeciwnym razie → `return 60`
- Prosta funkcja, zero stanu.

**`run_experiment(config: dict) -> dict`**

Sekwencja działań:
1. `env = gymnasium.make(config["env_id"])` — brak `render_mode` (trening, nie demo).
2. `net_arch = parse_net_arch(config["net_arch"])` — konwersja stringa.
3. `policy_kwargs = {"net_arch": net_arch}`.
4. Inicjalizacja `PPO("MlpPolicy", env, learning_rate=..., batch_size=..., gamma=..., n_steps=..., ent_coef=..., policy_kwargs=policy_kwargs, tensorboard_log="./logs/tensorboard/", device="auto")`.
5. Pomiar czasu: `start = time.time()`.
6. `model.learn(total_timesteps=config["total_timesteps"], tb_log_name=config["experiment_id"])`.
7. `training_time_s = time.time() - start`.
8. `mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)`.
9. `model.save(f"models/{config['experiment_id']}")` — sb3 dodaje `.zip` automatycznie.
10. `env.close()`.
11. Zwraca `{"mean_reward": float(mean_reward), "std_reward": float(std_reward), "training_time_s": round(training_time_s, 2)}`.

**`run_all_experiments(csv_path: str) -> None`**

1. `experiments = load_experiments(csv_path)`.
2. Iteracja po liście:
   - Jeśli `config["mean_reward"] is not None` — eksperyment ukończony, `continue` (obsługa restartu po crash'u).
   - Wypisz `print(f"[START] {config['experiment_id']} ...")` — jedyna forma progress feedback.
   - `metrics = run_experiment(config)`.
   - `save_results(csv_path, config["experiment_id"], metrics)`.
   - Wypisz `print(f"[DONE] {config['experiment_id']} | mean_reward={metrics['mean_reward']:.1f}")`.
   - `cooldown = get_cooldown_seconds(parse_net_arch(config["net_arch"]))`.
   - `time.sleep(cooldown)` — **zawsze**, nawet po ostatnim eksperymencie.

#### Importy wymagane

```
import time
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from src.config import load_experiments, save_results, parse_net_arch
```

#### Wymagania kodu

- `device="auto"` — bezwzględny wymóg (DKB-004).
- `tb_log_name=config["experiment_id"]` — identyfikacja logów (DKB-005).
- Brak hardkodowanych ścieżek — `csv_path` i katalog `models/` jako parametry lub stałe na poziomie modułu tylko jeśli uzasadnione.
- Katalog `models/` musi istnieć przed zapisem — `Path("models").mkdir(exist_ok=True)` na początku `run_experiment()`.
- Docstringi NumPy, type hints, Ruff-clean.

---

### T-3.2: `tests/test_training.py`

#### Wymagane scenariusze

| ID testu | Opis | Mock targets |
|---|---|---|
| `test_get_cooldown_small_net` | `[64, 64]` → 60 | brak |
| `test_get_cooldown_large_net` | `[1024, 1024, 1024]` → 120 | brak |
| `test_get_cooldown_boundary` | `sum == 1000` → 60 (granica nie jest >, jest <=) | brak |
| `test_run_experiment_returns_metrics` | Poprawny config → dict z kluczami `mean_reward`, `std_reward`, `training_time_s` | `gym.make`, `PPO`, `evaluate_policy` |
| `test_run_experiment_saves_model` | `model.save()` wywołane z poprawną ścieżką | `gym.make`, `PPO`, `evaluate_policy` |
| `test_run_experiment_uses_device_auto` | PPO zainicjalizowane z `device="auto"` | `PPO` |
| `test_run_experiment_tb_log_name` | `model.learn()` wywołane z `tb_log_name=experiment_id` | `PPO` |
| `test_run_all_experiments_sleeps` | `time.sleep()` wywołane z poprawnym cooldownem | `load_experiments`, `run_experiment`, `save_results`, `time.sleep` |
| `test_run_all_experiments_skips_completed` | Eksperyment z `mean_reward != None` → pominięty | `load_experiments`, `run_experiment`, `save_results` |
| `test_run_all_experiments_saves_results` | `save_results` wywołane raz per eksperyment | `load_experiments`, `run_experiment`, `save_results` |

#### Strategia mockowania

Wszystkie mocki przez `unittest.mock.patch`. Patch targets (ścieżki `src.training.*`):

| Cel | Ścieżka patcha |
|---|---|
| Gymnasium | `src.training.gym.make` |
| PPO | `src.training.PPO` |
| evaluate_policy | `src.training.evaluate_policy` |
| time.sleep | `src.training.time.sleep` |
| load_experiments | `src.training.load_experiments` |
| save_results | `src.training.save_results` |

**Przykładowy mock `run_experiment`:**

```python
mock_ppo_instance = MagicMock()
mock_ppo_class = MagicMock(return_value=mock_ppo_instance)
with patch("src.training.PPO", mock_ppo_class):
    with patch("src.training.gym.make", return_value=MagicMock()):
        with patch("src.training.evaluate_policy", return_value=(450.0, 25.0)):
            result = run_experiment(config)
```

#### Wymagania kodu testowego

- Fixtures dla typowych konfiguracji eksperymentów: mała sieć, duża sieć.
- Żaden test nie tworzy środowiska Gymnasium, nie trenuje modelu, nie zapisuje pliku.
- Type hints, docstringi NumPy, Ruff-clean.

---

### T-3.3: Weryfikacja integracji TensorBoard

**Nie wymaga osobnego kodu.** TensorBoard integracja jest efektem ubocznym poprawnej implementacji T-3.1:
- `tensorboard_log="./logs/tensorboard/"` w konstruktorze PPO.
- `tb_log_name=config["experiment_id"]` w `model.learn()`.

Kryteria akceptacji T-3.3 będą zweryfikowane w Epiku 5 (faktyczne uruchomienie treningu).

---

### Kryteria zamknięcia iteracji

| Kryterium | Weryfikacja |
|---|---|
| `ruff check src/ tests/` | exit 0 |
| `ruff format --check src/ tests/` | exit 0 |
| `mypy src/` | exit 0, zero błędów |
| `pytest --cov=src --cov-fail-under=100` | exit 0, 100% coverage |
| `pdoc src/ -o docs/` | exit 0, brak błędów docstringów |

Wszystkie kryteria realizuje `scripts/run_checks.sh`.
