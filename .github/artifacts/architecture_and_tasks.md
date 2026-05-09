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

- [ ] **T-1.1:** Utworzenie `requirements.txt` z pinowanymi wersjami zależności
  - Kryteria akceptacji: `pip install -r requirements.txt` w czystym venv kończy się sukcesem.
  - Zależności: `gymnasium`, `stable-baselines3`, `tensorboard`, `pygame`, `pytest`, `pytest-cov`, `ruff`, `mypy`, `pdoc`.
- [ ] **T-1.2:** Utworzenie `pyproject.toml` z konfiguracją Ruff, Mypy, Pytest
  - Kryteria akceptacji: `ruff check .` i `mypy src/` działają bez błędów konfiguracyjnych.
- [ ] **T-1.3:** Utworzenie struktury katalogów (`src/`, `tests/`, `data/`, `logs/`, `models/`, `docs/`)
  - Kryteria akceptacji: Katalogi istnieją, pliki `__init__.py` w `src/` i `tests/`.
- [ ] **T-1.4:** Utworzenie pliku `data/experiments.csv` z macierzą eksperymentów
  - Kryteria akceptacji: CSV zawiera kolumny: `experiment_id`, `env_id`, `net_arch`, `learning_rate`, `batch_size`, `gamma`, `n_steps`, `ent_coef`, `total_timesteps`. Kolumny wynikowe (puste): `mean_reward`, `std_reward`, `training_time_s`.
- [ ] **T-1.5:** Konfiguracja `.github/workflows/gatekeeper.yml`
  - Kryteria akceptacji: Pipeline uruchamia się na PR do `main`. Etapy: Ruff, Mypy, Pytest.

### Epik 2: Implementacja modułu konfiguracji

- [ ] **T-2.1:** Implementacja `src/config.py`
  - `load_experiments()` — wczytanie CSV do `list[dict]`.
  - `save_results()` — dopisanie metryk do CSV.
  - `parse_net_arch()` — parsowanie stringa architektury.
  - Kryteria akceptacji: Pełna dokumentacja NumPy, type hints, brak błędów Ruff/Mypy.
- [ ] **T-2.2:** Testy jednostkowe `tests/test_config.py`
  - Scenariusze: poprawny odczyt CSV, błędny plik, pusty plik, parsowanie architektury, zapis wyników.
  - Kryteria akceptacji: 100% coverage modułu `config.py`.

### Epik 3: Implementacja pętli treningowej

- [ ] **T-3.1:** Implementacja `src/training.py`
  - `run_experiment()` — pojedynczy trening PPO z konfiguracją z CSV.
  - `run_all_experiments()` — pętla po CSV z cooldownem i zapisem wyników.
  - `get_cooldown_seconds()` — logika cooldownu (60s domyślnie, 120s dla dużych sieci).
  - Kryteria akceptacji: Pełna dokumentacja NumPy, type hints, logowanie TensorBoard, zapis modelu.
- [ ] **T-3.2:** Testy jednostkowe `tests/test_training.py`
  - Scenariusze: mockowany trening, weryfikacja cooldownu, weryfikacja zapisu modelu, weryfikacja logów TensorBoard.
  - Kryteria akceptacji: 100% coverage modułu `training.py`. Brak fizycznego treningu w testach.
- [ ] **T-3.3:** Integracja z TensorBoard
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
