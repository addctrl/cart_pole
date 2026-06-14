# Changelog

Wszystkie istotne zmiany w projekcie są dokumentowane w tym pliku.
Format: [SemVer](https://semver.org/). Typ zmian: Dodane, Zmienione, Naprawione, Usunięte.

## [0.6.4] - 2026-06-14

### Dodane
- `src/humanoid_production.py` — produkcyjny skrypt treningu `Humanoid-v5` na 30M kroków z auto-resume oraz spójnym checkpointem (model + `VecNormalize`) oparty o najlepsze parametry z Optuny
- `scripts/run_humanoid_production.sh` — wrapper uruchomieniowy dla finalnego treningu produkcyjnego Humanoida
- `src/evaluate_humanoid_production.py` — dedykowany moduł CLI do ewaluacji ostatniego produkcyjnego modelu Humanoida
- `scripts/evaluate_humanoid_production.sh` — wrapper uruchomieniowy ewaluacji modelu `models/humanoid_prod/latest_model.zip`
- `src/tensorboard_export.py` — moduł CLI do eksportu scalarów TensorBoard z wielu runów jednocześnie do CSV (`pivot` i opcjonalnie `long`)
- `src/objective_score_csv.py` — moduł CLI do hurtowego przeliczania `objective_score = mean_reward - penalty * std_reward` dla jednego lub wielu plików CSV
- `tests/test_tensorboard_export.py` — testy jednostkowe eksportera TensorBoard z pełnym pokryciem modułu
- `tests/test_objective_score_csv.py` — testy jednostkowe przeliczania `objective_score` z pełnym pokryciem modułu
- `scripts/export_tensorboard_csv.sh` — wrapper uruchomieniowy do szybkiego eksportu danych TensorBoard
- `scripts/recompute_objective_scores.sh` — wrapper do przeliczania `objective_score` in-place dla wielu plików CSV
- `scripts/evaluate_cartpole.sh` — wrapper do ewaluacji najlepszego modelu CartPole z auto-wyboru po wynikach CSV
- `scripts/evaluate_lunarlander.sh` — wrapper do ewaluacji najlepszego modelu LunarLander z auto-wyboru po wynikach CSV

### Zmienione
- `README.md` — dodano sekcję analityczną: eksport TensorBoard do CSV (wiele runów) oraz ujednolicenie `objective_score` między seriami eksperymentów

## [0.6.3] - 2026-06-13

### Dodane
- `src/lunarlander_bayes.py` — dedykowany moduł CLI do optymalizacji bayesowskiej PPO dla `LunarLander-v3` z porównaniem architektur `[64, 64]` i `[128, 128]` w eksperymencie pre5
- `tests/test_lunarlander_bayes.py` — testy jednostkowe nowego runnera pre5 z pełnym pokryciem modułu

### Zmienione
- `.github/artifacts/adr.md` — dodano ADR-007 dokumentujący izolację eksperymentu pre5 dla LunarLandera
- `.github/artifacts/architecture_and_tasks.md` — dodano zadania T-5.6 i T-5.7 dla bayesowskiego porównania architektur `64 x 64` i `128 x 128`

## [0.6.2] - 2026-06-13

### Zmienione
- `src/humanoid_bayes.py` — dodano trwałe storage Optuny (`sqlite:///data/humanoid_optuna.db`) oraz `load_if_exists=True`, co umożliwia resume studium po awarii/przerwaniu
- `src/humanoid_bayes.py` — dodano parametr CLI `--optuna-storage` do jawnego sterowania backendem storage
- `tests/test_humanoid_bayes.py` — testy rozszerzono o asercje dla `storage` i `load_if_exists` w `create_study`
- `README.md` — doprecyzowano działanie resume i lokalnej bazy Optuny

## [0.6.1] - 2026-06-13

### Zmienione
- `src/humanoid_bayes.py` — rozszerzono search space PPO dla `Humanoid-v5` o `gae_lambda`, `clip_range`, `target_kl`, `n_epochs`, `vf_coef` i `normalize_advantage`, zgodnie z praktyką strojenia PPO w zadaniach ciągłej kontroli
- `src/humanoid_bayes.py` — funkcja celu Optuny dla Humanoida uwzględnia teraz stabilność polityki przez `objective_score = mean_reward - stability_penalty * std_reward`
- `README.md` — rekomendowana komenda Humanoida zawiera `--eval-episodes` i `--stability-penalty`, a opis wariantu odwołuje się do wniosków z dwóch zewnętrznych źródeł
- `tests/test_humanoid_bayes.py` — testy rozszerzono o nowe hiperparametry PPO i scoring stabilnościowy

## [0.6.0] - 2026-06-13

### Dodane
- `src/humanoid_bayes.py` — dedykowany moduł CLI do optymalizacji bayesowskiej PPO dla `Humanoid-v5` na stałej architekturze `[256, 256]`
- `tests/test_humanoid_bayes.py` — testy jednostkowe odseparowanego wariantu Humanoid
- `requirements-humanoid.txt` — opcjonalny zestaw zależności dla MuJoCo i Optuny, bez ingerencji w bazowy pipeline
- `DKB-008` w `.github/artifacts/dev_knowledge_base.md` — zasady izolacji zależności Humanoida

### Zmienione
- `.github/artifacts/architecture_and_tasks.md` — dodane zadania T-5.4 i T-5.5 dla eksperymentu 5 z Humanoidem
- `.github/artifacts/adr.md` — dodano ADR-006 dokumentujący izolację wariantu Humanoid
- `README.md` — instrukcja uruchomienia eksperymentu 5 dla `Humanoid-v5` z rekomendowanym pruningiem i większym budżetem niż smoke test
- `src/humanoid_bayes.py` — studium Humanoida raportuje wyniki pośrednie i realnie wspiera `MedianPruner`, zamiast czekać na pełne zakończenie każdej próby

## [0.5.2] - 2026-06-13

### Dodane
- `data/lunarlander_experiments.csv` — nowa seria OFAT dla `LunarLander-v3` na największej sieci `[1024, 1024, 1024]`: baseline zwycięzcy oraz wszystkie warianty hiperparametrów przy `600000` kroków

## [0.5.1] - 2026-06-13

### Dodane
- `.github/artifacts/lunarlander_analysis.md` — szczegółowa analiza wyników 5 treningów LunarLander-v3 i rekomendacja najlepszego modelu do demo

### Zmienione
- `README.md` — dodana komenda ewaluacji najlepszego modelu LunarLander oraz odwołanie do raportu analitycznego

## [0.5.0] - 2026-06-13

### Dodane
- `data/lunarlander_experiments.csv` — plan 5 treningów dla `LunarLander-v3`, każdy po 300000 kroków
- `DKB-007` w `.github/artifacts/dev_knowledge_base.md` — opis zależności Box2D i przejścia z `LunarLander-v2` na `LunarLander-v3`

### Zmienione
- `requirements.txt` — rozszerzenie `gymnasium` o `box2d` oraz dodanie `swig` dla LunarLander
- `README.md` — instrukcja uruchomienia serii LunarLander i opis 5 wybranych konfiguracji

## [0.4.1] - 2026-06-13

### Dodane
- `.github/artifacts/cartpole_analysis.md` — szczegółowa analiza porównawcza etapu 1 i etapu 2 dla CartPole-v1 z rekomendacjami do LunarLander-v2

### Zmienione
- `data/experiments.csv` — identyfikatory etapu 1 zostały ujednolicone do czytelnego schematu nazw zgodnego z etapem 2
- `models/` i `logs/tensorboard/` — nazwy artefaktów etapu 1 zostały wyrównane do nowych identyfikatorów eksperymentów
- `README.md` — przykłady ścieżek zaktualizowane do nowych identyfikatorów modeli i logów

## [0.4.0] - 2026-06-13

### Dodane
- `src/training.py` — CLI `python -m src.training --csv ...` zgodne z dokumentacją operacyjną projektu

### Zmienione
- `src/training.py` — środowisko treningowe jest opakowane w `Monitor`, co stabilizuje i uwiarygadnia metryki epizodów podczas ewaluacji polityki
- `tests/test_training.py` — test CLI dla modułu treningowego i dostosowanie mocków do `Monitor`
- `README.md` — pełna instrukcja operacyjna dla epika 5: komendy treningu, TensorBoard, ranking wyników i tabela wariantów hiperparametrów

## [0.3.1] - 2026-06-13

### Zmienione
- `data/experiments.csv` — rozbudowano macierz eksperymentów CartPole-v1 do 33 treningów zgodnych z OFAT: 3 architektury sieci i 5 hiperparametrów w wariantach mały/optymalny/duży przy stałym baseline dla każdej sieci

## [0.3.0] - 2026-06-13

### Dodane
- `src/evaluate.py` — skrypt ewaluacyjny PPO z CLI (`--model-path`, `--env-id`, `--episodes`) i renderowaniem `render_mode="human"`
- `tests/test_evaluate.py` — testy jednostkowe modułu ewaluacji z mockowaniem modelu, środowiska i argumentów CLI

### Zmienione
- `README.md` — instrukcja uruchomienia ewaluacji na macOS z `SDL_VIDEODRIVER=cocoa`
- `.github/artifacts/architecture_and_tasks.md` — oznaczenie zadań T-4.1 i T-4.2 jako ukończonych

## [0.2.0] - 2026-05-09

### Dodane
- `src/config.py` — moduł I/O CSV: `parse_net_arch()`, `load_experiments()`, `save_results()` z pełną dokumentacją NumPy i type hints
- `src/training.py` — pętla treningowa PPO: `get_cooldown_seconds()`, `run_experiment()`, `run_all_experiments()` z integracją TensorBoard i cooldownem dla M4
- `tests/test_config.py` — 11 testów jednostkowych, 100% coverage `src/config.py`
- `tests/test_training.py` — 10 testów jednostkowych (pełne mockowanie), 100% coverage `src/training.py`

## [0.1.0] - 2026-05-09

### Dodane
- Struktura katalogów projektu (`src/`, `tests/`, `data/`, `logs/`, `models/`, `docs/`, `scripts/`)
- `requirements.txt` z pinowanymi zależnościami (Python 3.12, Apple Silicon M4)
- `pyproject.toml` z konfiguracją Ruff, Mypy, Pytest
- `data/experiments.csv` — macierz 13 eksperymentów CartPole-v1 (strategia OFAT)
- `.github/workflows/gatekeeper.yml` — pipeline CI/CD na PR do `main`
- `scripts/run_checks.sh` — lokalny odpowiednik gatekeepingu (jeden skrypt)
