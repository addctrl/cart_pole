# Changelog

Wszystkie istotne zmiany w projekcie są dokumentowane w tym pliku.
Format: [SemVer](https://semver.org/). Typ zmian: Dodane, Zmienione, Naprawione, Usunięte.

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
