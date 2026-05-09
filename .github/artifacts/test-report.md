# Raport z Testów

> Szablon raportu aktualizowany przez agenta `2-qa.agent.md` po każdym przebiegu testów.
> Wymóg: **100% coverage** wszystkich modułów w `src/`.

---

## Status testów

| Pole | Wartość |
|---|---|
| **Data ostatniego przebiegu** | 2026-05-09 |
| **Wynik ogólny** | ✅ PASS |
| **Coverage** | 100% |
| **Wymagany próg coverage** | 100% |
| **Liczba testów** | 21 |
| **Przechodzące** | 21 |
| **Nieprzechodzące** | 0 |
| **Pominięte** | 0 |

---

## Pokrycie per moduł

| Moduł | Pokrycie | Brakujące linie | Status |
|---|---|---|---|
| `src/__init__.py` | 100% | brak | ✅ |
| `src/config.py` | 100% | brak | ✅ |
| `src/training.py` | 100% | brak | ✅ |
| `src/evaluate.py` | n/d — nie zaimplementowany (Epik 4) | — | ⏳ |

---

## Testy nieprzechodzące

| Test | Moduł | Opis błędu | Przypisany do | Status |
|---|---|---|---|---|
| brak | — | — | — | — |

---

## Brakujące testy

| Moduł | Funkcja/Metoda | Priorytet | Status |
|---|---|---|---|
| `src/evaluate.py` | `evaluate_model()`, `main()` | Wysoki | ⏳ Epik 4 |

---

## Weryfikacja docstringów

| Moduł | Funkcje bez docstringa | Status |
|---|---|---|
| `src/config.py` | brak — wszystkie funkcje pokryte NumPy PL | ✅ |
| `src/training.py` | brak — wszystkie funkcje pokryte NumPy PL | ✅ |
| `src/evaluate.py` | n/d — nie zaimplementowany | ⏳ |

Weryfikacja: `pdoc src/ -o docs/` — exit 0, dokumentacja HTML wygenerowana bez błędów.

---

## Historia przebiegów

| Data | Coverage | Testy pass/fail | Uwagi |
|---|---|---|---|
| 2026-05-09 | 100% | 21/0 | Epiki 2 i 3 — `config.py`, `training.py` i ich testy. Ruff ✅ Mypy strict ✅ pdoc ✅ |

---

## Komendy testowe

```bash
# Uruchomienie testów z coverage
pytest --cov=src --cov-report=term-missing --cov-fail-under=100

# Uruchomienie testów z verbose
pytest -v --cov=src --cov-report=term-missing --cov-fail-under=100

# Uruchomienie pojedynczego pliku testowego
pytest tests/test_config.py -v --cov=src.config --cov-report=term-missing

# Generowanie raportu HTML
pytest --cov=src --cov-report=html
```

---

## Zasady raportowania

1. **Każdy przebieg testów** musi zostać odnotowany w sekcji "Historia przebiegów".
2. **Każdy nieprzechodzący test** musi mieć wpis w sekcji "Testy nieprzechodzące" z opisem błędu i przypisaniem do agenta odpowiedzialnego (`1-developer.agent.md`).
3. **Coverage < 100%** — agent `2-qa.agent.md` musi wskazać brakujące linie i zaplanować dodatkowe testy.
4. **Brak docstringa** w publicznej funkcji/metodzie/klasie to błąd jakościowy — raportowany w sekcji "Weryfikacja docstringów".
