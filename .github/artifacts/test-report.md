# Raport z Testów

> Szablon raportu aktualizowany przez agenta `2-qa.agent.md` po każdym przebiegu testów.
> Wymóg: **100% coverage** wszystkich modułów w `src/`.

---

## Status testów

| Pole | Wartość |
|---|---|
| **Data ostatniego przebiegu** | — |
| **Wynik ogólny** | — |
| **Coverage** | — |
| **Wymagany próg coverage** | 100% |
| **Liczba testów** | — |
| **Przechodzące** | — |
| **Nieprzechodzące** | — |
| **Pominięte** | — |

---

## Pokrycie per moduł

| Moduł | Pokrycie | Brakujące linie | Status |
|---|---|---|---|
| `src/config.py` | — | — | — |
| `src/training.py` | — | — | — |
| `src/evaluate.py` | — | — | — |

---

## Testy nieprzechodzące

| Test | Moduł | Opis błędu | Przypisany do | Status |
|---|---|---|---|---|
| — | — | — | — | — |

---

## Brakujące testy

| Moduł | Funkcja/Metoda | Priorytet | Status |
|---|---|---|---|
| — | — | — | — |

---

## Weryfikacja docstringów

| Moduł | Funkcje bez docstringa | Status |
|---|---|---|
| `src/config.py` | — | — |
| `src/training.py` | — | — |
| `src/evaluate.py` | — | — |

---

## Historia przebiegów

| Data | Coverage | Testy pass/fail | Uwagi |
|---|---|---|---|
| — | — | — | — |

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
