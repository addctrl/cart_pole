# ADR — Rejestr Decyzji Architektonicznych

> Każda decyzja architektoniczna podjęta w projekcie jest logowana w tym pliku.
> Format: numer, data, status, kontekst, decyzja, uzasadnienie, konsekwencje.

---

## ADR-001: Wybór algorytmu PPO

| Pole | Wartość |
|---|---|
| **Data** | 2026-05-09 |
| **Status** | Zaakceptowana |
| **Decydent** | Architekt |

### Kontekst

Projekt wymaga algorytmu uczenia ze wzmocnieniem zdolnego do obsługi zarówno dyskretnych (CartPole), jak i ciągłych (potencjalne rozszerzenia) przestrzeni akcji. Algorytm musi być stabilny, dobrze udokumentowany i dostępny w bibliotece `stable-baselines3`.

### Decyzja

Wybrano **Proximal Policy Optimization (PPO)**.

### Uzasadnienie

1. **Stabilność treningu** — PPO stosuje obcinanie (clipping) funkcji celu, co zapobiega destrukcyjnie dużym aktualizacjom wag.
2. **Uniwersalność** — obsługuje przestrzenie dyskretne i ciągłe bez zmiany API.
3. **Prostota konfiguracji** — sb3 dostarcza gotową implementację z sensownymi wartościami domyślnymi.
4. **Dokumentacja** — PPO jest najczęściej używanym algorytmem w sb3, co oznacza obszerną dokumentację i community.
5. **Kompatybilność z celem badawczym** — łatwo parametryzowalna architektura sieci via `policy_kwargs`.

### Konsekwencje

- Inne algorytmy (A2C, DQN, SAC) nie są rozpatrywane. Zgodnie z YAGNI — nie są potrzebne.
- Porównanie algorytmów wykracza poza zakres projektu.

### Odrzucone alternatywy

| Algorytm | Powód odrzucenia |
|---|---|
| DQN | Tylko dyskretne przestrzenie akcji. Brak uniwersalności. |
| A2C | Mniej stabilny niż PPO przy małych batch'ach. |
| SAC | Wymaga ciągłej przestrzeni akcji. Nadmiarowa złożoność. |

---

## ADR-002: CSV jako format danych

| Pole | Wartość |
|---|---|
| **Data** | 2026-05-09 |
| **Status** | Zaakceptowana |
| **Decydent** | Architekt |

### Kontekst

System wymaga mechanizmu przechowywania konfiguracji eksperymentów (wejście) oraz wyników treningu (wyjście). Format musi być prosty, czytelny dla człowieka i nie wymagać dodatkowych zależności.

### Decyzja

Plik **CSV** (`data/experiments.csv`) pełni rolę zarówno pliku konfiguracyjnego, jak i bazy wyników.

### Uzasadnienie

1. **KISS** — CSV jest najprostszym formatem tabelarycznym. Otwiera się w każdym edytorze i arkuszu kalkulacyjnym.
2. **YAGNI** — bazy danych (SQLite, PostgreSQL) to nadmiarowa złożoność dla ~20 wierszy danych.
3. **Zero zależności** — moduł `csv` jest w bibliotece standardowej Pythona.
4. **Transparentność** — plik jest wersjonowany w Git, co zapewnia pełną audytowalność.
5. **Dwukierunkowość** — skrypt czyta parametry z CSV, a po treningu dopisuje metryki do tego samego wiersza.

### Konsekwencje

- Brak równoczesnego zapisu (brak problemu — trening jest sekwencyjny).
- Ograniczona skalowalność — nieistotne przy ~20 eksperymentach.
- Ręczna edycja pliku jest trywialna.

### Odrzucone alternatywy

| Format | Powód odrzucenia |
|---|---|
| SQLite | Nadmiarowa złożoność. Wymaga dodatkowego toolingu do inspekcji. |
| JSON | Mniej czytelny dla danych tabelarycznych. Trudniejszy w ręcznej edycji. |
| YAML | Podatny na błędy formatowania. Nadmiarowy. |

---

## ADR-003: Mechanizm cooldown via `time.sleep()`

| Pole | Wartość |
|---|---|
| **Data** | 2026-05-09 |
| **Status** | Zaakceptowana |
| **Decydent** | Architekt |

### Kontekst

MacBook Air M4 posiada **chłodzenie pasywne**. Sekwencyjne uruchamianie wielu eksperymentów treningowych (każdy obciążający CPU na 100%) prowadzi do thermal throttlingu, który:
1. Wydłuża czas treningu (CPU taktuje niżej).
2. Wprowadza niespójność w pomiarach czasu.
3. Potencjalnie uszkadza sprzęt przy długotrwałym obciążeniu.

### Decyzja

Pętla treningowa implementuje **wymuszony cooldown** via `time.sleep()` między eksperymentami:
- Domyślny: **60 sekund**.
- Dla architektury `[1024, 1024, 1024]`: **120 sekund**.

### Uzasadnienie

1. **Prostota** — `time.sleep()` to jedna linia kodu. Zero dodatkowych zależności.
2. **Deterministyczność** — stały czas przerwy jest przewidywalny i powtarzalny.
3. **Bezpieczeństwo sprzętu** — 60 sekund wystarczy na odprowadzenie ciepła w normalnych warunkach.
4. **Wydłużony cooldown dla dużych sieci** — architektura `[1024, 1024, 1024]` generuje znacząco więcej ciepła.

### Konsekwencje

- Całkowity czas pętli eksperymentów wydłuża się o ~20-40 minut (w zależności od liczby eksperymentów).
- Brak adaptacyjnego monitorowania temperatury (YAGNI — za złożone, za mały zysk).

### Odrzucone alternatywy

| Podejście | Powód odrzucenia |
|---|---|
| Monitoring temperatury CPU (`psutil`) | Nadmiarowa złożoność. Wymaga dodatkowej zależności. Odczyty na M4 mogą być nieprecyzyjne. |
| Brak cooldownu | Ryzyko thermal throttlingu i niespójności danych. |
| Cooldown adaptacyjny | YAGNI. Stały cooldown jest wystarczający. |

---

## ADR-004: Struktura katalogów projektu

| Pole | Wartość |
|---|---|
| **Data** | 2026-05-09 |
| **Status** | Zaakceptowana |
| **Decydent** | Architekt |

### Decyzja

```
cart_pole/
├── .github/
│   ├── agents/           # System-prompty agentów AI
│   ├── artifacts/        # Dokumentacja projektowa (PRD, ADR, backlog)
│   └── workflows/        # GitHub Actions
├── src/                  # Kod źródłowy
│   ├── __init__.py
│   ├── config.py         # Ładowanie konfiguracji z CSV
│   ├── training.py       # Pętla treningowa
│   └── evaluate.py       # Skrypt ewaluacyjny
├── tests/                # Testy (pytest)
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_training.py
│   └── test_evaluate.py
├── data/
│   └── experiments.csv   # Konfiguracja + wyniki
├── logs/
│   └── tensorboard/      # Logi TensorBoard
├── models/               # Wagi modeli (.zip)
├── docs/                 # Dokumentacja HTML (pdoc)
├── AGENTS.md
├── CHANGELOG.md
├── README.md
├── requirements.txt
├── pyproject.toml
└── zadanie.md
```

### Uzasadnienie

- Separacja kodu (`src/`), testów (`tests/`), danych (`data/`), logów (`logs/`) i modeli (`models/`).
- Flat structure w `src/` — brak zagnieżdżonych pakietów. KISS.
- Pliki agentów w `.github/agents/` — konwencja GitHub Copilot.

---

## ADR-005: Python 3.12 jako wersja docelowa

| Pole | Wartość |
|---|---|
| **Data** | 2026-05-09 |
| **Status** | Zaakceptowana |
| **Decydent** | Architekt |

### Decyzja

Projekt targetuje **Python 3.12** jako wersję docelową.

### Uzasadnienie

1. `stable-baselines3` w wersji ≥2.0 wspiera Python 3.8–3.12.
2. Python 3.12 zapewnia najlepszą wydajność (PEP 709 — inlined comprehensions).
3. Pełna kompatybilność z Ruff, Mypy i pytest.
4. Dostępność w GitHub Actions runners.

### Konsekwencje

- Kod nie musi zachowywać kompatybilności wstecznej z Python < 3.12.
- Type hints mogą korzystać z `X | Y` zamiast `Union[X, Y]`.
