# Optymalizacja Hiperparametrów i Architektury Sieci w Algorytmach RL

> Projekt zaliczeniowy — Sztuczna Inteligencja | Dr Maciej Kraszewski

Zautomatyzowany pipeline treningowy do badania wpływu architektury sieci neuronowej i hiperparametrów algorytmu PPO na zbieżność, stabilność i czas treningu agentów uczenia ze wzmocnieniem.

---

## Cel projektu

Udowodnienie analitycznego podejścia inżyniera do procesu uczenia maszynowego. Poprzez twarde dane z logów wykazanie, jak zmiana pojemności sieci neuronowej oraz manipulacja hiperparametrami PPO wpływają na jakość treningu.

### Środowiska

| Środowisko | Rola w projekcie |
|---|---|
| **CartPole-v1** | Baza analityczna — szybkie przemielenie macierzy eksperymentów |
| **LunarLander-v2** | Demo docelowe — weryfikacja najlepszych parametrów na żywo |

### Zakres badawczy

- **3 architektury sieci:** `[16,16]`, `[64,64]`, `[1024,1024,1024]`
- **5 hiperparametrów PPO** w 3 wariantach: `learning_rate`, `batch_size`, `gamma`, `n_steps`, `ent_coef`

---

## Stack technologiczny

| Komponent | Narzędzie |
|---|---|
| Algorytm RL | PPO (stable-baselines3) |
| Środowiska | Gymnasium |
| Język | Python 3.12 |
| Analityka | TensorBoard |
| CI/CD | GitHub Actions |
| Dokumentacja | pdoc (NumPy Style) |

---

## Wymagania systemowe

- **macOS** (Apple Silicon M4)
- **Python 3.12**
- **24 GB RAM** (zunifikowana pamięć)
- **pip** + **venv**

---

## Instalacja

### 1. Klonowanie repozytorium

```bash
git clone https://github.com/<user>/cart_pole.git
cd cart_pole
```

### 2. Utworzenie środowiska wirtualnego

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 3. Instalacja zależności

```bash
pip install -r requirements.txt
```

### 4. Weryfikacja instalacji

```bash
python -c "import gymnasium; import stable_baselines3; print('OK')"
```

---

## Uruchomienie

### Pętla eksperymentów (trening)

```bash
python -m src.training --csv data/experiments.csv
```

Skrypt:
1. Wczytuje macierz konfiguracji z `data/experiments.csv`.
2. Dla każdego wiersza uruchamia trening PPO.
3. Zapisuje wagi modelu do `models/`.
4. Loguje metryki do `logs/tensorboard/`.
5. Dopisuje wyniki (`mean_reward`, `std_reward`, `training_time_s`) do CSV.
6. Implementuje cooldown (`time.sleep`) między eksperymentami (ochrona przed thermal throttlingiem).

### Ewaluacja (demo na żywo)

```bash
python -m src.evaluate --model-path models/exp_001.zip --env-id CartPole-v1 --episodes 5
```

Ładuje wytrenowany model i renderuje grę w trybie graficznym (`render_mode="human"`).

### TensorBoard (analityka)

```bash
tensorboard --logdir=./logs/tensorboard/ --port=6006
```

Dashboard: `http://localhost:6006`

### Testy

```bash
pytest --cov=src --cov-report=term-missing --cov-fail-under=100
```

### Linter i typowanie

```bash
ruff check src/ tests/
ruff format --check src/ tests/
mypy src/ --strict
```

### Generowanie dokumentacji

```bash
pdoc src/ -o docs/
```

---

## Struktura projektu

```
cart_pole/
├── .github/
│   ├── agents/                         # System-prompty agentów AI
│   │   ├── 0-architect.agent.md
│   │   ├── 1-developer.agent.md
│   │   ├── 2-qa.agent.md
│   │   └── 3-gatekeeper-devops.agent.md
│   ├── artifacts/                      # Dokumentacja projektowa
│   │   ├── prd.md                      # Wymagania produktowe
│   │   ├── adr.md                      # Rejestr decyzji
│   │   ├── architecture_and_tasks.md   # Backlog i architektura
│   │   ├── dev_knowledge_base.md       # Baza wiedzy
│   │   └── test-report.md             # Raport z testów
│   └── workflows/
│       └── gatekeeper.yml             # Pipeline CI/CD
├── src/                               # Kod źródłowy
│   ├── __init__.py
│   ├── config.py                      # Ładowanie konfiguracji z CSV
│   ├── training.py                    # Pętla treningowa
│   └── evaluate.py                    # Skrypt ewaluacyjny
├── tests/                             # Testy (pytest)
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_training.py
│   └── test_evaluate.py
├── data/
│   └── experiments.csv                # Konfiguracja + wyniki
├── logs/
│   └── tensorboard/                   # Logi TensorBoard
├── models/                            # Wagi modeli (.zip)
├── docs/                              # Dokumentacja HTML (pdoc)
├── AGENTS.md                          # Manifest współpracy agentów
├── CHANGELOG.md                       # Historia zmian (SemVer)
├── README.md                          # Ten plik
├── requirements.txt                   # Zależności Python
├── pyproject.toml                     # Konfiguracja narzędzi
└── zadanie.md                         # Specyfikacja zadania
```

---

## Dokumentacja projektowa

| Dokument | Ścieżka | Opis |
|---|---|---|
| PRD | `.github/artifacts/prd.md` | Pełne wymagania produktowe |
| ADR | `.github/artifacts/adr.md` | Rejestr decyzji architektonicznych |
| Backlog | `.github/artifacts/architecture_and_tasks.md` | Architektura i harmonogram zadań |
| Baza wiedzy | `.github/artifacts/dev_knowledge_base.md` | Znane problemy i rozwiązania |
| Raport testów | `.github/artifacts/test-report.md` | Status testów i coverage |
| AGENTS | `AGENTS.md` | Zasady współpracy agentów AI |

---

## Licencja

Projekt zaliczeniowy. Użytek edukacyjny.