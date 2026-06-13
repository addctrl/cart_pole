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

### Macierz eksperymentów OFAT

Plik [data/experiments.csv](data/experiments.csv) zawiera pełną macierz **33 treningów** dla CartPole-v1:

- 3 architektury sieci: za mała, optymalna, zbyt duża.
- 1 baseline na każdą architekturę.
- 5 hiperparametrów PPO, każdy w 3 wariantach: za mały, optymalny, za duży.
- W każdym eksperymencie zmieniany jest dokładnie **jeden** czynnik, reszta pozostaje na poziomie bazowym.

Liczba treningów wynika ze wzoru: `3 sieci * (1 baseline + 5 parametrów * 2 odchylenia) = 33`.

### Parametry i warianty

#### Architektura sieci MLP

| Wariant | `net_arch` | Rola badawcza | Uzasadnienie |
|---|---|---|---|
| Za mała | `[16, 16]` | Minimalna pojemność modelu | Ma pokazać, czy zbyt mała sieć nie traci zdolności reprezentacji stanu. |
| Optymalna | `[64, 64]` | Baseline | Sensowny punkt odniesienia dla niskowymiarowego stanu CartPole-v1. |
| Zbyt duża | `[1024, 1024, 1024]` | Test przeinwestowania | Ma ujawnić koszt CPU, dłuższy cooldown i potencjalnie gorszą efektywność. |

#### Hiperparametry PPO

| Parametr | Za co odpowiada | Za mała wartość | Optymalna wartość | Za duża wartość | Dlaczego takie warianty |
|---|---|---|---|---|---|
| `learning_rate` | Tempo aktualizacji wag | `0.0001` | `0.0003` | `0.001` | Zakres obejmuje uczenie zbyt zachowawcze, referencyjne i zbyt agresywne. |
| `batch_size` | Stabilność gradientu i koszt pamięci | `64` | `256` | `512` | Daje porównanie między szybszą aktualizacją a bardziej wygładzonym gradientem. |
| `gamma` | Horyzont planowania agenta | `0.9` | `0.99` | `0.999` | Sprawdza zachowanie agenta krótkowzrocznego, zbalansowanego i bardzo dalekowzrocznego. |
| `n_steps` | Rozmiar rollout buffer przed aktualizacją | `512` | `2048` | `4096` | Pozwala ocenić wpływ krótkich i długich rolloutów na stabilność oraz czas treningu. |
| `ent_coef` | Siła wymuszenia eksploracji | `0.0` | `0.01` | `0.05` | Pokazuje różnicę między brakiem eksploracji dodatkowej, baseline i nadmiernym losowaniem. |

#### Parametry stałe w całym badaniu

| Parametr | Wartość | Uzasadnienie |
|---|---|---|
| `env_id` | `CartPole-v1` | Szybkie środowisko bazowe do zebrania danych analitycznych. |
| `total_timesteps` | `100000` | Wspólny budżet treningowy, aby porównanie było uczciwe między eksperymentami. |
| `device` | `auto` | Na macOS wybierze CPU zgodnie z ograniczeniami SB3 na Apple Silicon. |

### Pętla eksperymentów (trening)

```bash
source .venv/bin/activate
python -m src.training --csv data/experiments.csv
```

Komenda uruchamia pełną pętlę treningową sterowaną z CSV. Skrypt:
1. Wczytuje macierz konfiguracji z `data/experiments.csv`.
2. Dla każdego wiersza uruchamia trening PPO.
3. Zapisuje wagi modelu do `models/`.
4. Loguje metryki do `logs/tensorboard/`.
5. Dopisuje wyniki (`mean_reward`, `std_reward`, `training_time_s`) do CSV.
6. Implementuje cooldown (`time.sleep`) między eksperymentami (ochrona przed thermal throttlingiem).

Uwagi operacyjne:

- Ponowne uruchomienie tej samej komendy działa jak **resume**. Wiersze z uzupełnionym `mean_reward` są pomijane.
- Dla dużej sieci `[1024, 1024, 1024]` cooldown wynosi 120 sekund, dla pozostałych 60 sekund.
- Modele zapisują się jako `models/<experiment_id>.zip`.
- TensorBoard tworzy osobny katalog per eksperyment, np. `logs/tensorboard/exp_012_*`.

### Szybka weryfikacja startu treningu

Jeśli chcesz najpierw sprawdzić sam mechanizm uruchomienia bez pełnych 33 eksperymentów, przygotuj tymczasowy CSV z jednym wierszem i uruchom tę samą komendę `python -m src.training --csv <ścieżka>`. Logika treningu i zapisu artefaktów jest identyczna.

### Ewaluacja (demo na żywo)

```bash
export SDL_VIDEODRIVER=cocoa
python -m src.evaluate --model-path models/exp_001.zip --env-id CartPole-v1 --episodes 5
```

Ładuje wytrenowany model i renderuje grę w trybie graficznym (`render_mode="human"`).
Na macOS Apple Silicon ustawienie `SDL_VIDEODRIVER=cocoa` eliminuje typowe problemy z oknem pygame.

### TensorBoard (analityka)

```bash
source .venv/bin/activate
tensorboard --logdir=./logs/tensorboard/ --port=6006
```

Dashboard: `http://localhost:6006`

Po poprawnym treningu w logach są dostępne co najmniej następujące tagi scalar potwierdzone smoke testem:

- `rollout/ep_rew_mean`
- `rollout/ep_len_mean`
- `time/fps`
- `train/approx_kl`
- `train/clip_fraction`
- `train/clip_range`
- `train/entropy_loss`
- `train/explained_variance`
- `train/learning_rate`
- `train/loss`
- `train/policy_gradient_loss`
- `train/value_loss`

Do analizy jakości treningu najważniejsze są:

- `rollout/ep_rew_mean` — główny sygnał, czy agent faktycznie uczy się rozwiązywać środowisko.
- `train/loss`, `train/value_loss`, `train/policy_gradient_loss` — stabilność optymalizacji i jakość aktualizacji wag.
- `time/fps` oraz kolumna `training_time_s` w CSV — koszt obliczeniowy danego wariantu.

### Jak wskazać najlepszy trening

Po zakończeniu pętli podstawowe dane porównawcze masz w [data/experiments.csv](data/experiments.csv):

- `mean_reward` — główne kryterium sukcesu.
- `std_reward` — stabilność modelu; niższe odchylenie jest lepsze przy podobnej nagrodzie.
- `training_time_s` — koszt czasowy eksperymentu.

Rekomendowana kolejność oceny:

1. Najwyższy `mean_reward`.
2. Przy remisie niższy `std_reward`.
3. Przy dalszym remisie krótszy `training_time_s`.

Do szybkiego rankingu po zakończeniu treningów użyj:

```bash
source .venv/bin/activate
python - <<'PY'
import csv
from pathlib import Path

rows = list(csv.DictReader(Path("data/experiments.csv").open(encoding="utf-8", newline="")))
completed = [row for row in rows if row["mean_reward"]]
ranked = sorted(
	completed,
	key=lambda row: (
		-float(row["mean_reward"]),
		float(row["std_reward"]),
		float(row["training_time_s"]),
	),
)

for row in ranked[:10]:
	print(
		row["experiment_id"],
		row["net_arch"],
		row["learning_rate"],
		row["batch_size"],
		row["gamma"],
		row["n_steps"],
		row["ent_coef"],
		row["mean_reward"],
		row["std_reward"],
		row["training_time_s"],
		sep=", ",
	)
PY
```

Ten ranking nie zastępuje TensorBoard. CSV służy do wyboru zwycięzcy, a TensorBoard do zrozumienia **dlaczego** dany wariant wygrał lub przegrał.

### Komendy pomocnicze

Uruchomienie pełnego gatekeepingu:

```bash
source .venv/bin/activate
./scripts/run_checks.sh
```

Generowanie dokumentacji HTML:

```bash
source .venv/bin/activate
pdoc src/ -o docs/
```

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