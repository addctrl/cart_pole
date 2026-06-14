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
| **LunarLander-v3** | Demo docelowe i eksperyment pre5 — weryfikacja najlepszych parametrów na żywo oraz porównanie architektur w studium Optuny |

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

**Cart-pole**
```bash
source .venv/bin/activate
python -m src.training --csv data/experiments.csv
```

**Lunarlander**
```bash
source .venv/bin/activate
python -m src.training --csv data/lunarlander_experiments.csv
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
- TensorBoard tworzy osobny katalog per eksperyment, np. `logs/tensorboard/exp_012_s64x64_baseline_stage1_*`.

### LunarLander-v3 — plan 5 treningów

Druga gra korzysta z osobnego pliku [data/lunarlander_experiments.csv](data/lunarlander_experiments.csv). Zawiera 5 treningów po `300000` kroków:

- `ll_001_s64x64_batch_large_primary` — główny kandydat startowy, oparty o najlepszy kompromis z CartPole.
- `ll_002_s64x64_ent_zero_alt` — wariant alternatywny z wyłączoną dodatkową entropią.
- `ll_003_s64x64_lr_high_alt` — wariant alternatywny z wyższym learning rate.
- `ll_004_s16x16_gamma_high_compare` — najlepszy przedstawiciel małej sieci do porównania.
- `ll_005_s1024x1024x1024_batch_large_compare` — najlepszy przedstawiciel dużej sieci do porównania.

Uruchomienie:

```bash
source .venv/bin/activate
python -m src.training --csv data/lunarlander_experiments.csv
```

Wymagane zależności dla LunarLander:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Jeśli środowisko było utworzone przed dodaniem Box2D, wykonaj ponowną instalację zależności. `LunarLander-v3` wymaga `swig` i `gymnasium[box2d]`.

### LunarLander-v3 — eksperyment pre5 z optymalizacją bayesowską

Eksperyment pre5 jest **odseparowany** od bazowego pipeline'u CSV. Zamiast ręcznie przygotowanej listy wariantów uruchamia osobne studium Optuny, które porównuje wyłącznie dwie architektury:

- `[64, 64]` — lekki kandydat, który w poprzednich eksperymentach dawał sensowny koszt i pojedyncze dodatnie wyniki,
- `[128, 128]` — wariant pośredni, sprawdzający czy większa pojemność poprawi LunarLandera bez kosztu sieci `[1024, 1024, 1024]`.

Runner `python -m src.lunarlander_bayes` korzysta z tej samej idei co opisany artykuł o PPO hyperparameter optimization:

- search space jest głównie **zdyskretyzowany**, aby ograniczyć liczbę kombinacji na CPU,
- optimizer używa **TPE** i wykorzystuje historię prób do kolejnych sugestii,
- trening raportuje wyniki pośrednie, dzięki czemu **MedianPruner** może ucinać słabe konfiguracje,
- funkcja celu premiuje nie tylko średni reward, ale też stabilność polityki: `objective_score = mean_reward - stability_penalty * std_reward`.

Instalacja zależności dla workflow bayesowskiego:

```bash
source .venv/bin/activate
pip install -r requirements-humanoid.txt
```

Uruchomienie studium pre5:

```bash
source .venv/bin/activate
python -m src.lunarlander_bayes --trials 40 --timesteps 300000 --startup-trials 8 --pruner-warmup-steps 50000 --report-interval-timesteps 50000 --eval-episodes 20 --stability-penalty 0.1 --results-csv data/lunarlander_bayes_results.csv
```

Domyślne artefakty i resume:

- wyniki prób trafiają do `data/lunarlander_bayes_results.csv`,
- storage Optuny jest domyślnie zapisywany do `sqlite:///data/lunarlander_optuna.db`,
- ponowne uruchomienie tej samej komendy wznawia istniejące studium (`load_if_exists=True`),
- URI storage można nadpisać parametrem `--optuna-storage`.

Założenia pre5:

- środowisko: `LunarLander-v3`,
- architektury: `[64, 64]` oraz `[128, 128]`,
- algorytm: `PPO`,
- metoda strojenia: `Optuna TPE`,
- strojone hiperparametry: `learning_rate`, `batch_size`, `gamma`, `n_steps`, `ent_coef`, `gae_lambda`, `clip_range`, `target_kl`, `n_epochs`, `vf_coef`, `normalize_advantage`,
- early stopping: `MedianPruner` po raportach pośrednich,
- artefakty: modele w `models/`, TensorBoard w `logs/tensorboard/`, wyniki prób i statusy w `data/lunarlander_bayes_results.csv`.

Jeśli chcesz wykonać tani smoke test samego workflow, użyj mniejszego budżetu:

```bash
source .venv/bin/activate
python -m src.lunarlander_bayes --trials 2 --timesteps 100000 --startup-trials 1 --pruner-warmup-steps 50000 --report-interval-timesteps 50000 --eval-episodes 10 --stability-penalty 0.0 --results-csv data/lunarlander_bayes_smoke.csv
```

To studium nie rozszerza `python -m src.training --csv ...`. To świadoma decyzja architektoniczna: OFAT i Optuna odpowiadają na inne pytania badawcze i produkują inne artefakty.

### Humanoid-v5 — eksperyment 5 z optymalizacją bayesowską

Wariant Humanoid jest **celowo odseparowany** od bazowego pipeline'u CartPole/LunarLander. Używa jednej architektury sieci `256 x 256` i stroi wyłącznie hiperparametry PPO metodą bayesowską.

Instalacja opcjonalnych zależności tylko dla tej wariacji:

```bash
source .venv/bin/activate
pip install -r requirements-humanoid.txt
```

Uruchomienie studium:

```bash
source .venv/bin/activate
python -m src.humanoid_bayes --trials 30 --timesteps 1000000 --startup-trials 5 --pruner-warmup-steps 100000 --report-interval-timesteps 100000 --eval-episodes 20 --stability-penalty 0.1 --results-csv data/humanoid_bayes_results.csv
```

Start ewaluacji najlepszego modelu Humanoida:

```bash
export SDL_VIDEODRIVER=cocoa
source .venv/bin/activate
python -m src.evaluate --model-path models/<best_humanoid_model>.zip --env-id Humanoid-v5 --episodes 10
```

Start ewaluacji najlepszego modelu CartPole przez wrapper:

```bash
bash scripts/evaluate_cartpole.sh
```

Start ewaluacji najlepszego modelu LunarLander przez wrapper:

```bash
bash scripts/evaluate_lunarlander.sh
```

Resume i dashboard Optuny:

- domyślnie studium zapisuje się do `sqlite:///data/humanoid_optuna.db`,
- po przerwaniu procesu kolejne uruchomienie komendy wznawia to samo studium (`load_if_exists=True`),
- opcjonalnie URI storage można nadpisać parametrem `--optuna-storage`.

Dlaczego nie `10 x 300000`:

- dla `Humanoid-v5` taki budżet jest praktycznie smoke testem, a nie sensownym strojeniem PPO,
- TPE potrzebuje fazy startup zanim zacznie realnie wykorzystywać historię prób,
- sam `MedianPruner` bez raportów pośrednich nic nie daje, więc moduł trenuje teraz etapami i raportuje wynik co `100000` kroków,
- dla Humanoida sama średnia nagroda jest zbyt szumna, dlatego scorer Optuny używa teraz `mean_reward - 0.1 * std_reward`.

Jeśli chcesz tylko sprawdzić, czy środowisko i MuJoCo startują poprawnie, użyj krótszego smoke testu:

```bash
source .venv/bin/activate
python -m src.humanoid_bayes --trials 2 --timesteps 100000 --startup-trials 1 --pruner-warmup-steps 50000 --report-interval-timesteps 50000 --eval-episodes 10 --stability-penalty 0.0 --results-csv data/humanoid_bayes_smoke.csv
```

Źródła, na których opiera się ta konfiguracja:

- Joel Baptista: w PPO istotne są nie tylko `learning_rate`, `batch_size`, `n_steps`, ale też `clip_range`, `target_kl`, `gae_lambda`, `n_epochs`, `vf_coef` i `normalize_advantage`.
- arXiv 2406.18293: dla Humanoida warto uwzględniać stabilność polityki, a nie wyłącznie średni wynik, dlatego funkcja celu zawiera karę za wariancję.

Założenia eksperymentu 5:

- środowisko: `Humanoid-v5`
- architektura: `[256, 256]`
- algorytm: `PPO`
- metoda strojenia: `Optuna TPE` (optymalizacja bayesowska)
- strojenie PPO obejmuje także `gae_lambda`, `clip_range`, `target_kl`, `n_epochs`, `vf_coef` i `normalize_advantage`
- early stopping: `MedianPruner`, aktywowany po raportach pośrednich z treningu
- funkcja celu: `objective_score = mean_reward - stability_penalty * std_reward`
- artefakty: modele w `models/`, TensorBoard w `logs/tensorboard/`, wyniki prób w `data/humanoid_bayes_results.csv` z kolumnami `status`, `trained_timesteps` i `objective_score`

Istniejący `python -m src.training --csv ...` nie został rozszerzony o tryb Humanoida. To świadoma decyzja izolująca nową bibliotekę i nowy workflow od dotychczasowego kodu.

Jeśli chcesz uprościć uruchamianie, można przenieść te komendy do małych skryptów w `scripts/` albo makefile-like wrapperów. To dobry pomysł dla komend, które odpalasz często, zwłaszcza dla:

- treningu CartPole,
- treningu LunarLander,
- pre5 dla LunarLandera,
- Humanoida,
- TensorBoard i ewaluacji.

W tej chwili README pokazuje komendy bezpośrednie, ale technicznie da się je opakować w proste skrypty bashowe bez zmiany logiki projektu.

Gotowe wrappery są już w katalogu `scripts/` i odpowiadają najczęstszym operacjom:

- `scripts/run_cartpole_training.sh`
- `scripts/run_lunarlander_training.sh`
- `scripts/run_lunarlander_pre5.sh`
- `scripts/run_humanoid_bayes.sh`
- `scripts/run_tensorboard.sh`
- `scripts/evaluate_cartpole.sh [model_path] [episodes]`
- `scripts/evaluate_lunarlander.sh [model_path] [episodes]`
- `scripts/evaluate_humanoid.sh [model_path] [episodes]`
- `scripts/export_tensorboard_csv.sh [logdir] [output_csv] [tag1 tag2 ...]`
- `scripts/recompute_objective_scores.sh [stability_penalty] [csv1 csv2 ...]`

Każdy z nich zakłada aktywację `.venv` wewnątrz skryptu i uruchamia gotową komendę bez ręcznego klepania parametrów.
### Szybka weryfikacja startu treningu

Jeśli chcesz najpierw sprawdzić sam mechanizm uruchomienia bez pełnych 33 eksperymentów, przygotuj tymczasowy CSV z jednym wierszem i uruchom tę samą komendę `python -m src.training --csv <ścieżka>`. Logika treningu i zapisu artefaktów jest identyczna.

### Ewaluacja (demo na żywo)

```bash
export SDL_VIDEODRIVER=cocoa
python -m src.evaluate --model-path models/exp_016_s64x64_batch_large_stage1.zip --env-id CartPole-v1 --episodes 5
```

Ewaluacja najlepszego modelu LunarLander:

```bash
export SDL_VIDEODRIVER=cocoa
python -m src.evaluate --model-path models/ll_012_s1024x1024x1024_gamma_high_tune600k.zip --env-id LunarLander-v3 --episodes 10
```

Ładuje wytrenowany model i renderuje grę w trybie graficznym (`render_mode="human"`).
Na macOS Apple Silicon ustawienie `SDL_VIDEODRIVER=cocoa` eliminuje typowe problemy z oknem pygame.

### TensorBoard (analityka)

```bash
source .venv/bin/activate
tensorboard --logdir=./logs/tensorboard/ --port=6006
```

Dashboard: `http://localhost:6006`

### Eksport TensorBoard do CSV (wiele runow naraz)

Do eksportu danych z wielu treningow jednoczesnie uzyj wrappera:

```bash
bash scripts/export_tensorboard_csv.sh
```

Domyslnie zapisze:

- `data/tensorboard_scalars_pivot.csv` (uklad pivot: kolumny `run`, `step`, tagi),
- `data/tensorboard_scalars_long.csv` (uklad long: `run`, `step`, `tag`, `value`).

Mozesz wskazac inny katalog logow i plik wyjsciowy:

```bash
bash scripts/export_tensorboard_csv.sh logs/tensorboard data/porownanie_treningow.csv
```

Mozesz tez filtrowac tylko wybrane tagi (np. do raportu):

```bash
bash scripts/export_tensorboard_csv.sh logs/tensorboard data/porownanie_treningow.csv rollout/ep_rew_mean train/loss train/value_loss train/policy_gradient_loss
```

### Wspolny objective_score dla wszystkich serii

Aby miec jeden punkt odniesienia miedzy CartPole, LunarLander i Humanoid, przelicz `objective_score` hurtowo:

```bash
bash scripts/recompute_objective_scores.sh 0.1
```

Domyslnie skrypt przelicza i nadpisuje:

- `data/experiments.csv`,
- `data/lunarlander_experiments.csv`,
- `data/humanoid_bayes_results.csv`.

Mozesz tez podac wlasna liste plikow:

```bash
bash scripts/recompute_objective_scores.sh 0.1 data/experiments.csv data/lunarlander_experiments.csv data/humanoid_bayes_results_512x512.csv
```

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
| Analiza CartPole | `.github/artifacts/cartpole_analysis.md` | Szczegółowa analiza wyników etapu 1 i 2 oraz rekomendacje do LunarLander |
| Analiza LunarLander | `.github/artifacts/lunarlander_analysis.md` | Szczegółowa analiza 5 treningów LunarLander oraz rekomendacja modelu do demo |
| Raport testów | `.github/artifacts/test-report.md` | Status testów i coverage |
| AGENTS | `AGENTS.md` | Zasady współpracy agentów AI |

---

## Licencja

Projekt zaliczeniowy. Użytek edukacyjny.