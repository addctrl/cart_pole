# Prezentacja RL — Optymalizacja Hiperparametrów i Architektury Sieci
**Przedmiot:** Sztuczna inteligencja
**Wykładowca:** Dr. Maciej Kraszewski
**Sprzęt:** MacBook Air M4 (24 GB RAM, chłodzenie pasywne)

---

## Wymagania projektu

1. Wytrenować agenta grającego w grę z biblioteki Gymnasium.
2. Nie wybieramy gier z grupy Toy Text.
3. Należy udokumentować pracę z agentami AI / LLMami (historia promptów).
4. Podczas prezentacji:
    - opis gry (cel, przestrzeń akcji, przestrzeń stanu, system nagród),
    - przetestowane architektury sieci neuronowych (jak wpływa na wynik, tempo uczenia itd.),
    - przetestowanie hiperparametrów,
    - demo live wytrenowanego agenta.

---

## Architektura projektu

- **Algorytm:** PPO (`stable-baselines3`)
- **Polityka:** `MlpPolicy` z parametryzowaną architekturą via `policy_kwargs=dict(net_arch=...)`
- **Dane wejściowe/wyjściowe:** CSV (zero baz danych)
- **Logowanie:** TensorBoard (natywny callback SB3)
- **Metodologia:** OFAT (One Factor At a Time) → Optuna (optymalizacja bayesowska)
- **Pipeline:** w pełni zautomatyzowany — CSV steruje treningiem, wyniki zapisywane automatycznie
- **CI/CD:** GitHub Actions (Ruff + Mypy + Pytest, 100% coverage)
- **Cooldown termiczny:** `time.sleep()` między eksperymentami (60s / 120s dla dużych sieci)

---

## Przygotowanie (Epik 0–1)

- Opracowanie PRD (Product Requirements Document)
- Struktura katalogów projektu (`src/`, `tests/`, `data/`, `logs/`, `models/`, `docs/`)
- `requirements.txt` z pinowanymi zależnościami (Python 3.12, Apple Silicon M4)
- `pyproject.toml` z konfiguracją Ruff, Mypy, Pytest
- Pipeline CI/CD `.github/workflows/gatekeeper.yml`
- Scenariusze treningowe: `data/experiments.csv` — macierz 33 eksperymentów OFAT

### Moduły źródłowe (Epik 2–4)

| Moduł | Odpowiedzialność |
|---|---|
| `src/config.py` | I/O CSV: `parse_net_arch()`, `load_experiments()`, `save_results()` |
| `src/training.py` | Pętla treningowa PPO z cooldownem i TensorBoard |
| `src/evaluate.py` | Ewaluacja modelu z `render_mode="human"` (CLI) |

---

## Etap 1 — CartPole-v1, OFAT, 100 000 kroków

### Opis gry

| Parametr | Wartość |
|---|---|
| Typ | Classic Control |
| Przestrzeń stanu | `Box(4,)` — pozycja wózka, prędkość wózka, kąt drążka, prędkość kątowa |
| Przestrzeń akcji | `Discrete(2)` — lewo/prawo |
| Nagroda | +1 za każdy krok utrzymania drążka |
| Zakończenie | Drążek > 12°, wózek poza granicami, lub 500 kroków |
| Sufit jakości | `mean_reward = 500.0`, `std_reward = 0.0` |

### Zakres badawczy

- **3 architektury sieci:** `[16, 16]` (za mała), `[64, 64]` (optymalna), `[1024, 1024, 1024]` (zbyt duża)
- **5 hiperparametrów PPO:** `learning_rate`, `batch_size`, `gamma`, `n_steps`, `ent_coef` — każdy w 3 wariantach
- **33 treningi** (wzór: `3 sieci × (1 baseline + 5 param × 2 odchylenia) = 33`)
- **Strategia:** OFAT — zmiana jednego czynnika na raz, reszta na poziomie bazowym

### Wyniki zbiorcze

- **17/33** eksperymentów osiągnęło pełne rozwiązanie (`mean_reward=500`, `std_reward=0`)
- Średni `mean_reward`: 451.15
- Średni `training_time_s`: 39.92

### Top 5 modeli

| # | Experiment ID | Architektura | Wyróżnik | `mean_reward` | `std_reward` | Czas [s] |
|---|---|---|---|---|---|---|
| 1 | `exp_007_s16x16_gamma_high` | `[16, 16]` | `gamma=0.999` | 500.0 | 0.0 | 12.10 |
| 2 | `exp_016_s64x64_batch_large` | `[64, 64]` | `batch_size=512` | 500.0 | 0.0 | 12.24 |
| 3 | `exp_021_s64x64_ent_zero` | `[64, 64]` | `ent_coef=0.0` | 500.0 | 0.0 | 12.60 |
| 4 | `exp_014_s64x64_lr_high` | `[64, 64]` | `lr=0.001` | 500.0 | 0.0 | 12.79 |
| 5 | `exp_022_s64x64_ent_high` | `[64, 64]` | `ent_coef=0.05` | 500.0 | 0.0 | 12.95 |

### Wnioski per architektura

#### `[16, 16]` — za mała
- 1/11 pełnych rozwiązań. Średni wynik: 394.36.
- Szybkie treningi, ale zbyt krucha — wrażliwa na dobór hiperparametrów.
- Jedyny sukces: wysoki `gamma=0.999` (długi horyzont planowania kompensował mały model).

#### `[64, 64]` — sweet spot
- **7/11** pełnych rozwiązań. Średni wynik: 466.82. Średni czas: 13.63s.
- Najlepszy kompromis jakości, stabilności i kosztu obliczeniowego.
- Różne warianty hiperparametrów rozwiązywały zadanie stabilnie i szybko.

#### `[1024, 1024, 1024]` — przeinwestowanie
- 9/11 pełnych rozwiązań. Średni wynik: 492.26. Średni czas: **93.55s**.
- Działa, ale koszt nieproporcjonalny do zysku. Klasyczny przykład naruszenia YAGNI.
- Nie robi tego lepiej od `[64, 64]` — robi to jedynie 7× drożej.

### Analiza TensorBoard (do zaprezentowania)

| Metryka | Co pokazuje | Na co patrzeć |
|---|---|---|
| `rollout/ep_rew_mean` | Główna krzywa uczenia | Szybkość dojścia do 500, płynność wzrostu |
| `rollout/ep_len_mean` | Odpowiednik jakości (na CartPole = reward) | Spójność z `ep_rew_mean` |
| `train/loss`, `train/value_loss` | Stabilność optymalizacji | Brak eksplozji, stabilny przebieg po plateau |
| `train/entropy_loss` | Wpływ `ent_coef` | Tempo usztywniania polityki |
| `time/fps` | Koszt obliczeniowy | `[1024,1024,1024]` wyraźnie gorszy |

---

## Etap 1.5 — Dogrywka finalistów, 300 000 kroków

- 10 najlepszych modeli z etapu 1, dogrywka 3× dłuższy trening.
- **Wynik: 10/10 modeli osiągnęło `500.0` / `0.0`.**
- Czas wzrósł ~3× (np. 12s → 37s, 72s → 229s).
- **Wniosek: dogrywka nie poprawiła jakości.** CartPole ma sufit jakości osiągnięty już w etapie 1. Etap 2 potwierdził stabilność, nie przyniósł odkryć.

---

## Etap 2 — LunarLander-v3, OFAT, 300 000 kroków

### Opis gry

| Parametr | Wartość |
|---|---|
| Typ | Box2D |
| Przestrzeń stanu | `Box(8,)` — pozycja, prędkość, kąt, kontakt nóg |
| Przestrzeń akcji | `Discrete(4)` — nic, lewy silnik, główny silnik, prawy silnik |
| Nagroda | Złożona: lądowanie +100..+140, crash −100, noga na ziemi +10, silnik −0.3/klatka |
| Zakończenie | Lądowanie, crash lub 1000 kroków |

### Seria 1: transfer 5 najlepszych z CartPole (300 000 kroków)

5 modeli: 3× `[64, 64]`, 1× `[16, 16]`, 1× `[1024, 1024, 1024]`

| # | Experiment ID | Architektura | `mean_reward` | `std_reward` | Czas [s] |
|---|---|---|---|---|---|
| 1 | `ll_005_s1024x1024x1024_batch_large` | `[1024, 1024, 1024]` | **223.47** | 21.86 | 224.61 |
| 2 | `ll_003_s64x64_lr_high` | `[64, 64]` | 113.60 | 113.43 | 43.87 |
| 3 | `ll_001_s64x64_batch_large` | `[64, 64]` | −45.37 | 77.70 | 40.78 |
| 4 | `ll_002_s64x64_ent_zero` | `[64, 64]` | −65.59 | 18.91 | 44.48 |
| 5 | `ll_004_s16x16_gamma_high` | `[16, 16]` | **−144.69** | 33.70 | 40.39 |

### Kluczowy wniosek — odwrotność CartPole

**Na LunarLander duża sieć `[1024, 1024, 1024]` rzeczywiście zaczęła dawać przewagę jakościową.** To pierwszy moment, w którym wysoki koszt architektury nie jest już tylko stratą czasu.

- Najlepszy model z CartPole (`[64, 64]`, batch 512) **nie przetransferował się** — wynik ujemny.
- `[16, 16]` zawiodła całkowicie (−144.69). Za mała pojemność na złożoną dynamikę.
- `[64, 64]` z wysokim LR dał dodatni wynik, ale bardzo niestabilny (std 113).
- Brak entropii (`ent_coef=0.0`) — stabilnie słaby. **Eksploracja na LunarLander jest ważniejsza niż na CartPole.**

### Seria 2: dogrywka na `[1024, 1024, 1024]`, 600 000 kroków, warianty hiperparametrów

10 treningów z OFAT na najskuteczniejszej sieci. **Popełniony błąd:** zdecydowanie zbyt duża sieć — `[64, 64]` byłoby wystarczające, ale pozwoliło zebrać dane porównawcze dla dużej architektury.

### Dostrzeżenie problemu OFAT

- OFAT nie uwzględnia interakcji między parametrami.
- Zmiana jednego parametru przy ustalonych pozostałych może nie pokazać optymalnej kombinacji.
- **Wniosek prowadzący do kolejnego etapu:** parametry powinny być modyfikowane wspólnie → optymalizacja bayesowska → Optuna.

---

## Etap 3 — Optuna, optymalizacja bayesowska

### Przeskok metodologiczny: OFAT → TPE

- **OFAT** (One Factor At a Time): zmiana jednego parametru, reszta stała. Nie wykrywa interakcji.
- **TPE** (Tree-structured Parzen Estimator): optymalizacja bayesowska, wykorzystuje historię prób do inteligentnych sugestii.
- **MedianPruner:** wczesne ucinanie słabych prób (early stopping) — oszczędność czasu.
- **Funkcja celu:** `objective_score = mean_reward − stability_penalty × std_reward` — premiuje nie tylko średni wynik, ale też stabilność polityki.

### Eksperyment 3a: LunarLander, pre5 — porównanie `[64, 64]` vs `[128, 128]`

- 40 triali, 300 000 kroków każdy
- Wynik: **20 completed, 20 pruned** (MedianPruner wyciął połowę)
- Top wyniki — wyłącznie architektura `[128, 128]`:

| # | Trial | Architektura | `objective_score` | `mean_reward` | `std_reward` | Czas [s] |
|---|---|---|---|---|---|---|
| 1 | `trial_012` | `[128, 128]` | **278.4** | 280.3 | 18.6 | 110 |
| 2 | `trial_013` | `[128, 128]` | 265.2 | 266.9 | 16.7 | 106 |
| 3 | `trial_010` | `[128, 128]` | 264.5 | 266.1 | 16.5 | 110 |
| 4 | `trial_011` | `[128, 128]` | 262.5 | 264.4 | 19.0 | 110 |
| 5 | `trial_023` | `[128, 128]` | 259.8 | 261.5 | 16.8 | 81 |

**Wniosek:** `[128, 128]` zdecydowanie dominuje nad `[64, 64]` na LunarLander z optymalizacją bayesowską. Lepsza nagroda **i** mniejsze odchylenie — bardziej stabilna polityka.

---

## Etap 4 — Humanoid-v5 (realizacja pozaplanowa)

### Opis gry

| Parametr | Wartość |
|---|---|
| Typ | MuJoCo (fizyka 3D) |
| Przestrzeń stanu | `Box(376,)` — pozycje i prędkości 17 stawów humanoidalnego robota |
| Przestrzeń akcji | `Box(17,)` — momenty sił na 17 stawach (ciągła!) |
| Nagroda | Złożona: nagroda za ruch do przodu, kary za energie, kontakty, upadek |
| Zakończenie | Upadek (niska pozycja torsu) lub limit kroków |
| Złożoność | **Najtrudniejsze standardowe środowisko MuJoCo** — 376-wymiarowa obserwacja, 17-wymiarowa ciągła akcja |

### Cel

Sprawdzenie, czy pipeline i metodologia projektu skalują się do realnie trudnego problemu RL. Humanoid jest środowiskiem **rzędów wielkości** trudniejszym niż CartPole czy LunarLander.

### Izolacja techniczna

- Osobny moduł: `src/humanoid_bayes.py`
- Osobny plik zależności: `requirements-humanoid.txt` (MuJoCo + Optuna)
- Osobny CSV wyników — zero ingerencji w bazowy pipeline

### Seria 1: Optuna, architektura `[256, 256]`, 40 triali × 1M kroków

- **8 completed, 28 pruned** (MedianPruner wyciął 70%)
- Najlepszy wynik: `trial_005` — `mean_reward=489.3`, `objective_score=478.1`

### Seria 2: Optuna, architektura `[512, 512]`, 51 triali × 1M kroków

- **8 completed, 42 pruned** (MedianPruner wyciął 84%)
- Najlepszy wynik: `trial_038` — `mean_reward=1441.8`, `objective_score=1381.3`
- Sieć `[512, 512]` daje **3× lepszy wynik** niż `[256, 256]` na Humanoidzie

| Architektura | Najlepszy `mean_reward` | Najlepszy `objective_score` |
|---|---|---|
| `[256, 256]` | 489.3 | 478.1 |
| `[512, 512]` | **1441.8** | **1381.3** |

### Baza danych Optuny

Wyniki optymalizacji bayesowskiej są przechowywane w plikach SQLite:
- `data/humanoid_optuna.db` — studium dla architektury `[256, 256]` (40 triali)
- Studium `[512, 512]` — 51 triali (osobna baza / CSV `data/humanoid_bayes_results_512x512.csv`)

Storage SQLite umożliwia **resume po przerwaniu** — `load_if_exists=True` w `create_study()`. Po awarii/przerwaniu procesu kolejne uruchomienie komendy wznawia studium dokładnie tam, gdzie zostało przerwane, bez utraty historii prób.

Top 5 z bazy danych (seria `[512, 512]`):

| # | Trial | `objective_score` | `mean_reward` | `std_reward` | Czas [s] |
|---|---|---|---|---|---|
| 1 | `trial_038` | **1381.3** | 1441.8 | 604.9 | 699 |
| 2 | `trial_033` | 871.9 | 894.1 | 221.8 | 425 |
| 3 | `trial_003` | 828.2 | 847.6 | 194.0 | 499 |
| 4 | `trial_005` | 717.4 | 734.7 | 173.1 | 407 |
| 5 | `trial_000` | 690.2 | 702.9 | 127.3 | 411 |

### Trening produkcyjny: 30 milionów kroków — pełna analiza

#### Wybrane hiperparametry (zwycięzca: `trial_038`, architektura `[512, 512]`)

| Parametr | Wartość | Co oznacza | Dlaczego ta wartość |
|---|---|---|---|
| `net_arch` | `[512, 512]` | Sieć MLP z 2 ukrytymi warstwami po 512 neuronów | Humanoid ma 376-wymiarowy stan — potrzebna duża pojemność reprezentacji. `[256, 256]` dawało 3× gorsze wyniki. |
| `learning_rate` | `2.446e-05` | Tempo aktualizacji wag — jak agresywnie optymalizator zmienia parametry sieci | Bardzo niski LR. Humanoid wymaga ostrożnych aktualizacji — agresywny LR destabilizuje politykę w środowisku z 17 ciągłymi akcjami. |
| `batch_size` | `512` | Ilość próbek w jednej aktualizacji gradientu | Duży batch wygładza gradient — kluczowe przy szumnym środowisku z dużą wariancją nagród. |
| `gamma` | `0.99` | Współczynnik dyskontowania — jak daleko agent patrzy w przyszłość | Wysoki gamma = długi horyzont planowania. Humanoid musi planować sekwencje ruchów na wiele kroków do przodu. |
| `n_steps` | `1024` | Ilość kroków w rollout buffer przed aktualizacją polityki | Mniejszy niż domyślny 2048 — częstsze aktualizacje kosztem mniejszego okna obserwacji. Kompromis znaleziony przez Optunę. |
| `ent_coef` | `1.434e-04` | Siła wymuszenia eksploracji (bonus entropijny w funkcji straty) | Bliski zeru — polityka może się specjalizować bez sztucznego wymuszania losowości. Optuna wybrała minimalną eksplorację. |
| `gae_lambda` | `0.95` | Parametr GAE (Generalized Advantage Estimation) — bias vs variance w estymacji advantage | Standardowa wartość. Lambda=0.95 daje dobry kompromis między dokładnością (lambda=1) a stabilnością (lambda=0). |
| `clip_range` | `0.2` | Zakres obcinania w funkcji celu PPO — jak dużo polityka może się zmienić w jednym update | Domyślna wartość SB3. Zbyt niski clip range = za wolna nauka, zbyt wysoki = niestabilność. |
| `target_kl` | `0.02683` | Próg dywergencji KL — jeśli aktualizacja za bardzo zmienia politykę, trening epoki jest przerywany | Kluczowy parametr ochronny. Zapobiega katastrofalnemu zapominaniu (catastrophic forgetting). |
| `n_epochs` | `15` | Ile razy ten sam batch danych jest używany do aktualizacji wag | Wysoki — 15 epok na jeden rollout. Ale `target_kl` chroni przed nadmierną zmianą polityki. |
| `vf_coef` | `0.75` | Waga value function loss w łącznej funkcji straty | Wysoki nacisk na dokładną estymację wartości stanu — ważne w środowisku z dużą wariancją nagród. |
| `normalize_advantage` | `False` | Czy normalizować estymaty advantage do średniej 0 i std 1 | Brak normalizacji — Optuna uznała, że surowe advantage dają lepsze gradienty w tym środowisku. |
| `VecNormalize` | `norm_obs=True, norm_reward=True` | Normalizacja obserwacji i nagród w czasie rzeczywistym (running mean/std) | **Kluczowe dla MuJoCo.** Bez tego 376-wymiarowe obserwacje mają zupełnie różne skale, a nagrody wahają się o rzędy wielkości. |

#### Mechanizm treningu produkcyjnego

- **Auto-resume:** checkpoint model + `VecNormalize` co 1M kroków (`SyncedCheckpointCallback`)
- Plik modelu: `models/humanoid_prod/latest_model.zip`
- Plik normalizacji: `models/humanoid_prod/latest_vecnormalize.pkl`
- 30 checkpointów zapisanych w trakcie treningu
- Po przerwaniu — restart komendy automatycznie wznawia od ostatniego checkpointu

#### Ewolucja metryk w trakcie 30M kroków

Dane z logu `logs/humanoid_production_30m_20260614_143329.log` (669 167 linii, 28 MB):

| Krok | `ep_rew_mean` | `ep_len_mean` | `action_std` | `entropy` | `explained_var` | `fps` | Czas |
|---|---|---|---|---|---|---|---|
| 0 (start) | 96 | 21.6 | 1.000 | −24.1 | −0.501 | 3 288 | 0s |
| 500k | 1 140 | 226 | 0.918 | −22.7 | 0.976 | 1 635 | 10 min |
| 1M | 2 460 | — | 0.886 | — | — | 1 596 | 17 min |
| 2M | 3 500 | 628 | 0.866 | −21.6 | 0.934 | 1 578 | 21 min |
| 4M | 4 530 | — | 0.779 | — | — | 1 677 | — |
| 6M | 5 030 | — | 0.709 | — | — | 1 733 | — |
| 8M | 4 510 | — | 0.642 | — | — | 1 769 | — |
| 10M | 5 670 | 741 | 0.602 | −14.9 | 0.906 | 1 802 | 1.5h |
| 14M | 6 250 | 765 | 0.527 | — | — | 1 879 | — |
| 18M | 6 450 | — | 0.503 | — | — | 1 977 | — |
| 20M | 6 490 | 772 | 0.499 | −10.2 | 0.950 | 2 008 | 2.8h |
| 26M | 6 900 | — | 0.483 | — | — | 2 073 | — |
| **30M (koniec)** | **6 140** | **711** | **0.484** | **−8.66** | **0.960** | **2 127** | **3.9h** |

#### Analiza ewolucji treningu

##### Faza 1: Zimny start (0 → 500k kroków)

- Agent zaczyna od losowej polityki: `ep_rew_mean=96`, epizody trwają ~22 kroki.
- Humanoid natychmiast upada — nie umie jeszcze balansować.
- `explained_variance` ujemna (−0.501) — value function nic nie przewiduje.
- `action_std=1.0` — pełna eksploracja, ruchy są losowe.

##### Faza 2: Szybka nauka (500k → 2M)

- **Najszybszy wzrost w całym treningu.** Reward rośnie z 1 140 do 3 500 (3× w 1.5M kroków).
- Humanoid uczy się chodzić — `ep_len` rośnie z 226 do 628.
- `action_std` spada z 0.918 do 0.866 — polityka zaczyna się specjalizować.
- `explained_variance` rośnie do 0.934 — value function zaczyna dobrze przewidywać zwroty.

##### Faza 3: Dojrzewanie (2M → 10M)

- Wzrost zwalnia: 3 500 → 5 670 (+62% w 8M kroków vs +207% w poprzednich 1.5M).
- `action_std` spada z 0.866 do 0.602 — polityka coraz bardziej deterministyczna.
- `entropy` spada z −21.6 do −14.9 — przestrzeń eksplorowanych zachowań się zawęża.
- Przejściowy spadek na 8M (4 510) — prawdopodobnie faza restrukturyzacji polityki.

##### Faza 4: Plateau z oscylacjami (10M → 30M)

- Reward stabilizuje się w zakresie 5 700 – 6 900.
- Peak: ~6 900 na 26M kroków.
- `action_std` stabilizuje się na ~0.484 — polityka jest w pełni wyspecjalizowana.
- `entropy` spada z −14.9 do −8.66 — agent "wie co robić", minimalna losowość.
- `explained_variance` utrzymuje się na 0.90–0.96 — value function bardzo dobrze modeluje środowisko.
- **Nie ma dalszego wzrostu** — sugeruje to, że 30M kroków to sensowny punkt końcowy dla tej konfiguracji.

#### Analiza early stoppingu (`target_kl`)

Parametr `target_kl=0.02683` wyzwolił **24 609 early stopów** w trakcie 30M kroków.

Early stopping przerywał aktualizację epoki, gdy dywergencja KL między starą a nową polityką przekroczyła próg. Przy `n_epochs=15` oznacza to, że w wielu iteracjach model nie wykonał pełnych 15 epok aktualizacji.

**Dystrybucja momentu przerwania (na której epoce z 15 nastąpił stop):**

| Epoka | Liczba wyzwoleń | Udział |
|---|---|---|
| 1–3 | 4 018 | 16.3% |
| 4–6 | 11 551 | 46.9% |
| 7–9 | 5 862 | 23.8% |
| 10–12 | 2 476 | 10.1% |
| 13–14 | 702 | 2.9% |

**Interpretacja:**

- W ~63% przypadków early stop nastąpił na epoce 4–9 — model potrzebował kilku epok zanim aktualizacja stała się zbyt agresywna.
- Pierwszy early stop pojawił się dopiero po ~1.4M kroków (linia 30794 logu) — wcześniej polityka zmieniała się na tyle wolno, że 15 epok nie przekraczało progu KL.
- Pod koniec treningu (>25M) early stop występuje wcześniej (epoka 2–5) — polityka jest już wyspecjalizowana i każda zmiana jest bardziej destrukcyjna.
- `target_kl` jest **kluczowym parametrem ochronnym**: bez niego 15 epok na każdym rollout mogłoby prowadzić do catastrophic forgetting.

#### Analiza FPS — dowód na brak thermal throttlingu

| Faza treningu | FPS | Trend |
|---|---|---|
| Start (0–10k kroków) | 3 288 → 2 200 | Spadek — stabilizacja po zimnym starcie |
| 500k | 1 635 | Stabilizacja (dłuższe epizody = więcej pracy na krok) |
| 2M | 1 578 | Minimum — najdłuższe epizody w proporcji do kosztu |
| 10M | 1 802 | **Wzrost** — early stopping skraca aktualizacje |
| 20M | 2 008 | **Dalszy wzrost** |
| 30M (koniec) | **2 127** | **Maksimum na końcu treningu** |

**Kluczowa obserwacja:** FPS **rośnie** w trakcie treningu, a nie spada. To oznacza:

1. **Zero thermal throttlingu.** Gdyby CPU dławił się, FPS spadałby w czasie.
2. Wzrost FPS wynika z early stoppingu — pod koniec treningu aktualizacje są przerywane wcześniej (epoka 2–5 zamiast 15), więc mniej obliczeń na iterację.
3. MacBook Air M4 utrzymał stabilną wydajność przez **3.9 godziny** ciągłego obciążenia CPU (sumaryczny czas samego treningu produkcyjnego, nie licząc Optuny).

#### Entropia i `action_std` — specjalizacja polityki

- `action_std` spadło z 1.000 (losowe ruchy) do 0.484 (precyzyjne sterowanie).
- `entropy_loss` spadło z −24.1 do −8.66.
- Polityka przeszła od chaotycznego eksplorowania do wyuczonego chodu humanoidalnego.
- Spadek entropii o ~64% oznacza, że agent „wie co robić" w większości stanów — nie marnuje akcji na losowe ruchy.
- `action_std` stabilizuje się na ~0.484 od 24M — oznacza konwergencję: dalsze zmniejszanie losowości już nie poprawia wyniku.

### 🔥 Kluczowa obserwacja sprzętowa

**MacBook Air M4 nawet się nie zaczął pocić.**

- **3.9 godziny** czystego treningu produkcyjnego (30M kroków)
- **~8 godzin łącznie** z Optuną (91 triali × 1M + 30M produkcja)
- Temperatura procesora **nie przekroczyła 85°C**
- **Thermal throttling, którego się obawialiśmy, NIGDY się nie pojawił**
- FPS **rósł** w trakcie treningu (1 635 → 2 127) — CPU przyspieszał, nie spowalniał
- Mechanizm cooldownu (`time.sleep()`) między eksperymentami Optuny okazał się wystarczający
- Apple Silicon M4 z 24 GB zunifikowanej pamięci sprawdził się jako platforma do treningu RL nawet na najtrudniejszym standardowym środowisku MuJoCo

To podważa potoczne przekonanie, że chłodzenie pasywne dyskwalifikuje laptopa do pracy z ML/RL.

### Wnioski z treningu produkcyjnego Humanoida

1. **Niski learning rate jest kluczowy** — `2.45e-05` to ~10× mniej niż domyślny SB3 (`3e-04`). Humanoid wymaga ostrożnych aktualizacji.
2. **`target_kl` jest niezbędny** — 24 609 early stopów zapobiegło destabilizacji polityki. Bez tego mechanizmu 15-epokowe aktualizacje mogłyby zniszczyć wyuczone zachowanie.
3. **`VecNormalize` jest obowiązkowy dla MuJoCo** — bez normalizacji obserwacji i nagród sieć nie konwerguje. Checkpoint musi zawierać zarówno model, jak i statystyki normalizacji.
4. **Punkt nasycenia: ~20M kroków** — po tym progu reward oscyluje wokół 6 000–6 900 bez wyraźnego trendu wzrostowego. Trening do 30M potwierdził stabilność, ale nie przyniósł przełomu.
5. **Minimalna entropia = wyspecjalizowana polityka** — `ent_coef=1.4e-04` pozwoliło agentowi szybko się specjalizować. Na Humanoidzie eksploracja jest mniej ważna niż precyzja sterowania.
6. **FPS jako dowód na brak throttlingu** — rosnący FPS w 4-godzinnym treningu jednoznacznie wyklucza thermal throttling.

---

## Wnioski końcowe — synteza 3 gier

### 1. Nie istnieje jedna architektura najlepsza zawsze

| Środowisko | Najlepsza architektura | Dlaczego |
|---|---|---|
| CartPole-v1 | `[64, 64]` | Proste środowisko — mała sieć wystarczy, duża to marnotrawstwo |
| LunarLander-v3 | `[128, 128]` | Pośrednia złożoność — wymaga więcej pojemności niż `[64, 64]` |
| Humanoid-v5 | `[512, 512]` | 376-wymiarowy stan, 17-wymiarowa ciągła akcja — duża sieć jest konieczna |

**Wniosek:** Złożoność środowiska determinuje optymalną pojemność sieci. Mapa jest prosta: proste środowisko → mała sieć, złożone → duża.

### 2. OFAT ma granice — optymalizacja bayesowska daje lepsze wyniki

- OFAT nie wykrywa interakcji między parametrami.
- Optuna z TPE + MedianPruner: inteligentne przeszukiwanie, wczesne odcinanie słabych prób.
- Na LunarLander: Optuna (pre5) dała `mean_reward=280.3` vs OFAT `mean_reward=223.5`.

### 3. Dobry wynik na prostym środowisku nie gwarantuje transferu

- Najlepszy model CartPole (`[64, 64]`, batch 512) dał **ujemny wynik** na LunarLander.
- Architektura `[16, 16]` — sukces na CartPole, katastrofa na LunarLander (−144).
- Każde środowisko wymaga osobnego strojenia.

### 4. Punkt nasycenia jakości

- CartPole osiągnął sufit (`500.0`) już przy 100k kroków. Dogrywka 300k nie poprawiła nic.
- Więcej nie znaczy lepiej — istnieje punkt, po którym trening jest czystym kosztem bez zysku.

### 5. Stabilność polityki ma znaczenie

- Na LunarLander i Humanoid samo `mean_reward` jest niewystarczające.
- Funkcja celu `objective_score = mean − penalty × std` premiuje stabilne polityki.
- Model z `mean=1441` ale `std=605` (Humanoid) jest mniej wiarygodny niż model z `mean=894` i `std=222`.

### 6. MacBook Air M4 jako platforma ML

- 8h treningu Humanoida bez thermal throttlingu (max ~85°C).
- Chłodzenie pasywne nie jest problemem przy rozsądnym cooldownie.
- Apple Silicon z 24GB zunifikowanej pamięci radzi sobie z MuJoCo, PPO i Optuną jednocześnie.

---

## Analiza logów TensorBoard — dane z eksperymentów

Dane ze 192 runów treningowych TensorBoard zostały pomyślnie wyeksportowane do płaskich plików CSV (`data/tensorboard_scalars_long.csv` to 18 MB objętości i ponad 275 tysięcy iteracji pomiarowych). Umożliwiło to wyciągnięcie obiektywnych, popartych liczbami konkluzji dla każdego etapu.

### Analiza 1: CartPole — wpływ architektury sieci

**Zbiór danych:** `exp_001` do `exp_033` (33 eksperymenty etapu 1 OFAT)

### Analiza 1: CartPole — wpływ architektury sieci

**Zbiór danych:** `exp_001` do `exp_033` (33 eksperymenty etapu 1 OFAT)

**1. Szybkość zbieżności vs rozmiar sieci (`rollout/ep_rew_mean`)**
- Architektura **`[16, 16]` nie wystarczyła**, by niezawodnie osiągnąć sufit środowiska (500 pkt). Jej najlepsze runy dochodziły do średniej rzędu ~450 (`batch_small`, `nsteps_small`), lecz większość zatrzymywała się na barierze 250–350 pkt.
- Architektura **`[64, 64]` osiągała sufit (500)** w okolicach 83k – 86k kroków. W tym przypadku pojemność reprezentacji umożliwiła idealne rozwiązanie środowiska przy standardowym horyzoncie epizodów.
- Architektura **`[1024, 1024, 1024]` uczyła się najszybciej pod kątem liczby kroków**, dochodząc do perfekcji już w 71k – 79k kroków, udowadniając przewagę pojemności pamięci w optymalizacji wstecznej.

**2. Wydajność obliczeniowa (`time/fps`)**
Dla tak prostego środowiska ekstremalnie duża sieć to absurd. Uśredniony koszt przetwarzania FPS (Frames Per Second) wyniósł:
- `[16, 16]`: **8144 FPS**
- `[64, 64]`: **7629 FPS**
- `[1024, 1024, 1024]`: **1237 FPS**

**Wniosek:** Choć monstrualnie wielka sieć oszczędza ~10% kroków treningowych by zbiec do maksimum prostego środowiska, czasowo robi to blisko 7× wolniej. Optymalnym balansem dla CartPole-v1 jest `[64, 64]`.

**3. Entropia polityki (`train/entropy_loss`)**
Spadek entropii od startu (ok. `-0.69`) jest niewielki (delta `0.11 – 0.17`) dla sieci małych i średnich, co dowodzi, że optymalne rozwiązanie CartPole wcale nie wymaga wąsko wyspecjalizowanej, sztywnej polityki. Gigantyczna sieć "wpycha" natomiast swoją reprezentację i w większości triali pikuje ze specjalizacją do `-0.36`, co na siłę determinuje agenta bez wyraźnej potrzeby.

**4. Szczegółowy wpływ hiperparametrów w wariantach OFAT**

- **Dla małej sieci `[16, 16]`** (baseline: 350 pkt):
  - **Przyspieszenie:** `batch_small` (+97 pkt, finał: 447) oraz `nsteps_small` (+100 pkt, finał: 450). Szum małego batcha oraz częstsze uaktualnianie gradientu pozwoliły skrajnie ubogiej sieci unikać minimów lokalnych i w bólach uczyć się z "małych porcji" wiedzy.
  - **Katastrofa:** `ent_zero` (-149 pkt) wymusił natychmiastowe zamrożenie eksploracji usztywniając złą strategię, uaktualnienia starymi danymi w `nsteps_large` (-137 pkt) zdezorientowały sieć, a przy `lr_low` (-99 pkt) sieć nie zdążyła się po prostu niczego nauczyć.

- **Dla optymalnej sieci `[64, 64]`** (baseline utknął na 328 pkt):
  - **Przełom (rozwiązanie zadania):** Zwycięskim parametrem dla tej architektury okazał się **`batch_small` (+172 pkt)**, osiągając limit 500 pkt środowiska jako najszybszy run etapu (w zaledwie 71k kroków). Pomógł również wyższy skok optymalizatora **`lr_high`** (osiągnął 500 w 79k kroków). Pojemność `64x64` znakomicie chłonęła uaktualnienia pod presją agresywniejszych parametrów.
  - **Regres:** Zaburzenie proporcji bufora rolloutów `nsteps_small` (-143 pkt) sprawiło, że gęste "szarpane" pakiety danych przy większej liczbie wag zachwiały zbieżnością, a skrajne zmiany dyskontowania grawitacji `gamma_low/high` (-65 pkt) wytrącały balans ułożonego agenta.

- **Dla gigantycznej sieci `[1024, 1024, 1024]`** (baseline osiągnął 500 pkt w 86k kroków):
  - **Odporność na błędy przez nadmiarowość:** Zmiana połowy hiperparametrów w tym `batch_large`, `nsteps_large`, `ent_zero`, a nawet `lr_high` nie zatrzymywała postępów — wszystkie dotarły do 500 punktów. Nadmiarowość architektury pozwoliła kompensować parametry samą pojemnością wag.
  - **Wyjątki, które zabiły ulep:** Nawet trylion neuronów zepsuje równowagę, jeśli na horyzoncie zastosujemy **`gamma_low`** (-215 pkt, zbiegło ledwie do 285). Ograniczenie przewidywania fizyki jest nieuleczalne przez samą sieć. Zaszkodziło jej również zaszumione aktualizowanie `nsteps_small` (-126 pkt).

### Analiza 2: LunarLander — eksperyment OFAT 300k kroków

**Zbiór danych:** 5 eksperymentów izolowanych `ll_001` do `ll_005`. 

| Eksperyment | Peak nagrody | Opis dynamiki i wnioski |
|---|---|---|
| `ll_005` (`[1024...]`) | **+215 @ 253k** | Zdecydowany zwycięzca w grupie, buduje konsekwentnie nagrodę od wartości skrajnie ujemnych (-140 na 50k) do szczytu w przedziale 250k kroków. |
| `ll_003` (`lr_high`) | +178 @ 200k | Uczy się ekstremalnie szybko (+178 w zaledwie 200k), lecz agresywny learning rate w 300k destabilizuje lot. |
| `ll_004` (`[16, 16]`) | +37 @ 284k | Ledwo remisuje z siłami grawitacji. Pojemność 16x16 okazuje się technicznie niezdolna do udźwignięcia nieliniowości nawigacji 2D z ciągłym napędem. |
| `ll_002` (`ent_coef=0.0`) | -10 @ 112k | Katastrofalna w skutkach wczesna konwergencja. Brak premii za eksplorację sprawia, że lądownik bardzo wcześnie usztywnia złą strategię, rozbijając się notorycznie bez poszukiwania lepszych wariantów sterowania. |

### Analiza 3: LunarLander — dogrywka 600k (duża sieć)

**Zbiór danych:** 11 wariantów wokół architektury `[1024, 1024, 1024]` dla podwojonego dystansu 600 tysięcy kroków. Horyzont 300k okazał się mocno zaniżony. 

| Parametr bazowy | Wynik na @100k | Na @300k | Na końcu @600k | Peak dogrywki (max) |
|---|---|---|---|---|
| `baseline` | -37 pkt | +90 pkt | +130 pkt | +134 pkt @ 356k |
| **`gamma_high`** | +23 pkt | +257 pkt | +260 pkt | **+278 pkt @ 491k** |
| `lr_high` | +31 pkt | +81 pkt | +232 pkt | +262 pkt @ 567k |
| `gamma_low` | -48 pkt | -116 pkt | -142 pkt | ujemny całą linię |

**Wniosek ogólny:** Środowisko z bezwładnością kosmiczną stawia kluczowy nacisk na długofalowe przewidywanie (związane ze współczynnikiem dyskontowania). Wariant `gamma_high` przyniósł absolutnie najlepszy wynik. Agent planujący z `gamma_low` "żyje w teraźniejszości" doprowadzając do ciągłych i powtarzalnych wypadków.

**Szczegółowa wrażliwość długoterminowa OFAT:**
Baseline osiągnął wynik pozytywny (+130 na końcu, z peak +134). Jak zmiana poszczególnych parametrów wpłynęła na uczenie przez 600k kroków?

- **Najwięksi wygrani (szybsza konwergencja na dużym plusie):**
  - `gamma_high`: Absolutny dominator na dystansie (peak **+278** / final **+260**, różnica do bazowego modelu: +144 pkt). Bezwładność nawigacji nagradza dalekie planowanie.
  - `lr_high`: Piekielnie szybki lot w górę (peak **+262**, różnica: +128 pkt). Przy relatywnie trudnym środowisku odważniejsze stawianie kroków gradientu pomogło uniknąć lokalnych dołków i przyspieszyło start.
  - `nsteps_large`: Lepsze, szerokie uśrednienie rolloutu dało modelowi czystszy estymator w dogrywce, ograniczając szum (peak **+233**, różnica: +99 pkt względem baseline).

- **Najwięksi przegrani (zepsucie lotu i regres polityki):**
  - `gamma_low`: Zjawisko opisane wyżej, niezdolność lądownika do przewidywania utraty wysokości permanentnie zrzuciła go na ziemię (-180 pkt różnicy, wynik w całości ujemny).
  - `batch_small`: Choć znakomicie sprawdził się wcześniej na prostym CartPole, szum i drgania w aktualizacjach na lądowniku wyrzuciły ułożony wektor z orbity, skutkując w dogrywce gorszym wynikiem od kontrolnego baseline'u (-9 pkt straty na końcu).
  - `ent_zero`: Agent nie uczy się odważnych podejść. Choć na początku dość szybko wstrzelił się na szczyt +243 (w ok. 300k kroków), w wyniku zabetonowania jednej strategii i ucięcia losowej eksploracji jego końcowy wynik zaczął dramatycznie pikować obsuwając się na końcu eksperymentu na +176. Eksploracja pod koniec treningu bywa kluczowa by poprawić wady wektora wag.

### Analiza 4: Humanoid Optuna — fizyczne zatorowanie pojemności

**Zbiór danych:** Najlepsze warianty bayesowskie z Optuny dla `[256, 256]` (trial 005) vs `[512, 512]` (trial 038) w pierwszych 1M kroków treningu MuJoCo.

Śledzenie `rollout/ep_rew_mean` rozbija tezę, o skuteczności średnich architektur MLP w złożonych środowiskach kontroli motorycznej.

| Architektura | Start @100k | Mileston @500k | Zakończenie @1M | Opis dynamiki progresji |
|---|---|---|---|---|
| `[256, 256]` | 243 pkt | 384 pkt | 409 pkt | Sieć uderza w **sufit możliwości swojej reprezentacji**. Nasyca się przy 500k i już nie ewoluuje zgrabnie, a przyrost od 500k do 1M wynosi żenujące +25 punktów. |
| `[512, 512]` | **349 pkt** | **545 pkt** | **983 pkt** | Dwukrotnie wyższa wyporność sieci **utrzymuje wysoki pionowy gradient uczenia** od startu pod sam koniec. W rezultacie przy 1 mln iteracji osiąga blisko mnożnik x2.4 nagrody mniejszej siostry. |

---

## Demo live

### Przygotowane modele do pokazania

| Gra | Model | Komenda |
|---|---|---|
| CartPole-v1 | `exp_016_s64x64_batch_large_stage1` | `bash scripts/evaluate_cartpole.sh` |
| LunarLander-v3 | `ll_012_s1024x1024x1024_gamma_high_tune600k` | `bash scripts/evaluate_lunarlander.sh` |
| Humanoid-v5 | `models/humanoid_prod/latest_model.zip` | `bash scripts/evaluate_humanoid_production.sh` |

### Backup

Jeśli demo live nie zadziała: nagranie screencast z ewaluacji.

---

## Chronologia projektu (z changelogu)

| Wersja | Data | Co się wydarzyło |
|---|---|---|
| `0.1.0` | 2026-05-09 | Struktura katalogów, `requirements.txt`, `pyproject.toml`, macierz 13 eksperymentów, CI/CD, `run_checks.sh` |
| `0.2.0` | 2026-05-09 | `config.py`, `training.py`, testy 100% coverage |
| `0.3.0` | 2026-06-13 | `evaluate.py` z CLI i renderowaniem |
| `0.3.1` | 2026-06-13 | Rozbudowa macierzy do 33 eksperymentów OFAT |
| `0.4.0` | 2026-06-13 | CLI treningu `python -m src.training`, `Monitor` wrapper |
| `0.4.1` | 2026-06-13 | Analiza porównawcza CartPole etap 1 vs 2 |
| `0.5.0` | 2026-06-13 | LunarLander: 5 treningów, zależności Box2D |
| `0.5.1` | 2026-06-13 | Analiza LunarLander — rekomendacja modelu |
| `0.5.2` | 2026-06-13 | Dogrywka LunarLander na `[1024,1024,1024]` (600k) |
| `0.6.0` | 2026-06-13 | Humanoid: moduł bayesowski, izolacja MuJoCo |
| `0.6.1` | 2026-06-13 | Rozszerzony search space Humanoida + scoring stabilnościowy |
| `0.6.2` | 2026-06-13 | Resume Optuny (SQLite storage) |
| `0.6.3` | 2026-06-13 | LunarLander pre5: porównanie `[64,64]` vs `[128,128]` |
| `0.6.4` | 2026-06-14 | Humanoid produkcyjny 30M, eksport TB do CSV, ewaluacja |

---

## Dokumentacja pracy z AI

Cała komunikacja z modelami AI (Gemini, Claude) wyeksportowana do repozytorium. System agentowy:

| Agent | Rola |
|---|---|
| Architekt | Product Owner — planowanie, delegowanie, wetowanie |
| Developer | Implementacja kodu Python |
| QA | Testy, egzekwowanie 100% coverage |
| Gatekeeper | Pipeline CI/CD |