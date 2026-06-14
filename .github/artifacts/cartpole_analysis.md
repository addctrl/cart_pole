# Analiza CartPole — Etap 1 i Etap 2

> Data analizy: 2026-06-13
> Zakres: porównanie 33 eksperymentów etapu 1 (100000 kroków) oraz 10 eksperymentów etapu 2 (300000 kroków)

## 1. Cel analizy

Celem analizy jest odpowiedź na trzy pytania:

1. Które konfiguracje PPO dały najlepszy wynik na CartPole-v1?
2. Czy dogrywka finalistów do 300000 kroków wniosła realną poprawę jakości?
3. Które modele warto zabrać jako punkt startowy do LunarLander-v2?

## 2. Metodologia porównania

Ranking modeli został zbudowany według kolejności:

1. Najwyższy `mean_reward`
2. Najniższy `std_reward`
3. Najkrótszy `training_time_s`

To oznacza, że jeśli dwa modele osiągnęły `500.0` oraz `0.0`, lepszy jest ten, który zrobił to szybciej.

## 3. Wyniki zbiorcze

### Etap 1

- Liczba eksperymentów: 33
- Liczba pełnych rozwiązań (`mean_reward=500`, `std_reward=0`): 17
- Średni `mean_reward`: 451.15
- Średni `training_time_s`: 39.92

### Etap 2

- Liczba eksperymentów: 10
- Liczba pełnych rozwiązań (`mean_reward=500`, `std_reward=0`): 10
- Średni `mean_reward`: 500.0
- Średni `training_time_s`: 83.55

## 4. Najlepsze modele ogółem

| Miejsce | Experiment ID | Architektura | Hiperparametr wyróżniający | `mean_reward` | `std_reward` | `training_time_s` |
|---|---|---|---|---|---|---|
| 1 | `exp_007_s16x16_gamma_high_stage1` | `[16, 16]` | `gamma=0.999` | 500.0 | 0.0 | 12.10 |
| 2 | `exp_016_s64x64_batch_large_stage1` | `[64, 64]` | `batch_size=512` | 500.0 | 0.0 | 12.24 |
| 3 | `exp_021_s64x64_ent_zero_stage1` | `[64, 64]` | `ent_coef=0.0` | 500.0 | 0.0 | 12.60 |
| 4 | `exp_014_s64x64_lr_high_stage1` | `[64, 64]` | `learning_rate=0.001` | 500.0 | 0.0 | 12.79 |
| 5 | `exp_022_s64x64_ent_high_stage1` | `[64, 64]` | `ent_coef=0.05` | 500.0 | 0.0 | 12.95 |
| 6 | `exp_020_s64x64_nsteps_large_stage1` | `[64, 64]` | `n_steps=4096` | 500.0 | 0.0 | 13.03 |
| 7 | `exp_012_s64x64_baseline_stage1` | `[64, 64]` | baseline | 500.0 | 0.0 | 13.29 |
| 8 | `exp_015_s64x64_batch_small_stage1` | `[64, 64]` | `batch_size=64` | 500.0 | 0.0 | 21.44 |
| 9 | `exp_034_s16x16_gamma_high_stage2` | `[16, 16]` | etap 2 dla top-1 | 500.0 | 0.0 | 36.78 |
| 10 | `exp_035_s64x64_batch_large_stage2` | `[64, 64]` | etap 2 dla top-2 | 500.0 | 0.0 | 37.56 |

## 5. Wnioski z etapu 1

### 5.1 Architektura `[16, 16]`

- Tylko 1 z 11 eksperymentów osiągnął pełne rozwiązanie z `500.0` i `0.0`.
- Średni wynik tej architektury to 394.36.
- To oznacza, że mała sieć jest za słaba jako ogólny baseline, ale może trafić w dobre ustawienie przy odpowiednim `gamma`.
- Najlepszy przypadek to `exp_007_s16x16_gamma_high_stage1`, czyli bardzo długi horyzont planowania (`gamma=0.999`).

Interpretacja:
Mała sieć może wystarczyć do prostego środowiska, ale jest dużo bardziej wrażliwa na dobór hiperparametrów. Daje szybkie treningi, lecz nie daje tak stabilnego marginesu bezpieczeństwa jak `[64, 64]`.

### 5.2 Architektura `[64, 64]`

- 7 z 11 eksperymentów osiągnęło pełne rozwiązanie z `500.0` i `0.0`.
- Średni wynik to 466.82.
- Średni czas treningu to 13.63 s.
- To jest najlepszy kompromis między jakością i kosztem obliczeniowym.

Interpretacja:
To jest faktyczny sweet spot dla CartPole-v1. Sieć nie jest ani za mała, ani przewymiarowana. Różne warianty hiperparametrów w ramach `[64, 64]` rozwiązywały zadanie stabilnie i szybko.

### 5.3 Architektura `[1024, 1024, 1024]`

- 9 z 11 eksperymentów osiągnęło pełne rozwiązanie z `500.0` i `0.0`.
- Średni wynik to 492.26.
- Średni czas treningu to 93.55 s.

Interpretacja:
Duża sieć działa, ale koszt jest nieproporcjonalny do zysku. To jest klasyczny przykład naruszenia YAGNI. Model rozwiązuje CartPole, ale nie robi tego lepiej od `[64, 64]`; robi to jedynie znacznie drożej.

## 6. Wnioski z etapu 2

Etap 2 polegał na dograniu 10 finalistów z 100000 do 300000 kroków.

### Najważniejsza obserwacja

Wszystkie 10 modeli z etapu 2 skończyły z wynikiem:

- `mean_reward = 500.0`
- `std_reward = 0.0`

To oznacza, że drugi etap nie poprawił już jakości w metryce końcowej CSV. Poprawa nie była możliwa do zaobserwowania, bo CartPole ma naturalny sufit jakości na poziomie 500 kroków.

### Koszt dodatkowych 200000 kroków

Dla finalistów czas wzrósł mniej więcej trzykrotnie:

| Etap 1 | Etap 2 | Czas etap 1 | Czas etap 2 | Różnica | Mnożnik |
|---|---|---|---|---|---|
| `exp_007_s16x16_gamma_high_stage1` | `exp_034_s16x16_gamma_high_stage2` | 12.10 | 36.78 | +24.68 | 3.04x |
| `exp_016_s64x64_batch_large_stage1` | `exp_035_s64x64_batch_large_stage2` | 12.24 | 37.56 | +25.32 | 3.07x |
| `exp_021_s64x64_ent_zero_stage1` | `exp_036_s64x64_ent_zero_stage2` | 12.60 | 40.43 | +27.83 | 3.21x |
| `exp_014_s64x64_lr_high_stage1` | `exp_037_s64x64_lr_high_stage2` | 12.79 | 40.68 | +27.89 | 3.18x |
| `exp_022_s64x64_ent_high_stage1` | `exp_038_s64x64_ent_high_stage2` | 12.95 | 39.43 | +26.48 | 3.04x |
| `exp_020_s64x64_nsteps_large_stage1` | `exp_039_s64x64_nsteps_large_stage2` | 13.03 | 40.35 | +27.32 | 3.10x |
| `exp_012_s64x64_baseline_stage1` | `exp_040_s64x64_baseline_stage2` | 13.29 | 40.34 | +27.05 | 3.04x |
| `exp_015_s64x64_batch_small_stage1` | `exp_041_s64x64_batch_small_stage2` | 21.44 | 63.29 | +41.85 | 2.95x |
| `exp_027_s1024x1024x1024_batch_large_stage1` | `exp_042_s1024x1024x1024_batch_large_stage2` | 71.62 | 228.81 | +157.19 | 3.19x |
| `exp_033_s1024x1024x1024_ent_high_stage1` | `exp_043_s1024x1024x1024_ent_high_stage2` | 83.26 | 267.79 | +184.53 | 3.22x |

Interpretacja:
Etap 2 potwierdził stabilność finalistów, ale nie poprawił jakości końcowej na CartPole. Z punktu widzenia analityki był to etap potwierdzający, a nie odkrywczy.

## 7. Co widać i czego szukać w TensorBoard

### `rollout/ep_rew_mean`

To najważniejszy wykres. Pokazuje średnią nagrodę w czasie.

Na co patrzeć:

- Jak szybko krzywa dochodzi do 500.
- Czy wzrost jest płynny, czy poszarpany.
- Czy po dojściu do wysokich wartości nie pojawiają się spadki.

Interpretacja dla tego projektu:

- Dobre modele powinny dojść do sufitu 500 szybko i bez dużych wahań.
- Modele z etapu 2 powinny po pewnym czasie po prostu dłużej utrzymywać plateau, a nie bić nowe rekordy.

### `rollout/ep_len_mean`

Na CartPole długość epizodu jest w praktyce odpowiednikiem jakości.

- Wzrost do 500 oznacza, że agent utrzymuje drążek do limitu środowiska.
- Jeśli `ep_len_mean` i `ep_rew_mean` rosną razem, nauka jest spójna.

### `train/loss`, `train/value_loss`, `train/policy_gradient_loss`

Te wykresy służą do diagnostyki procesu uczenia, nie do wyboru zwycięzcy samego w sobie.

Na co patrzeć:

- Czy nie ma eksplozji wartości.
- Czy przebieg jest stabilny po osiągnięciu wysokich rewardów.
- Czy bardziej agresywne ustawienia (`learning_rate=0.001`) nie dają niestabilnych skoków na początku.

### `train/entropy_loss`

Pomaga ocenić wpływ `ent_coef`.

Interpretacja:

- Wyższe `ent_coef` powinno dłużej utrzymywać eksplorację.
- Na CartPole widać, że zarówno `ent_coef=0.0`, jak i `0.05` potrafiły dojść do optimum w architekturze `[64, 64]`.
- To sugeruje, że środowisko jest na tyle proste, iż eksploracja nie jest głównym wąskim gardłem.

### `time/fps`

To metryka kosztu obliczeniowego.

Interpretacja:

- Modele `[1024, 1024, 1024]` powinny mieć wyraźnie gorszy koszt na jednostkę treningu.
- To ważny argument prezentacyjny: większa sieć nie daje lepszego wyniku, ale odbiera czas.

## 8. Najważniejsze wnioski projektowe

1. CartPole-v1 zostało nasycone już w etapie 1. Dla wielu konfiguracji metryka końcowa osiągnęła sufit środowiska.
2. Drugi etap nie pokazał poprawy jakości, tylko wzrost kosztu. To znaczy, że dla tego środowiska 300000 kroków nie daje istotnej wartości dodanej dla finalistów.
3. Architektura `[64, 64]` jest najlepszym kompromisem jakości, stabilności i czasu treningu.
4. Architektura `[16, 16]` potrafi wygrać czasowo w jednym wariancie, ale jest zbyt krucha jako wybór domyślny.
5. Architektura `[1024, 1024, 1024]` potwierdza tezę YAGNI: działa, ale jest obliczeniowo nieuzasadniona.

## 9. Rekomendacje do LunarLander-v2

### Rekomendowany model startowy

`exp_016_s64x64_batch_large_stage1`

Uzasadnienie:

- Pełne rozwiązanie (`500.0`, `0.0`).
- Bardzo krótki czas treningu jak na model stabilny.
- Architektura `[64, 64]` jest rozsądnym kompromisem pojemności i kosztu.
- `batch_size=512` sugeruje bardziej stabilną aktualizację gradientu bez wejścia w koszt dużej sieci.

### Dwa sensowne modele zapasowe

- `exp_021_s64x64_ent_zero_stage1`
- `exp_014_s64x64_lr_high_stage1`

Uzasadnienie:

- Oba rozwiązują CartPole perfekcyjnie.
- Reprezentują inne zachowanie optymalizacyjne niż baseline i mogą inaczej zachowywać się w bardziej złożonym środowisku.

### Czego nie brać jako pierwszego wyboru

- `[1024, 1024, 1024]` — za drogie obliczeniowo.
- `[16, 16]` — zbyt ryzykowne jako architektura wyjściowa do trudniejszego problemu.

## 10. Konkluzja końcowa

Najsilniejszy wniosek analityczny jest prosty:

**Dla CartPole większa sieć i dłuższy trening nie dały lepszego modelu niż dobrze dobrane warianty `[64, 64]`.**

Projekt potwierdził trzy rzeczy:

1. istnieje punkt nasycenia jakości,
2. koszt obliczeniowy może rosnąć bez zysku jakościowego,
3. najlepszy model do dalszej pracy nie jest najbardziej rozbudowanym modelem, tylko najbardziej opłacalnym kompromisem.
