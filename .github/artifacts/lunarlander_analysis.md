# Analiza LunarLander — seria 5 treningów

> Data analizy: 2026-06-13
> Zakres: 5 treningów PPO na `LunarLander-v3`, każdy po `300000` kroków

## 1. Cel analizy

Celem tej serii było sprawdzenie, które ustawienia wyłonione po CartPole najlepiej transferują się do trudniejszego środowiska z bogatszą funkcją nagrody i bardziej złożoną dynamiką sterowania.

Badanie miało odpowiedzieć na trzy pytania:

1. Czy zwycięzca z CartPole pozostaje zwycięzcą na LunarLander?
2. Czy średnia architektura `[64, 64]` nadal daje najlepszy kompromis?
3. Czy duża sieć `[1024, 1024, 1024]` zaczyna wreszcie uzasadniać swój koszt?

## 2. Ranking końcowy

Ranking jest liczony według:

1. najwyższy `mean_reward`,
2. niższy `std_reward`,
3. krótszy `training_time_s`.

| Miejsce | Experiment ID | Architektura | Parametry | `mean_reward` | `std_reward` | `training_time_s` |
|---|---|---|---|---|---|---|
| 1 | `ll_005_s1024x1024x1024_batch_large_compare` | `[1024, 1024, 1024]` | `lr=0.0003`, `batch=512`, `gamma=0.99`, `n_steps=2048`, `ent=0.01` | 223.47 | 21.86 | 224.61 |
| 2 | `ll_003_s64x64_lr_high_alt` | `[64, 64]` | `lr=0.001`, `batch=256`, `gamma=0.99`, `n_steps=2048`, `ent=0.01` | 113.60 | 113.43 | 43.87 |
| 3 | `ll_001_s64x64_batch_large_primary` | `[64, 64]` | `lr=0.0003`, `batch=512`, `gamma=0.99`, `n_steps=2048`, `ent=0.01` | -45.37 | 77.70 | 40.78 |
| 4 | `ll_002_s64x64_ent_zero_alt` | `[64, 64]` | `lr=0.0003`, `batch=256`, `gamma=0.99`, `n_steps=2048`, `ent=0.0` | -65.59 | 18.91 | 44.48 |
| 5 | `ll_004_s16x16_gamma_high_compare` | `[16, 16]` | `lr=0.0003`, `batch=256`, `gamma=0.999`, `n_steps=2048`, `ent=0.01` | -144.69 | 33.70 | 40.39 |

## 3. Kluczowy wniosek

Najmocniejsza obserwacja jest odwrotna niż na CartPole:

**Na LunarLander duża sieć `[1024, 1024, 1024]` rzeczywiście zaczęła dawać przewagę jakościową.**

To pierwszy moment w projekcie, w którym wysoki koszt architektury nie jest już tylko stratą czasu. Model `ll_005_s1024x1024x1024_batch_large_compare` osiągnął dodatni i wyraźnie najwyższy `mean_reward`, podczas gdy mniejsze architektury były znacznie słabsze lub wręcz ujemne.

## 4. Interpretacja wyników per model

### 4.1 `ll_005_s1024x1024x1024_batch_large_compare`

- Najlepszy wynik końcowy: `223.47`
- Niski `std_reward`: `21.86`
- Bardzo wysoki koszt: `224.61 s`

Interpretacja:

To jedyny model, który wygląda jak realny kandydat do dalszego rozwijania. Wynik dodatni i relatywnie małe odchylenie oznaczają, że polityka zaczęła uczyć się sensownego sterowania, a nie tylko losowego „przeżywania” epizodów.

Wniosek:

LunarLander jest środowiskiem wystarczająco złożonym, aby większa pojemność modelu miała realną wartość.

### 4.2 `ll_003_s64x64_lr_high_alt`

- Drugi wynik: `113.60`
- Bardzo wysokie odchylenie: `113.43`
- Niski koszt: `43.87 s`

Interpretacja:

To model z wyższym `learning_rate`, który uczy się częściowo, ale bardzo niestabilnie. Prawdopodobnie ma epizody bardzo dobre przeplatane bardzo słabymi. To nie jest jeszcze model produkcyjnie stabilny, ale jest dobrym sygnałem, że `[64, 64]` nie jest całkowicie skreślone.

Wniosek:

Wyższy `learning_rate` pomógł średniej sieci ruszyć z miejsca, ale nie dał stabilności.

### 4.3 `ll_001_s64x64_batch_large_primary`

- Wynik ujemny: `-45.37`
- Duże odchylenie: `77.70`

Interpretacja:

To ważny sygnał analityczny: najlepszy kandydat z CartPole nie przetransferował się dobrze do LunarLandera. Środowisko jest bardziej złożone, ma inną funkcję nagrody i wymaga subtelniejszej polityki sterowania niż balansowanie drążka.

Wniosek:

Dobry wynik na CartPole nie gwarantuje dobrej generalizacji do trudniejszego środowiska.

### 4.4 `ll_002_s64x64_ent_zero_alt`

- Wynik ujemny: `-65.59`
- Najniższe odchylenie wśród słabych modeli: `18.91`

Interpretacja:

Model jest stabilny, ale stabilnie słaby. Brak dodatkowej entropii ograniczył eksplorację za mocno jak na środowisko, które wymaga szukania bardziej złożonych strategii lądowania.

Wniosek:

Na LunarLander eksploracja jest ważniejsza niż na CartPole.

### 4.5 `ll_004_s16x16_gamma_high_compare`

- Najgorszy wynik: `-144.69`
- Średnie odchylenie: `33.70`

Interpretacja:

Mała sieć nie ma wystarczającej pojemności reprezentacyjnej do opanowania złożoności LunarLandera. To wynik zgodny z intuicją: to, co jeszcze dawało pojedyncze sukcesy na CartPole, tutaj już nie wystarcza.

Wniosek:

Architektura `[16, 16]` powinna zostać odrzucona dla drugiej gry.

## 5. Porównanie z CartPole

### CartPole

- środowisko proste,
- szybkie dojście do sufitu jakości `500`,
- duża sieć była kosztowna i zbędna,
- `[64, 64]` dawało najlepszy kompromis.

### LunarLander

- środowisko dużo trudniejsze,
- brak sufitu osiągniętego przez wszystkie modele,
- duża sieć dała wyraźnie najlepszy wynik,
- mała sieć zawiodła całkowicie.

### Wniosek przekrojowy

**Nie istnieje jedna architektura najlepsza zawsze.**

Na prostym środowisku większa sieć to marnowanie zasobów. Na złożonym środowisku może stać się uzasadniona, bo zyskuje zdolność reprezentacji bardziej wymagającej polityki.

## 6. Jak czytać TensorBoard dla LunarLandera

### `rollout/ep_rew_mean`

To nadal najważniejszy wykres, ale tutaj interpretacja jest trudniejsza niż w CartPole.

Na co patrzeć:

- czy krzywa wychodzi z głębokich wartości ujemnych,
- czy wzrost jest stopniowy czy skokowy,
- czy model utrzymuje dodatnie rewardy po ich osiągnięciu.

Interpretacja:

- `ll_005` powinien mieć najwyższy i najbardziej spójny trend wzrostowy,
- `ll_003` prawdopodobnie pokaże wyraźne oscylacje,
- modele słabsze mogą mieć płaskie lub niestabilne krzywe wokół zera albo poniżej zera.

### `train/loss`, `train/value_loss`, `train/policy_gradient_loss`

Tutaj szukaj przede wszystkim niestabilności:

- duże skoki,
- brak wygaszania,
- bardzo agresywne zmiany przy `learning_rate=0.001`.

W `ll_003` warto sprawdzić, czy wyższy wynik nie jest okupiony bardzo chaotycznym przebiegiem strat.

### `train/entropy_loss`

Najważniejsze przy porównaniu `ll_001` vs `ll_002`.

Interpretacja:

- jeśli `ent_coef=0.0` utrudniał eksplorację, powinieneś zobaczyć szybsze usztywnienie polityki,
- jeśli `ent_coef=0.01` pozwalał na lepsze badanie przestrzeni akcji, krzywa rewardu powinna mieć więcej okazji do poprawy.

### `time/fps`

Tutaj widać koszt złożoności architektury.

`ll_005` jest najlepszy jakościowo, ale bardzo drogi. To jest główny trade-off do pokazania w prezentacji.

## 7. Wnioski końcowe

1. Najlepszym modelem tej serii jest `ll_005_s1024x1024x1024_batch_large_compare`.
2. Najlepszy model z CartPole nie utrzymał przewagi po transferze do LunarLandera.
3. Średnia architektura `[64, 64]` nadal ma sens, ale wymaga dalszego strojenia, bo obecnie tylko wariant z wysokim `learning_rate` dał dodatni wynik.
4. Mała architektura `[16, 16]` nie nadaje się do tego środowiska.
5. Dla LunarLandera większa pojemność modelu może być uzasadniona, mimo wysokiego kosztu obliczeniowego.

## 8. Rekomendacja do dalszej pracy

### Model główny do dalszej ewaluacji

`ll_005_s1024x1024x1024_batch_large_compare`

### Model zapasowy do obserwacji w TensorBoard

`ll_003_s64x64_lr_high_alt`

Uzasadnienie:

- `ll_005` daje najlepszy realny wynik jakościowy,
- `ll_003` jest najciekawszym lekkim wariantem, choć niestabilnym,
- pozostałe modele nie pokazują obecnie wystarczającej jakości, aby warto było na nich budować demo.
