# PRD — Dokument Wymagań Produktu

| Pole | Wartość |
|---|---|
| **Nazwa projektu** | Optymalizacja hiperparametrów i architektury sieci w algorytmach RL |
| **Kontekst** | Projekt zaliczeniowy — Sztuczna Inteligencja |
| **Wykładowca** | Dr Maciej Kraszewski |
| **Termin oddania** | Koniec czerwca 2026 |
| **Zasób sprzętowy** | MacBook Air M4 (24 GB RAM, zunifikowana pamięć, chłodzenie pasywne) |
| **Wersja dokumentu** | 1.0.0 |
| **Data utworzenia** | 2026-05-09 |

---

## 1. Wizja i cel produktu

Celem projektu nie jest stworzenie idealnego agenta AI. Celem jest **udowodnienie analitycznego podejścia inżyniera do procesu uczenia maszynowego**.

Projekt wykazuje, poprzez twarde dane z logów, jak zmiana pojemności sieci neuronowej (zasada YAGNI) oraz manipulacja hiperparametrami algorytmu PPO wpływają na:
- **Zbieżność** — czy agent uczy się rozwiązywać zadanie.
- **Stabilność** — czy nagroda nie oscyluje nadmiernie.
- **Czas treningu** — koszt obliczeniowy w kontekście ograniczonego sprzętu.

Produktem końcowym jest:
1. Zautomatyzowany pipeline treningowy sterowany plikiem CSV.
2. Wykresy analityczne z TensorBoard.
3. Demo na żywo wytrenowanego agenta.

---

## 2. Zakres funkcjonalny

Projekt składa się z dwóch niezależnych środowisk obsługiwanych przez ten sam zunifikowany kod:

### 2.1 CartPole-v1 (Baza analityczna)

| Parametr | Wartość |
|---|---|
| Typ | Classic Control |
| Przestrzeń stanu | `Box(4,)` — pozycja wózka, prędkość wózka, kąt drążka, prędkość kątowa |
| Przestrzeń akcji | `Discrete(2)` — lewo/prawo |
| Nagroda | +1 za każdy krok utrzymania drążka |
| Zakończenie | Drążek > 12°, wózek poza granicami, lub 500 kroków |
| Cel w projekcie | Szybkie przemielenie macierzy eksperymentów i generacja wykresów |

### 2.2 LunarLander-v2 (Demo docelowe)

| Parametr | Wartość |
|---|---|
| Typ | Box2D |
| Przestrzeń stanu | `Box(8,)` — pozycja, prędkość, kąt, kontakt nóg |
| Przestrzeń akcji | `Discrete(4)` — nic, lewy silnik, główny silnik, prawy silnik |
| Nagroda | Złożona: lądowanie +100..+140, crash -100, noga na ziemi +10, silnik -0.3/klatka |
| Zakończenie | Lądowanie, crash lub 1000 kroków |
| Cel w projekcie | Weryfikacja najlepszych parametrów na żywo |

---

## 3. Zakres badawczy

System przyjmuje plik konfiguracyjny CSV. Dla każdego wiersza wykonuje trening i loguje wyniki.

### 3.1 Zmienna główna — Architektura sieci MLP

| Wariant | Architektura | Cel badawczy |
|---|---|---|
| Za mała | `[16, 16]` | Sprawdzenie minimalnej pojemności informacyjnej |
| Optymalna | `[64, 64]` | Baseline dla złożoności wektora stanu |
| Zbyt duża | `[1024, 1024, 1024]` | Wykazanie spadku wydajności i ryzyka overfittingu |

### 3.2 Zmienne poboczne — Hiperparametry PPO

| Hiperparametr | Opis | Warianty (3 poziomy) |
|---|---|---|
| `learning_rate` | Współczynnik uczenia — agresywność optymalizatora | niski, bazowy, wysoki |
| `batch_size` | Wielkość partii — stabilność gradientu, obciążenie RAM | mały, bazowy, duży |
| `gamma` | Współczynnik dyskontowania — horyzont planowania | krótki, bazowy, długi |
| `n_steps` | Rozmiar bufora — kroki przed aktualizacją wag | mały, bazowy, duży |
| `ent_coef` | Współczynnik entropii — wymuszanie eksploracji | brak, bazowy, wysoki |

Konkretne wartości liczbowe definiowane są w pliku `data/experiments.csv`.

---

## 4. Wymagania techniczne

### 4.1 Stack technologiczny

| Komponent | Narzędzie |
|---|---|
| Algorytm RL | PPO (`stable-baselines3`) |
| Środowiska | Gymnasium (`gymnasium`) |
| Język | Python 3.12 |
| Linter/Formatter | Ruff |
| Typowanie statyczne | Mypy (strict) |
| Testy | Pytest + pytest-cov |
| Dokumentacja | pdoc (NumPy Style Docstrings) |
| Analityka | TensorBoard |
| CI/CD | GitHub Actions |
| Zarządzanie danymi | CSV (bez baz danych) |

### 4.2 Ograniczenia sprzętowe

1. **MacBook Air M4** — jedyny zasób obliczeniowy.
2. **24 GB RAM zunifikowanej pamięci** — limituje `batch_size` i `n_steps`.
3. **Chłodzenie pasywne** — wymusza mechanizm `time.sleep()` między eksperymentami.
4. **Brak GPU CUDA** — sb3 pracuje na CPU (`device="auto"` → `cpu`).

### 4.3 Ochrona sprzętu

Pętla treningowa **musi** implementować wymuszony cooldown:
- Domyślny cooldown: **60 sekund** między eksperymentami.
- Cooldown dla architektury `[1024, 1024, 1024]`: **120 sekund**.
- Mechanizm: `time.sleep()` w pętli iterującej po wierszach CSV.

### 4.4 Zarządzanie danymi

- Parametry wejściowe: plik `data/experiments.csv`.
- Po zakończeniu treningu: skrypt **dopisuje** metryki wynikowe (`ep_rew_mean`, `loss`, czas treningu) do tego samego pliku CSV.
- Logi TensorBoard: katalog `logs/tensorboard/<nazwa_eksperymentu>/`.
- Wagi modeli: katalog `models/<nazwa_eksperymentu>.zip`.

---

## 5. Zasady projektowe

Bezwzględne przestrzeganie zasad zdefiniowanych w `AGENTS.md`:
1. **KISS** — minimalna złożoność implementacji. Projekt ma cel badawczy, nie produkcyjny.
2. **YAGNI** — brak funkcjonalności spoza tego dokumentu. Brak baz danych, frameworków webowych, dashboardów.
3. **SOLID** — modularność kodu. Separacja odpowiedzialności.
4. **Dokumentacja** — docstringi NumPy, pdoc, 100% coverage testów.
5. **Kontrola wersji** — praca na branch'ach, gatekeeping via GitHub Actions.

---

## 6. Definition of Done

1. Skrypt automatyzujący trening czyta macierz konfiguracji z pliku CSV i loguje wyniki bez ingerencji programisty.
2. Każdy eksperyment odkłada metryki (`ep_rew_mean`, `loss`) do czytelnych wykresów w TensorBoard.
3. Skrypt ewaluacyjny (`evaluate.py`) ładuje najlepsze wagi z dysku i renderuje wizualnie na żywo poczynania agenta (`render_mode="human"`).
4. Wygenerowane wykresy z TensorBoard potwierdzają wpływ wielkości sieci na czas i jakość uczenia.
5. Pokrycie testami: **100%**.
6. Pipeline CI/CD: zielony status na wszystkich etapach (Ruff, Mypy, Pytest).
7. Cała komunikacja z modelami AI wyeksportowana i dołączona do repozytorium.
