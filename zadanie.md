# Projekt zaliczeniowy
--- 
**Przedmiot:** Sztuczna inteligencja
**Wykładowca:** Dr. Maciej Kraszewski 

Opis wymagań do projektu. Prezentacje należy przeprowadzić na ostatnim zjeździe.

---
## Wymagania
1. Wytrenować agenta grającego w grę z biblioteki Gymnasium.
2. Nie wybieramy gier z grupy Toy Text.
3. Należy udokumentować pracę z agentami AI / LLMami itd. (np. historia promptów).
4. Podczas prezentacji:
    - opis gry (cel, przestrzeń akcji, przestrzeń stanu, system nagród),
    - przetestowane architektury sieci neuronowych (jak wpływa na wynik, tempo uczenia itd.),
    - przetestowanie hiperparametrów,
    - demo live wytrenowanego agenta.

--- 
## Materiały
1. **Dokumentacja Gymnasium:** https://gymnasium.farama.org/environments/classic_control/
2. **Dokumentacja Stable-Baselines3:** https://stable-baselines3.readthedocs.io/

---
## Decyzje architektoniczne
1. **Środowiska i cel:** Wykonane zostaną dwa projekty:
    - **Cart Pole:** Środowisko bazowe (poligon do testów), które ma na celu błyskawiczne zebranie danych analitycznych w mnogiej konfiguracji hiperparametrów. 
    - **Lunar Lander:** Środowisko średnio-zaawansowane o złożonej funkcji nagrody, które posłuży jako docelowy przykład do prezentacji.  
2. **Hardware:** Wykorzystanie wyłącznie lokalnych zasobów MacBook'a Air M4 (24 GB RAM zunifikowanej pamięci). Ze względu na chłodzenie pasywne, architektura skryptów wymusza przerwy w treningu, zapobiegające thermal throttlingowi.
3. **Algorytm:** Proximal Policy Optimization (PPO) z biblioteki `stable-baselines3`. Stabilny algorytm off-the-policy działający na dyskretnych i ciągłych przestrzeniach akcji.

---
## Eksperymenty i analityka
Do zrealizowania w ramach testów optymalizacyjnych: 

1. **Architektura sieci neuronowej:** 
    Przetestowane zostaną 3 wielkości sieci, aby twardo wykazać, że większa sieć nie oznacza lepszego wyniku. Celem jest udowodnienie zasady YAGNI w kontekście przeinwestowania w liczbę parametrów:
    - *Za mała:* `[16, 16]` (Sprawdzenie pojemności informacyjnej modelu)
    - *Optymalna:* `[64, 64]` (Baseline dla tego typu złożoności wektora stanu)
    - *Zbyt duża:* `[1024, 1024, 1024]` (Wykazanie spadku wydajności obliczeniowej i ryzyka overfittingu)

2. **Hiperparametry:** Zdefiniowano 5 zmiennych podlegających modyfikacji w 3 wariantach w celu zbadania ich wpływu na zbieżność i stabilność modelu:
    - `learning_rate` (Współczynnik uczenia - wpływ na agresywność optymalizatora)
    - `batch_size` (Wielkość partii danych - wpływ na stabilność gradientu oraz obciążenie pamięci RAM)
    - `gamma` (Współczynnik dyskontowania - określa horyzont planowania i "dalekowzroczność" agenta)
    - `n_steps` (Rozmiar bufora - liczba kroków symulacji przed aktualizacją wag)
    - `ent_coef` (Współczynnik entropii - sztuczne wymuszanie eksploracji i zapobieganie uwięzieniu modelu w minimum lokalnym)

---
## Definition of Done
- Skrypt automatyzujący trening czyta macierz konfiguracji z pliku `.csv` i loguje wyniki bez ingerencji programisty.
- Każdy eksperyment odkłada metryki (`ep_rew_mean`, `loss`) do czytelnych wykresów w TensorBoard.
- Skrypt inferencyjny (`evaluate.py`) jest w stanie załadować najlepsze wagi z dysku i wyrenderować wizualnie na żywo poczynania agenta (`render_mode="human"`).
- Cała komunikacja z modelami (Sonnet/Opus) jest wyeksportowana i dołączona do dokumentacji repozytorium.

---
## Backlog realizacji
[x] Stworzenie repozytorium
[x] Przygotowanie planu realizacji zadania
[x] Przygotowanie folderu .github oraz struktury pod pracę z agentami AI
[x] Zaprojektowanie bazy promptów agentów. baza modeli: Claude Opus 4.6 (zakładając dostępność w ramach subskrypcji Google) + Claude Connet 4.6 (Github Copilot)
[x] Przygotowanie wsadu do plików kontekstowych: 
    [x] PRD.md (Project Requirements Document - plik produktowy)
    [x] ADR.md (Architecture Decision Record - backlog decyzji)
    [x] Architecture_and_Tasks.md
    [x] Dev_Knowledge_Base.md
    [x] AGENTS.md


---
## Materiał wsadowy dla promptu

### `AGENTS.md` załozenia:
1. Plik opisuje relacje między agentami, ogólne dobre praktyki, których mają przestrzegać agenci w celu optymalizacji długości promptów agentów i utrzymania jednego punktu kontroli jakości projektu.

2. **Zasady jakich bezwzględnie muszą przestrzegać agenci:**
    - **KISS** - Keep it Simple, Stupid - projekt ma jasno określony cel badawczy, nie będzie on służył dalszemu rozwojowi, realizowane jest wyłącznie wymagane minimum implementacji kodu, oraz jego prostota. Główne założenie projektu to analiza wpływu hiperparametrów na optymalność treningu. 
    - **YAGNI** - You Aren't Gonna Need It - nie potrzebujemy skalowalności i dodatkowych funkcjonalności, zasoby Macbook'a są limitowane, wszystko co jest zbędnym nadmiarem obliczeniowym, lub zwiększa złożoność algorytmów jest zbędne.
    - **SOLID** 
    - **Pełna dokumentacja projektu:**
        - **Struktura:** Doctring w formacie NumPy 
        - **Prezentacja:** Pdoc
        - **Pokrycie:** wszystkie klasy, metody, funkcje posiadają pełną dokumentację w języku polskim wraz z opisem parametrów, typów danych i zwracanych argumentów. Testy jednostkowe oraz integracyjne dodatkowo posiadają scenariusze testowe.
    - **Pokrycie testami:** Mimo utrzymywania projektu na niskim poziomie rozwoju istotnym jest 100% coverage wszystkich funkcji i metod. Wskazane jest mock'owanie danych. Kod powinien zawierać pokrycie w testach jednostkowych oraz integracyjnych testach e2e całego pipeline'u.
    - **Logowanie decyzji:** Wszystkie decyzje podjęte w wyniku interakcji z użytkownikiem lub w wyniku błędów implementacyjnych wykrytych przez agentów muszą być logowane w pliku `ADR.md`. Niezbędna jest pełna audytowalność procesu powstawania projektu.
    - **Planowanie pracy:** plik `architecture_and_tasks.md` Zawiera pełny backlog realizacji, agenci po wykonaniu pracy muszą logować stan projektu, odhaczając wykonane zadania oraz wskazując jakie sugestie przekazują do następnych epików, jeżeli takie będą konieczne. Plik określa architekturę projektu i jego aktualny stan.
    - **Historia zmian:** `CHANGELOG.md` Pełna dokumentacja zmian w projekcie, wersjonowanie **SemVer**.
    - **Analityka:** Projekt bazuje na danych testowych przekazywanych poprzez plik .csv z parametrami, jednocześnie w tym pliku muszą znaleźć się finalnie również wyniki treningu dla każdego treningu. Jednocześnie zachowane są wszystkie pliki z wagami, oraz pliki wykorzystywane dalej do pełnej analityki treningu za pomocą TensorBoard.
    - **Kontrola wersji:** Praca nigdy nie może odbywać się na gałęzi main, każdy z agentów powinien utworzyć nową gałąź, która dopiero po przejściu przez gatekeeping może zostać zmergowana.
    - **Gatekeeping:** Wykorzystanie GitHub Actions. Żaden kod nie może zostać włączony do gałęzi `main` bez przejścia pipeline'u na etapie Pull Requestu. Rurociąg uruchamia agent `3-gatekeeper-devops.agent.md` i musi obejmować:
        - **Linter i formatter (Ruff):** Pojedyncze, błyskawiczne narzędzie weryfikujące składnię, sortujące importy.
        - **Typowanie (Mypy):** Statyczna analiza kodu pod kątem zgodności typów wejściowych i wyjściowych.
        - **Walidacja dokumentacji:** Wymuszona przez Ruff weryfikacja istnienia i poprawności struktury docstringów w standardzie NumPy.
        - **Testy i coverage (Pytest + pytest-cov):** Uruchomienie zestawu testów jednostkowych i integracyjnych. Pipeline zwraca błąd (Exit 1), jeśli pokrycie spadnie poniżej 100%.

3. **Wysokopoziomowy opis agentów do stworzenia:**
-`01-architect.agent.md` 
    - **Rola:** Główny architekt projektu / Product Owner. Strażnikiem wizji, architektury i decyzyjności. Jego celem jest przekładanie wymagań biznesowych z pliku PRD na konkretne, techniczne zadania dla agenta `1-developer.agent.md`.
    - **Zadania:**  
        - Inicjowanie i aktualizowanie plików dokumentacji architektonicznej (`ADR.md`, `Architecture_and_Tasks.md`).
        - Dzielenie epików na najmniejsze, samowystarczalne zadania zgodnie z zasadą KISS.
        - Ocenianie propozycji rozwiązań od agentów deweloperskich i wetowanie tych, które łamią zasadę YAGNI (np. propozycje wdrożenia ciężkich baz danych zamiast arkuszy CSV).
        - Nadzorowanie przepływu pracy i delegowanie zadań do agenta `1-developer.agent.md`.
    - **Ograniczenia:**  
        - Ściśle przestrzega zasad zapisanych w pliku `AGENTS.md`
        - Nie pisze kodu wykonawczego. Od tego jest `1-developer.agent.md`.
        - Nigdy nie modyfikuje kodu, ograniczasz się do zaktualizowania backlogu w pliku `Architecture_and_Tasks.md` i pliku `ADR.md`.
        - Odpowiedzi muszą być zwięzłe, dyrektywne i pozbawione "sztucznej uprzejmości".

- `1-developer.agent.md`
    - **Rola:** Senior Developer. Odpowiada za fizyczną implementację logiki biznesowej, modeli uczenia ze wzmocnieniem oraz skryptów narzędziowych.
    - **Zadania:**  
        - Pisze zwięzły, modularny kod w języku Python zgodnie z wytycznymi `0-architect.agent.md`.
        - Implementuje środowiska Gymnasium oraz algorytmy z biblioteki `stable-baselines3`.
        - Tworzy pełną dokumentację w standardzie NumPy Style Docstrings dla każdej nowej funkcji i klasy.
        - Aktualizuje plik `CHANGELOG.md` w ujęciu SemVer po każdym zamkniętym zadaniu.
    - **Ograniczenia:**  
        - Ściśle przestrzega zasad zapisanych w pliku `AGENTS.md`
        - Pracuje wyłącznie na dedykowanych branch'ach, nigdy nie pushuje bezpośrednio na `main`.
        - Nie podejmuje decyzji architektonicznych bez logowania ich w `ADR.md` i autoryzacji `0-architect.agent.md`.
        - Wszystkie błędy oraz napotkane problemy opisuje w pliku `dev-knowledge-base.md`, przy napotkaniu na błąd w pierwszej kolejności tam sprawdza, czy taki problem już zaistniał, jeżeli tak stosuje znane rozwiązanie, jeżeli nie opisuje problem i jego rozwiązanie. 

- `2-qa.agent.md`
    - **Rola:** QA. Weryfikuje poprawność implementacji, rygorystycznie egzekwuje 100% pokrycia testami oraz sprawdza spójność dokumentacji.
    - **Zadania:**
        - Pisze i utrzymuje testy jednostkowe z wykorzystaniem frameworka `pytest`.
        - Nigdy nie modyfikuje kodu, by przeszedł on testy. Loguje rzeczy do poprawnienia w pliku `test-report.md` 
        - Implementuje techniki mockowania, aby testy weryfikowały logikę bez konieczności fizycznego odpalania czasochłonnych pętli treningowych RL.
        - Weryfikuje istnienie oraz poprawność składniową Docstringów (NumPy) we wszystkich plikach `.py`.
        - Diagnozuje błędy i luki zgłoszone przez bramki CI/CD i generuje łaty (patches).
    - **Ograniczenia:**  
        - Ściśle przestrzega zasad zapisanych w pliku `AGENTS.md`
        - Nie tworzy nowych funkcjonalności produkcyjnych.
        - Pod żadnym pozorem nie obniża progu wymaganego pokrycia testami (coverage).
        - Traktuje kod testowy z takim samym rygorem jakościowym jak kod produkcyjny.

- `3-gatekeeper-devops.agent.md`
    - **Rola:** Strażnik pipeline'u i devops. Automatyzuje procesy integracji, pilnuje rygoru jakości i stabilności głównej gałęzi repozytorium. Zapewnia brak konfliktów i bezpieczeństwo
    - **Zadania:**  
        - Konfiguruje i utrzymuje skrypty GitHub Actions (`.github/workflows/gatekeeper.yml`).
        - Zarządza konfiguracją narzędzi statycznej analizy kodu (Ruff, Mypy) i bezwzględnie wymusza status *Exit 1* w przypadku wykrycia jakichkolwiek nieprawidłowości.
        - Kontroluje Pull Requesty – przepuszcza do gałęzi `main` wyłącznie kod, który przeszedł weryfikację lintera i 100% testów.
        - Automatyzuje proces generowania dokumentacji HTML za pomocą `pdoc` oraz zarządzania plikami wyjściowymi TensorBoard.
        - Sprawdza, czy w kodzie nie występują sekrety oraz pliki niepożądane.
    - **Ograniczenia:**  
        - Ściśle przestrzega zasad zapisanych w pliku `AGENTS.md`
        - Jest bezlitosny dla jakości kodu – nie pozwala na żadne wyjątki  w rurociągu integracyjnym.
        - Nie pisze testów jednostkowych; jego rolą jest wyłącznie izolacja i uruchomienie procesu weryfikacji.

4. **Wysokopoziomowy opis pliku PRD.md:**
- **Cel dokumentu:** Plik jest profesjonalnym dokumentem PRD dla opisanego w tym pliku zadania.
- **Metadane projektu:**
    - **Nazwa:** Optymalizacja hiperparametrów i architektury sieci w algorytmach RL.
    - **Kontekst:** Projekt zaliczeniowy z przedmiotu Sztuczna Inteligencja.
    - **Wykładowca:** Dr Maciej Kraszewski.
    - **Termin oddania:** Koniec czerwca 2026.
    - **Zasób sprzętowy:** MacBook Air M4 (24GB RAM, zunifikowana pamięć, chłodzenie pasywne - ryzyko thermal throttlingu).

- **Wizja i cel produktu:** Celem projektu nie jest stworzenie idealnego agenta AI, lecz **udowodnienie analitycznego podejścia inżyniera do procesu uczenia maszynowego**. Projekt ma wykazać, poprzez twarde dane z logów, jak zmiana pojemności sieci neuronowej (zasada YAGNI) oraz manipulacja hiperparametrami algorytmu PPO wpływają na zbieżność, stabilność i czas treningu. Produktem końcowym jest zautomatyzowany i udokumentowany pipeline treningowy oraz widowiskowe demo na zaliczenie.

- **Zakres unkcjonalny:** Projekt składa się z dwóch niezależnych środowisk, obsługiwanych przez ten sam zunifikowany kod:
    1.  **CartPole-v1 (Baza analityczna):** Szybkie w treningu środowisko o dyskretnej przestrzeni akcji. Posłuży do błyskawicznego przemielenia macierzy 18 eksperymentów i wygenerowania wykresów.
    2.  **LunarLander-v2 (Demo docelowe):** Środowisko o podwyższonej złożoności z bogatym systemem nagród. Posłuży do weryfikacji najlepszych wag na żywo.

- **Zakres Badawczy:** System musi być w stanie przyjąć plik konfiguracyjny (CSV) i dla każdego wiersza wykonać trening, badając:
-   **Zmienną główną (Architektura Sieci MLP):** 
    - Za mała `[16, 16]`
    - Optymalna `[64, 64]`
    - Zbyt duża `[1024, 1024, 1024]` (cel: wykazanie spadku wydajności / overfittingu).
-   **Zmienne poboczne (5 hiperparametrów PPO w 3 wariantach):**
    - `learning_rate`
    - `batch_size`
    - `gamma`
    - `n_steps`
    - `ent_coef`

- **Wytyczne techniczne i ograniczenia:**
    -   **Algorytm:** PPO (Proximal Policy Optimization) z biblioteki `stable-baselines3`.
    -   **Język:** Python (najnowsza stabilna wersja kompatybilna z sb3).
    -   **Zasady projektowe:** Bezwzględne zachowanie praktyk opisanych w pliku `AGENTS.md`.
    -   **Ochrona sprzętu:** Pętla treningowa musi posiadać mechanizm wymuszonego usypiania wątku (np. `time.sleep`) pomiędzy eksperymentami, aby zapobiec przegrzaniu pasywnie chłodzonego procesora M4.
    -   **Zarządzanie danymi:** Parametry wejściowe pobierane są z pliku CSV. Skrypt po treningu musi dopisać do tego samego pliku metryki wynikowe oraz odłożyć pełne logi dla TensorBoard.

- **Definition of Done:** 
    1. Kod automatycznie wykonuje pętlę po pliku konfiguracyjnym bez interwencji człowieka.
    2. Istnieje wyodrębniony skrypt ewaluacyjny, który potrafi załadować wytrenowany model i odpalić grę w trybie graficznym (`render_mode="human"`).
    3. Wygenerowane są wykresy z TensorBoard potwierdzające wpływ wielkości sieci na czas i jakość uczenia.

- **Instrukcja dla Agenta Architekta:** Na podstawie powyższych danych wejściowych, wygeneruj profesjonalny plik `PRD.md`. Dokument ma być napisany w języku polskim, w formie bezosobowej (trzecioosobowej), w sposób wysoce sformalizowany, zwięzły i techniczny. Nie dodawaj funkcjonalności, które nie zostały wymienione w tym pliku.