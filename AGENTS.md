# AGENTS.md — Manifest Współpracy Agentów

> Nadrzędny dokument regulujący zasady współpracy agentów AI w projekcie.
> **Każdy agent musi przeczytać ten plik przed podjęciem jakiegokolwiek działania.**

---

## 1. Agenci

| Agent | Plik | Rola |
|---|---|---|
| **Architekt** | `.github/agents/0-architect.agent.md` | Product Owner. Strażnik wizji i architektury. Planuje, deleguje, wetuje. |
| **Developer** | `.github/agents/1-developer.agent.md` | Senior Developer. Implementuje kod Python. |
| **QA** | `.github/agents/2-qa.agent.md` | Inżynier QA. Pisze testy, egzekwuje 100% coverage. |
| **Gatekeeper** | `.github/agents/3-gatekeeper-devops.agent.md` | DevOps. Pilnuje pipeline'u CI/CD. |

---

## 2. Przepływ pracy

```
Architekt ──► Developer ──► QA ──► Gatekeeper ──► main
   │              │           │          │
   │              │           │          └── Pipeline CI/CD
   │              │           └── Testy + coverage
   │              └── Implementacja + CHANGELOG
   └── Planowanie + ADR + backlog
```

### Reguły przepływu

1. **Architekt** formułuje zadanie → **Developer** implementuje na branch'u.
2. **Developer** kończy implementację → **QA** pisze/uruchamia testy.
3. **QA** potwierdza 100% coverage → **Developer** tworzy PR do `main`.
4. **Gatekeeper** uruchamia pipeline → merge lub blokada.
5. Żaden agent nie pomija etapów. Brak skrótów.

---

## 3. Zasady bezwzględne

### 3.1 KISS — Keep It Simple, Stupid

Projekt ma jasno określony cel badawczy. Nie będzie służył dalszemu rozwojowi. Realizowane jest **wyłącznie wymagane minimum implementacji**. Złożoność kodu musi być proporcjonalna do złożoności problemu.

**Praktycznie oznacza to:**
- Brak klas abstrakcyjnych tam, gdzie wystarczy funkcja.
- Brak wzorców projektowych „na zapas".
- Flat structure — brak zagnieżdżonych pakietów.
- Jeden plik = jeden moduł odpowiedzialności.

### 3.2 YAGNI — You Aren't Gonna Need It

Zasoby MacBooka są limitowane. Wszystko, co jest zbędnym nadmiarem obliczeniowym lub zwiększa złożoność algorytmów, jest **zabronione**.

**Zabronione:**
- Bazy danych (SQLite, PostgreSQL). Format danych: CSV.
- Frameworki webowe (Flask, FastAPI). Wizualizacja: TensorBoard.
- Dodatkowe algorytmy RL (DQN, SAC). Algorytm: PPO.
- Systemy kolejkowania zadań (Celery). Pętla treningowa: sekwencyjna.
- Dockeryzacja. Środowisko: venv.

### 3.3 SOLID

- **S (Single Responsibility):** Każdy moduł ma jedną odpowiedzialność. `config.py` — dane. `training.py` — trening. `evaluate.py` — ewaluacja.
- **O (Open/Closed):** Konfiguracja via CSV — rozszerzanie przez dodanie wiersza, nie modyfikację kodu.
- **L (Liskov Substitution):** Nie dotyczy — projekt nie używa dziedziczenia.
- **I (Interface Segregation):** Nie dotyczy — brak interfejsów. Funkcje mają minimalne sygnatury.
- **D (Dependency Inversion):** Ścieżki plików i nazwy środowisk jako parametry, nie hardkodowane wartości.

### 3.4 Dokumentacja

| Element | Wymaganie |
|---|---|
| **Format docstringów** | NumPy Style |
| **Język** | Polski |
| **Pokrycie** | Wszystkie klasy, metody, funkcje — bez wyjątków |
| **Sekcje wymagane** | Opis, `Parameters`, `Returns`, `Raises` (gdzie applicable) |
| **Generowanie HTML** | `pdoc` |
| **Prezentacja** | `pdoc src/ -o docs/` |

### 3.5 Pokrycie testami

| Element | Wymaganie |
|---|---|
| **Framework** | `pytest` + `pytest-cov` |
| **Wymagane pokrycie** | **100%** — bez wyjątków |
| **Mockowanie** | Obowiązkowe dla treningu RL, renderowania, I/O |
| **Kod testowy** | Ten sam rygor co produkcyjny: type hints, docstringi, Ruff, Mypy |

### 3.6 Logowanie decyzji

Wszystkie decyzje architektoniczne muszą być zalogowane w `.github/artifacts/adr.md`. Wymagane pola:
- Numer ADR
- Data
- Kontekst
- Decyzja
- Uzasadnienie
- Konsekwencje
- Odrzucone alternatywy

### 3.7 Planowanie pracy

Plik `.github/artifacts/architecture_and_tasks.md` zawiera pełny backlog. Agenci po wykonaniu pracy:
1. Odhaczają wykonane zadania (`[x]`).
2. Wskazują sugestie dla kolejnych epików.
3. Aktualizują status.

### 3.8 Historia zmian

Plik `CHANGELOG.md` — pełna dokumentacja zmian. Format: **SemVer** (MAJOR.MINOR.PATCH).

```markdown
## [0.1.0] - 2026-05-XX
### Dodane
- Opis zmian
```

### 3.9 Kontrola wersji

- Praca **nigdy** na gałęzi `main`.
- Format branch'y: `feat/<opis>`, `fix/<opis>`, `docs/<opis>`.
- Merge wyłącznie przez PR z zielonym pipeline'em.

### 3.10 Gatekeeping (CI/CD)

Pipeline GitHub Actions (`.github/workflows/gatekeeper.yml`) uruchamiany na PR do `main`. Etapy:

| Etap | Narzędzie | Warunek sukcesu |
|---|---|---|
| Linter | Ruff | Zero błędów |
| Formatter | Ruff format | Zero różnic |
| Typowanie | Mypy (strict) | Zero błędów |
| Testy | Pytest + pytest-cov | 100% coverage |
| Docstringi | Ruff (reguły D) | Kompletne NumPy docstringi |

**Exit 1** przy jakimkolwiek naruszeniu. Brak wyjątków.

---

## 4. Pliki kontekstowe agentów

| Plik | Ścieżka | Opis |
|---|---|---|
| PRD | `.github/artifacts/prd.md` | Wymagania produktowe |
| ADR | `.github/artifacts/adr.md` | Rejestr decyzji architektonicznych |
| Backlog | `.github/artifacts/architecture_and_tasks.md` | Architektura i zadania |
| Baza wiedzy | `.github/artifacts/dev_knowledge_base.md` | Znane problemy i rozwiązania |
| Raport testów | `.github/artifacts/test-report.md` | Status testów i coverage |

---

## 5. Styl komunikacji

- **Język:** Polski.
- **Ton:** Senior Engineer / Lead. Bezpośredni, techniczny, konkretny.
- **Zakaz:** Sztuczna uprzejmość, pochwały, emotikony, marketing.
- **Format:** Polecenia, konstatacje, decyzje. Bez „może warto rozważyć" — albo rób, albo nie.
