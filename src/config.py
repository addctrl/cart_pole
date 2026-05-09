"""Moduł konfiguracji eksperymentów — operacje I/O na pliku CSV."""

import ast
import csv
from pathlib import Path
from typing import Any


def parse_net_arch(raw: str) -> list[int]:
    """Parsuj string architektury sieci neuronowej na listę dodatnich intów.

    Parameters
    ----------
    raw : str
        String w formacie ``'[64, 64]'`` reprezentujący rozmiary
        kolejnych warstw ukrytych sieci.

    Returns
    -------
    list[int]
        Lista dodatnich intów — rozmiary kolejnych warstw ukrytych.

    Raises
    ------
    ValueError
        Jeśli format stringa jest niepoprawny lub elementy nie są
        pozytywnymi intami.
    """
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError) as exc:
        raise ValueError(f"Niepoprawny format architektury sieci: {raw!r}") from exc

    if not isinstance(parsed, list):
        raise ValueError(
            f"Architektura sieci musi być listą, nie: {type(parsed).__name__!r}"
        )

    arch: list[int] = []
    for item in parsed:
        if not isinstance(item, int) or item <= 0:
            raise ValueError(
                f"Elementy architektury muszą być pozytywnymi intami: {raw!r}"
            )
        arch.append(item)

    return arch


def load_experiments(path: str) -> list[dict[str, Any]]:
    """Wczytaj macierz eksperymentów z pliku CSV.

    Konwertuje typy pól: ``learning_rate``, ``gamma``, ``ent_coef``
    na ``float``; ``batch_size``, ``n_steps``, ``total_timesteps`` na
    ``int``; kolumny wynikowe na ``float | None``.

    Parameters
    ----------
    path : str
        Ścieżka do pliku CSV z konfiguracją eksperymentów.

    Returns
    -------
    list[dict[str, Any]]
        Lista słowników z eksperymentami i skonwertowanymi typami pól.
        Puste kolumny wynikowe (``mean_reward``, ``std_reward``,
        ``training_time_s``) są zwracane jako ``None``.

    Raises
    ------
    FileNotFoundError
        Jeśli plik CSV nie istnieje pod podaną ścieżką.
    ValueError
        Jeśli dla dowolnego eksperymentu ``n_steps`` nie jest podzielne
        przez ``batch_size`` (wymóg PPO, DKB-003).
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Plik CSV nie istnieje: {path}")

    experiments: list[dict[str, Any]] = []

    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            exp: dict[str, Any] = dict(row)

            exp["learning_rate"] = float(exp["learning_rate"])
            exp["gamma"] = float(exp["gamma"])
            exp["ent_coef"] = float(exp["ent_coef"])
            exp["batch_size"] = int(exp["batch_size"])
            exp["n_steps"] = int(exp["n_steps"])
            exp["total_timesteps"] = int(exp["total_timesteps"])

            for col in ("mean_reward", "std_reward", "training_time_s"):
                val = exp[col]
                exp[col] = float(val) if val else None

            n_steps: int = exp["n_steps"]
            batch_size: int = exp["batch_size"]
            if n_steps % batch_size != 0:
                raise ValueError(
                    f"Eksperyment {exp['experiment_id']!r}: "
                    f"n_steps={n_steps} musi być podzielne przez "
                    f"batch_size={batch_size} (DKB-003)."
                )

            experiments.append(exp)

    return experiments


def save_results(
    path: str,
    experiment_id: str,
    metrics: dict[str, float],
) -> None:
    """Zapisz metryki wynikowe do istniejącego wiersza w pliku CSV.

    Wczytuje cały plik do pamięci, aktualizuje wskazany wiersz
    i nadpisuje plik. Operacja efektywnie atomowa dla małych plików.

    Parameters
    ----------
    path : str
        Ścieżka do pliku CSV z eksperymentami.
    experiment_id : str
        Identyfikator eksperymentu (kolumna ``experiment_id``) do
        zaktualizowania.
    metrics : dict[str, float]
        Słownik z metrykami: ``mean_reward``, ``std_reward``,
        ``training_time_s``.

    Raises
    ------
    ValueError
        Jeśli ``experiment_id`` nie istnieje w pliku CSV.
    """
    csv_path = Path(path)
    rows: list[dict[str, Any]] = []
    fieldnames: list[str] = []
    found = False

    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            if row["experiment_id"] == experiment_id:
                row.update({k: str(v) for k, v in metrics.items()})
                found = True
            rows.append(dict(row))

    if not found:
        raise ValueError(f"Eksperyment {experiment_id!r} nie istnieje w pliku CSV.")

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
