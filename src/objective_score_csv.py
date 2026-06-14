"""Narzędzia do wyliczania objective_score dla tabel wynikowych CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def compute_objective_score(
    mean_reward: float,
    std_reward: float,
    stability_penalty: float,
) -> float:
    """Policz objective_score dla pojedynczego rekordu wynikow.

    Parameters
    ----------
    mean_reward : float
        Srednia nagroda modelu.
    std_reward : float
        Odchylenie standardowe nagrody.
    stability_penalty : float
        Kara stabilnosci odejmowana od sredniej nagrody.

    Returns
    -------
    float
        Wynik objective_score.
    """
    return mean_reward - (stability_penalty * std_reward)


def recompute_objective_score_csv(
    csv_path: str,
    stability_penalty: float,
    output_path: str | None = None,
) -> str:
    """Przelicz objective_score dla wszystkich rekordow CSV.

    Parameters
    ----------
    csv_path : str
        Sciezka do wejsciowego pliku CSV.
    stability_penalty : float
        Kara stabilnosci dla objective_score.
    output_path : str | None, default=None
        Sciezka do pliku wyjsciowego. Gdy ``None``, wynik jest zapisywany
        obok pliku z sufiksem ``_with_objective``.

    Returns
    -------
    str
        Sciezka do zapisanego pliku wynikowego.

    Raises
    ------
    ValueError
        Gdy brak wymaganych kolumn ``mean_reward`` lub ``std_reward``.
    """
    source = Path(csv_path)
    target = (
        Path(output_path)
        if output_path is not None
        else source.with_name(f"{source.stem}_with_objective{source.suffix}")
    )

    with source.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = list(reader.fieldnames or [])
        if "mean_reward" not in fieldnames or "std_reward" not in fieldnames:
            raise ValueError(
                "CSV musi zawierac kolumny mean_reward i std_reward."
            )
        if "objective_score" not in fieldnames:
            fieldnames.append("objective_score")

        rows: list[dict[str, str]] = []
        for row in reader:
            mean_raw = (row.get("mean_reward") or "").strip()
            std_raw = (row.get("std_reward") or "").strip()

            if mean_raw and std_raw:
                objective = compute_objective_score(
                    mean_reward=float(mean_raw),
                    std_reward=float(std_raw),
                    stability_penalty=stability_penalty,
                )
                row["objective_score"] = f"{objective:.10f}".rstrip("0").rstrip(".")
            else:
                row["objective_score"] = ""

            rows.append(row)

    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return str(target)


def main() -> None:
    """Uruchom CLI przeliczania objective_score dla wielu CSV."""
    parser = argparse.ArgumentParser(
        description=(
            "Przelicza objective_score = mean_reward - penalty * std_reward "
            "dla jednego lub wielu plikow CSV."
        )
    )
    parser.add_argument(
        "--csv-paths",
        nargs="+",
        required=True,
        help="Lista sciezek do CSV z kolumnami mean_reward i std_reward.",
    )
    parser.add_argument(
        "--stability-penalty",
        type=float,
        default=0.1,
        help="Wspolczynnik kary stabilnosci (domyslnie: 0.1).",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Nadpisz wskazane pliki CSV zamiast tworzyc kopie _with_objective.",
    )
    args = parser.parse_args()

    for path in args.csv_paths:
        output = recompute_objective_score_csv(
            csv_path=path,
            stability_penalty=args.stability_penalty,
            output_path=path if args.in_place else None,
        )
        print(f"Przeliczono objective_score: {output}")


if __name__ == "__main__":  # pragma: no cover
    main()
