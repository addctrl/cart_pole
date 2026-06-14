"""Eksport danych skalarnych TensorBoard do CSV dla wielu runow."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def discover_event_files(logdir: str) -> list[Path]:
    """Znajdz wszystkie pliki eventow TensorBoard pod wskazanym katalogiem.

    Parameters
    ----------
    logdir : str
        Sciezka do katalogu nadrzednego z runami TensorBoard.

    Returns
    -------
    list[Path]
        Posortowana lista plikow ``events.out.tfevents*``.
    """
    root = Path(logdir)
    return sorted(root.rglob("events.out.tfevents*"))


def extract_scalar_rows(logdir: str, tags: set[str] | None = None) -> list[dict[str, Any]]:
    """Wyciagnij wszystkie skalary TensorBoard do postaci listy wierszy.

    Parameters
    ----------
    logdir : str
        Sciezka do katalogu nadrzednego z runami TensorBoard.
    tags : set[str] | None, default=None
        Opcjonalny zbior tagow do filtrowania. Jesli ``None``, zwracane sa
        wszystkie znalezione tagi skalarne.

    Returns
    -------
    list[dict[str, Any]]
        Lista rekordow ze strukturą: ``run``, ``step``, ``tag``, ``value``.
    """
    root = Path(logdir)
    rows: list[dict[str, Any]] = []

    for event_file in discover_event_files(logdir):
        run_name = str(event_file.parent.relative_to(root))
        accumulator = EventAccumulator(str(event_file))
        accumulator.Reload()

        available_tags = accumulator.Tags().get("scalars", [])
        selected_tags = [
            tag for tag in available_tags if tags is None or tag in tags
        ]

        for tag in selected_tags:
            for event in accumulator.Scalars(tag):
                rows.append(
                    {
                        "run": run_name,
                        "step": int(event.step),
                        "tag": tag,
                        "value": float(event.value),
                    }
                )

    rows.sort(key=lambda row: (str(row["run"]), int(row["step"]), str(row["tag"])))
    return rows


def write_long_csv(rows: list[dict[str, Any]], output_csv: str) -> None:
    """Zapisz surowe wiersze skalarne do formatu long CSV.

    Parameters
    ----------
    rows : list[dict[str, Any]]
        Wiersze zwrocone przez ``extract_scalar_rows``.
    output_csv : str
        Sciezka wyjsciowa pliku CSV.
    """
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["run", "step", "tag", "value"])
        writer.writeheader()
        writer.writerows(rows)


def write_pivot_csv(rows: list[dict[str, Any]], output_csv: str) -> int:
    """Zapisz dane skalarne do formatu pivot (kolumna per tag).

    Parameters
    ----------
    rows : list[dict[str, Any]]
        Wiersze zwrocone przez ``extract_scalar_rows``.
    output_csv : str
        Sciezka wyjsciowa pliku CSV.

    Returns
    -------
    int
        Liczba zapisanych rekordow pivot.
    """
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    grouped: dict[tuple[str, int], dict[str, float]] = {}
    tags: set[str] = set()

    for row in rows:
        run = str(row["run"])
        step = int(row["step"])
        tag = str(row["tag"])
        value = float(row["value"])

        tags.add(tag)
        key = (run, step)
        if key not in grouped:
            grouped[key] = {}
        grouped[key][tag] = value

    ordered_tags = sorted(tags)
    fieldnames = ["run", "step", *ordered_tags]

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for run, step in sorted(grouped.keys()):
            row: dict[str, Any] = {"run": run, "step": step}
            row.update(grouped[(run, step)])
            writer.writerow(row)

    return len(grouped)


def main() -> None:
    """Uruchom CLI eksportu TensorBoard do CSV."""
    parser = argparse.ArgumentParser(
        description=(
            "Eksport danych skalarnych TensorBoard z wielu runow jednoczesnie "
            "do CSV (pivot i opcjonalnie long)."
        )
    )
    parser.add_argument(
        "--logdir",
        required=True,
        help="Katalog nadrzedny z logami TensorBoard (np. logs/tensorboard).",
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Sciezka wyjsciowa CSV w formacie pivot.",
    )
    parser.add_argument(
        "--long-output-csv",
        help="Opcjonalna sciezka wyjsciowa CSV w formacie long.",
    )
    parser.add_argument(
        "--tags",
        nargs="*",
        help="Opcjonalna lista tagow do filtrowania.",
    )
    args = parser.parse_args()

    selected_tags = set(args.tags) if args.tags else None
    rows = extract_scalar_rows(args.logdir, selected_tags)

    if not rows:
        print("Nie znaleziono zadnych danych skalarnych w tym katalogu.")
        return

    pivot_rows = write_pivot_csv(rows, args.output_csv)
    print(f"Zapisano {pivot_rows} rekordow pivot do: {args.output_csv}")

    if args.long_output_csv:
        write_long_csv(rows, args.long_output_csv)
        print(f"Zapisano {len(rows)} rekordow long do: {args.long_output_csv}")


if __name__ == "__main__":  # pragma: no cover
    main()
