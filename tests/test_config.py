"""Testy jednostkowe modułu ``src.config``."""

from pathlib import Path

import pytest

from src.config import load_experiments, parse_net_arch, save_results

# ---------------------------------------------------------------------------
# Stałe pomocnicze
# ---------------------------------------------------------------------------

_CSV_HEADER = (
    "experiment_id,env_id,net_arch,learning_rate,batch_size,"
    "gamma,n_steps,ent_coef,total_timesteps,"
    "mean_reward,std_reward,training_time_s"
)

_ROW_NO_RESULTS = 'exp_001,CartPole-v1,"[64, 64]",0.0003,256,0.99,2048,0.01,100000,,,'

_ROW_WITH_RESULTS = (
    'exp_002,CartPole-v1,"[16, 16]",0.0003,256,0.99,2048,0.01,100000,450.0,25.0,123.45'
)

_ROW_INVALID_BATCH = 'exp_003,CartPole-v1,"[64, 64]",0.0003,300,0.99,512,0.01,100000,,,'


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def csv_valid(tmp_path: Path) -> Path:
    """Zwróć ścieżkę do tymczasowego CSV z dwoma poprawnymi wierszami.

    Pierwszy wiersz ma puste kolumny wynikowe (``None``), drugi
    posiada wypełnione wartości liczbowe.

    Parameters
    ----------
    tmp_path : Path
        Wbudowana w pytest ścieżka do katalogu tymczasowego.

    Returns
    -------
    Path
        Ścieżka do utworzonego pliku CSV.
    """
    csv_file = tmp_path / "experiments.csv"
    csv_file.write_text(
        "\n".join([_CSV_HEADER, _ROW_NO_RESULTS, _ROW_WITH_RESULTS]),
        encoding="utf-8",
    )
    return csv_file


@pytest.fixture
def csv_empty(tmp_path: Path) -> Path:
    """Zwróć ścieżkę do tymczasowego CSV zawierającego wyłącznie nagłówek.

    Parameters
    ----------
    tmp_path : Path
        Wbudowana w pytest ścieżka do katalogu tymczasowego.

    Returns
    -------
    Path
        Ścieżka do pliku CSV z samym nagłówkiem (brak wierszy danych).
    """
    csv_file = tmp_path / "experiments.csv"
    csv_file.write_text(_CSV_HEADER, encoding="utf-8")
    return csv_file


@pytest.fixture
def csv_invalid_batch(tmp_path: Path) -> Path:
    """Zwróć ścieżkę do CSV z wierszem naruszającym warunek n_steps % batch_size.

    Wiersz zawiera ``n_steps=512`` i ``batch_size=300``,
    co narusza wymaganie PPO (DKB-003).

    Parameters
    ----------
    tmp_path : Path
        Wbudowana w pytest ścieżka do katalogu tymczasowego.

    Returns
    -------
    Path
        Ścieżka do pliku CSV z niepoprawną kombinacją hiperparametrów.
    """
    csv_file = tmp_path / "experiments.csv"
    csv_file.write_text(
        "\n".join([_CSV_HEADER, _ROW_INVALID_BATCH]),
        encoding="utf-8",
    )
    return csv_file


@pytest.fixture
def csv_for_save(tmp_path: Path) -> Path:
    """Zwróć ścieżkę do CSV z dwoma wierszami używanego w testach zapisu.

    Parameters
    ----------
    tmp_path : Path
        Wbudowana w pytest ścieżka do katalogu tymczasowego.

    Returns
    -------
    Path
        Ścieżka do pliku CSV gotowego do testowania ``save_results``.
    """
    csv_file = tmp_path / "experiments.csv"
    csv_file.write_text(
        "\n".join([_CSV_HEADER, _ROW_NO_RESULTS, _ROW_WITH_RESULTS]),
        encoding="utf-8",
    )
    return csv_file


# ---------------------------------------------------------------------------
# Testy: load_experiments
# ---------------------------------------------------------------------------


def test_load_experiments_valid(csv_valid: Path) -> None:
    """Zweryfikuj poprawny odczyt CSV i konwersję typów.

    Sprawdza konwersję wszystkich typów pól oraz poprawne
    mapowanie pustych kolumn wynikowych na ``None`` i niepustych
    na ``float``.

    Parameters
    ----------
    csv_valid : Path
        Fixture — ścieżka do poprawnego pliku CSV.
    """
    experiments = load_experiments(str(csv_valid))

    assert len(experiments) == 2

    exp1 = experiments[0]
    assert exp1["experiment_id"] == "exp_001"
    assert isinstance(exp1["learning_rate"], float)
    assert isinstance(exp1["gamma"], float)
    assert isinstance(exp1["ent_coef"], float)
    assert isinstance(exp1["batch_size"], int)
    assert isinstance(exp1["n_steps"], int)
    assert isinstance(exp1["total_timesteps"], int)
    assert exp1["mean_reward"] is None
    assert exp1["std_reward"] is None
    assert exp1["training_time_s"] is None

    exp2 = experiments[1]
    assert isinstance(exp2["mean_reward"], float)
    assert isinstance(exp2["std_reward"], float)
    assert isinstance(exp2["training_time_s"], float)
    assert exp2["mean_reward"] == pytest.approx(450.0)
    assert exp2["std_reward"] == pytest.approx(25.0)
    assert exp2["training_time_s"] == pytest.approx(123.45)


def test_load_experiments_file_not_found(tmp_path: Path) -> None:
    """Zweryfikuj rzucenie FileNotFoundError dla nieistniejącego pliku.

    Parameters
    ----------
    tmp_path : Path
        Wbudowana w pytest ścieżka do katalogu tymczasowego.
    """
    missing = str(tmp_path / "nonexistent.csv")
    with pytest.raises(FileNotFoundError):
        load_experiments(missing)


def test_load_experiments_empty_csv(csv_empty: Path) -> None:
    """Zweryfikuj zwrócenie pustej listy dla CSV z samym nagłówkiem.

    Parameters
    ----------
    csv_empty : Path
        Fixture — ścieżka do pliku CSV z wyłącznie nagłówkiem.
    """
    result = load_experiments(str(csv_empty))
    assert result == []


def test_load_experiments_invalid_batch_size(csv_invalid_batch: Path) -> None:
    """Zweryfikuj rzucenie ValueError gdy n_steps nie jest podzielne przez batch_size.

    Parameters
    ----------
    csv_invalid_batch : Path
        Fixture — ścieżka do pliku CSV z naruszonym warunkiem DKB-003.
    """
    with pytest.raises(ValueError, match="n_steps=512"):
        load_experiments(str(csv_invalid_batch))


# ---------------------------------------------------------------------------
# Testy: parse_net_arch
# ---------------------------------------------------------------------------


def test_parse_net_arch_valid_two_layers() -> None:
    """Zweryfikuj parsowanie dwuwarstwowej architektury."""
    result = parse_net_arch("[64, 64]")
    assert result == [64, 64]


def test_parse_net_arch_valid_three_layers() -> None:
    """Zweryfikuj parsowanie trzywarstwowej architektury."""
    result = parse_net_arch("[1024, 1024, 1024]")
    assert result == [1024, 1024, 1024]


def test_parse_net_arch_invalid_string() -> None:
    """Zweryfikuj rzucenie ValueError dla niepoprawnego stringa."""
    with pytest.raises(ValueError, match="Niepoprawny format"):
        parse_net_arch("invalid")


def test_parse_net_arch_not_a_list() -> None:
    """Zweryfikuj rzucenie ValueError gdy literal_eval zwraca nie-listę."""
    with pytest.raises(ValueError, match="musi być listą"):
        parse_net_arch("64")


def test_parse_net_arch_invalid_elements() -> None:
    """Zweryfikuj rzucenie ValueError gdy elementy zawierają wartości ujemne."""
    with pytest.raises(ValueError, match="pozytywnymi intami"):
        parse_net_arch("[64, -1]")


# ---------------------------------------------------------------------------
# Testy: save_results
# ---------------------------------------------------------------------------


def test_save_results_valid(csv_for_save: Path) -> None:
    """Zweryfikuj aktualizację właściwego wiersza i nienaruszalność pozostałych.

    Parameters
    ----------
    csv_for_save : Path
        Fixture — ścieżka do pliku CSV gotowego do testowania zapisu.
    """
    metrics = {"mean_reward": 480.0, "std_reward": 10.5, "training_time_s": 55.3}
    save_results(str(csv_for_save), "exp_001", metrics)

    experiments = load_experiments(str(csv_for_save))

    updated = next(e for e in experiments if e["experiment_id"] == "exp_001")
    assert updated["mean_reward"] == pytest.approx(480.0)
    assert updated["std_reward"] == pytest.approx(10.5)
    assert updated["training_time_s"] == pytest.approx(55.3)

    untouched = next(e for e in experiments if e["experiment_id"] == "exp_002")
    assert untouched["mean_reward"] == pytest.approx(450.0)


def test_save_results_experiment_not_found(csv_for_save: Path) -> None:
    """Zweryfikuj rzucenie ValueError dla nieistniejącego experiment_id.

    Parameters
    ----------
    csv_for_save : Path
        Fixture — ścieżka do pliku CSV gotowego do testowania zapisu.
    """
    metrics = {"mean_reward": 0.0, "std_reward": 0.0, "training_time_s": 0.0}
    with pytest.raises(ValueError, match="nie istnieje"):
        save_results(str(csv_for_save), "exp_999", metrics)
