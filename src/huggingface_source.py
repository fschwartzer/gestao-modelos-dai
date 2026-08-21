from __future__ import annotations

from pathlib import Path


DOWNLOAD_PATTERNS = [
    "*.sqlite",
    "*.sqlite3",
    "*.db",
    "*.dai",
    "*.DAI",
    "*.csv",
    "*.csv.gz",
    "**/*.sqlite",
    "**/*.sqlite3",
    "**/*.db",
    "**/*.dai",
    "**/*.DAI",
    "**/catalogo_modelos.csv",
    "**/amostras_modelos.csv.gz",
]


def download_hf_snapshot(
    *,
    repo_id: str,
    revision: str,
    token: str | None,
) -> Path:
    # Importação tardia: os modos demonstração/upload continuam funcionando
    # mesmo quando a dependência opcional ainda não foi instalada no ambiente.
    from huggingface_hub import snapshot_download

    snapshot_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        token=token,
        allow_patterns=DOWNLOAD_PATTERNS,
    )
    return Path(snapshot_path)


def _find_unique_named_file(root: Path, filename: str) -> Path | None:
    matches = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name.casefold() == filename.casefold()
    )

    if len(matches) > 1:
        raise ValueError(
            f"Há mais de um arquivo chamado {filename} no repositório."
        )

    return matches[0] if matches else None


def locate_hf_sources(
    root: Path,
) -> tuple[Path | None, tuple[Path, ...], Path | None, Path | None]:
    sqlite_files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.casefold() in {".sqlite", ".sqlite3", ".db"}
    )

    preferred_db = [
        path
        for path in sqlite_files
        if path.name.casefold() == "trabalhos_tecnicos.sqlite3"
    ]

    if preferred_db:
        db_path = preferred_db[0]
    elif len(sqlite_files) == 1:
        db_path = sqlite_files[0]
    elif len(sqlite_files) > 1:
        raise ValueError(
            "Foram encontrados vários bancos SQLite e nenhum se chama "
            "'trabalhos_tecnicos.sqlite3'."
        )
    else:
        db_path = None

    dai_paths = tuple(
        sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.casefold() == ".dai"
        )
    )

    catalog_path = _find_unique_named_file(
        root,
        "catalogo_modelos.csv",
    )
    samples_path = _find_unique_named_file(
        root,
        "amostras_modelos.csv.gz",
    )

    return db_path, dai_paths, catalog_path, samples_path
