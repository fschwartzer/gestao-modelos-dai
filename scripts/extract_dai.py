from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def canonical_name(path: Path) -> str:
    return re.sub(r"\(\d+\)$", "", path.stem).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def python_scalar(value: object) -> object:
    return value.item() if isinstance(value, np.generic) else value


def extract_one(path: Path, include_personal_metadata: bool) -> tuple[dict, list[dict]]:
    package = joblib.load(path)
    if not isinstance(package, dict):
        raise TypeError("O pacote não contém um dicionário no nível raiz.")

    data_block = package.get("dados", {}) or {}
    transformations = package.get("transformacoes", {}) or {}
    model_block = package.get("modelo", {}) or {}
    diagnostics = model_block.get("diagnosticos", {}) or {}
    general = diagnostics.get("gerais", {}) or {}
    period = package.get("periodo_dados_mercado", {}) or {}
    appraisal = package.get("avaliacao", {}) or {}
    author = package.get("elaborador", {}) or {}

    model_frame = data_block.get("df")
    complete_frame = data_block.get("df_completo")
    excluded = data_block.get("outliers_excluidos", [])
    predictors = transformations.get("X")
    target = transformations.get("y")

    n_model = len(model_frame) if isinstance(model_frame, pd.DataFrame) else None
    n_complete = len(complete_frame) if isinstance(complete_frame, pd.DataFrame) else n_model
    n_outliers = len(excluded) if hasattr(excluded, "__len__") else None
    pct_outliers = (
        100 * n_outliers / n_complete
        if n_complete and n_outliers is not None
        else None
    )

    record = {
        "modelo_nome": canonical_name(path),
        "arquivo": path.name,
        "artifact_sha256": sha256_file(path),
        "artifact_size_bytes": path.stat().st_size,
        "artifact_mtime": pd.Timestamp(path.stat().st_mtime, unit="s").isoformat(),
        "versao_formato": package.get("versao"),
        "data_inicial": period.get("data_inicial"),
        "data_final": period.get("data_final"),
        "coluna_data": period.get("coluna_data"),
        "n_modelo": n_model,
        "n_completo": n_complete,
        "n_outliers": n_outliers,
        "pct_outliers": pct_outliers,
        "variavel_alvo": getattr(target, "name", None),
        "preditoras_json": json.dumps(
            list(predictors.columns) if isinstance(predictors, pd.DataFrame) else [],
            ensure_ascii=False,
        ),
        "transformacoes_json": json.dumps(transformations.get("info", []), ensure_ascii=False),
        "r2": python_scalar(general.get("r2")),
        "r2_ajustado": python_scalar(general.get("r2_ajustado")),
        "desvio_padrao_residuos": python_scalar(general.get("desvio_padrao_residuos")),
        "mse": python_scalar(general.get("mse")),
        "equacao": diagnostics.get("equacao"),
        "tipo_y": appraisal.get("tipo_y"),
        "coluna_area": appraisal.get("coluna_area"),
        "elaborador_nome": author.get("nome_completo") if include_personal_metadata else None,
        "elaborador_lotacao": author.get("lotacao") if include_personal_metadata else None,
        "status": "a_classificar",
    }

    samples: list[dict] = []
    if isinstance(model_frame, pd.DataFrame) and {"lat", "lon"}.issubset(model_frame.columns):
        coordinates = model_frame[["lat", "lon"]].copy()
        coordinates["latitude"] = pd.to_numeric(coordinates["lat"], errors="coerce")
        coordinates["longitude"] = pd.to_numeric(coordinates["lon"], errors="coerce")
        date_column = period.get("coluna_data")
        dates = (
            pd.to_datetime(model_frame[date_column], errors="coerce")
            if date_column in model_frame.columns
            else pd.Series(pd.NaT, index=model_frame.index)
        )
        for sequence, (index, row) in enumerate(coordinates.dropna().iterrows(), start=1):
            value_date = dates.loc[index] if index in dates.index else pd.NaT
            samples.append(
                {
                    "modelo_nome": canonical_name(path),
                    "sample_id": f"{canonical_name(path)}-{sequence:05d}",
                    "latitude": float(row["latitude"]),
                    "longitude": float(row["longitude"]),
                    "data_ref": None if pd.isna(value_date) else value_date.date().isoformat(),
                }
            )
    return record, samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extrai metadados e coordenadas não pessoais de arquivos .dai confiáveis."
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="Pasta com os arquivos .dai")
    parser.add_argument("--output-dir", type=Path, required=True, help="Pasta de saída")
    parser.add_argument(
        "--trust-source",
        action="store_true",
        help="Confirma que os .dai são internos e confiáveis. Pickle pode executar código.",
    )
    parser.add_argument(
        "--include-personal-metadata",
        action="store_true",
        help="Inclui nome e lotação do elaborador. Desativado por padrão.",
    )
    parser.add_argument(
        "--omit-samples",
        action="store_true",
        help="Não exporta as coordenadas das amostras.",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    if not arguments.trust_source:
        print(
            "Execução interrompida: use --trust-source somente se os .dai forem de origem confiável.",
            file=sys.stderr,
        )
        return 2
    files = sorted(arguments.input_dir.glob("*.dai"))
    if not files:
        print("Nenhum arquivo .dai encontrado.", file=sys.stderr)
        return 1

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    catalog: list[dict] = []
    samples: list[dict] = []
    errors: list[dict] = []
    for path in files:
        try:
            record, extracted_samples = extract_one(path, arguments.include_personal_metadata)
            catalog.append(record)
            if not arguments.omit_samples:
                samples.extend(extracted_samples)
            print(f"OK  {path.name}")
        except Exception as error:  # mantém o lote e registra incompatibilidades
            errors.append({"arquivo": path.name, "erro": f"{type(error).__name__}: {error}"})
            print(f"ERRO {path.name}: {error}", file=sys.stderr)

    pd.DataFrame(catalog).to_csv(arguments.output_dir / "catalogo_modelos.csv", index=False)
    if not arguments.omit_samples:
        pd.DataFrame(samples).to_csv(
            arguments.output_dir / "amostras_modelos.csv.gz",
            index=False,
            compression="gzip",
        )
    pd.DataFrame(errors, columns=["arquivo", "erro"]).to_csv(
        arguments.output_dir / "erros_extracao.csv", index=False
    )
    print(f"\nModelos extraídos: {len(catalog)} | Erros: {len(errors)}")
    return 0 if catalog else 1


if __name__ == "__main__":
    raise SystemExit(main())

