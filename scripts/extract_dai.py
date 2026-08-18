from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.dai import extract_dai_path  # noqa: E402


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
    seen_model_names: set[str] = set()
    for path in files:
        try:
            record, extracted_samples = extract_dai_path(
                path,
                trust_source=True,
                include_personal_metadata=arguments.include_personal_metadata,
            )
            model_key = str(record["modelo_nome"]).casefold()
            if model_key in seen_model_names:
                raise ValueError(
                    "Identificador de modelo duplicado no lote; mantenha somente uma versão."
                )
            seen_model_names.add(model_key)
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

