from __future__ import annotations

import gzip
import sqlite3
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "demo"
RNG = np.random.default_rng(20260813)


MODELS = [
    ("MOD_V_TER_Z1_DEMO_001", "V_TER", "Z1", -30.035, -51.220, 2019, 2025, 0.88),
    ("MOD_V_TER_Z2_DEMO_002", "V_TER", "Z2", -30.015, -51.145, 2021, 2025, 0.91),
    ("MOD_V_TER_Z3_DEMO_001", "V_TER", "Z3", -30.080, -51.145, 2020, 2024, 0.84),
    ("MOD_V_TER_Z4_DEMO_003", "V_TER", "Z4", -30.145, -51.220, 2018, 2023, 0.86),
    ("MOD_V_TER_Z5_DEMO_001", "V_TER", "Z5", -30.205, -51.175, 2019, 2024, 0.80),
    ("MOD_V_AP_Z1_DEMO_004", "V_AP", "Z1", -30.040, -51.205, 2020, 2026, 0.76),
    ("MOD_V_AP_Z3_DEMO_002", "V_AP", "Z3", -30.075, -51.155, 2021, 2025, 0.79),
    ("MOD_V_TCOND_Z4_DEMO_001", "V_TCOND", "Z4", -30.170, -51.205, 2018, 2024, 0.83),
    ("MOD_A_CCOM_Z1_DEMO_002", "A_CCOM", "Z1", -30.030, -51.225, 2024, 2025, 0.78),
    ("MOD_A_EDIF_Z1_DEMO_001", "A_EDIF", "Z1", -30.045, -51.215, 2024, 2025, 0.82),
    ("MOD_A_LOJA_Z2_DEMO_003", "A_LOJA", "Z2", -30.010, -51.165, 2023, 2025, 0.87),
    ("MOD_A_SALA_Z1_DEMO_002", "A_SALA", "Z1", -30.032, -51.220, 2024, 2025, 0.90),
    ("MOD_V_SALA_Z1_DEMO_001", "V_SALA", "Z1", -30.028, -51.218, 2020, 2024, 0.73),
    ("MOD_V_RCOND_Z4_DEMO_002", "V_RCOND", "Z4", -30.155, -51.210, 2021, 2025, 0.81),
]

FAMILY_WEIGHTS = {
    "V_TER": 0.56,
    "V_AP": 0.13,
    "V_TCOND": 0.07,
    "A_CCOM": 0.05,
    "A_EDIF": 0.04,
    "A_LOJA": 0.04,
    "A_SALA": 0.035,
    "V_SALA": 0.035,
    "V_RCOND": 0.04,
}

TYPE_CODES = ["LA", "PT", "IT", "PTF", "PIV"]
TYPE_LABELS = {
    "LA": "Laudo de Avaliação",
    "PT": "Parecer Técnico",
    "IT": "Informação Técnica",
    "PTF": "Parecer Técnico Fundamentado",
    "PIV": "Parecer Indicativo de Valor",
}


def create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE trabalhos (
            trabalho_id TEXT PRIMARY KEY,
            nome TEXT NOT NULL,
            nome_original TEXT NOT NULL,
            tipo_codigo TEXT NOT NULL,
            tipo_label TEXT NOT NULL,
            ano INTEGER,
            endereco_principal TEXT,
            numero_principal TEXT,
            endereco_resumo TEXT,
            modelo_resumo TEXT,
            total_registros INTEGER NOT NULL,
            total_imoveis INTEGER NOT NULL,
            total_modelos INTEGER NOT NULL,
            tem_coordenadas INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE trabalho_imoveis (
            imovel_id INTEGER PRIMARY KEY AUTOINCREMENT,
            trabalho_id TEXT NOT NULL,
            endereco TEXT,
            numero TEXT,
            label TEXT NOT NULL,
            coord_x REAL,
            coord_y REAL
        );
        CREATE TABLE trabalho_imovel_modelos (
            imovel_id INTEGER NOT NULL,
            modelo_nome TEXT NOT NULL,
            PRIMARY KEY (imovel_id, modelo_nome)
        );
        CREATE TABLE trabalho_modelos (
            trabalho_id TEXT NOT NULL,
            modelo_nome TEXT NOT NULL,
            ordem INTEGER NOT NULL,
            PRIMARY KEY (trabalho_id, modelo_nome)
        );
        CREATE INDEX idx_demo_ano ON trabalhos (ano);
        CREATE INDEX idx_demo_modelo ON trabalho_modelos (modelo_nome);
        """
    )


def choose_model() -> tuple:
    families = list(FAMILY_WEIGHTS)
    family = RNG.choice(families, p=[FAMILY_WEIGHTS[key] for key in families])
    candidates = [model for model in MODELS if model[1] == family]
    return candidates[int(RNG.integers(0, len(candidates)))]


def generate_work_data(connection: sqlite3.Connection, n_works: int = 650) -> None:
    year_values = np.arange(2019, 2027)
    year_weights = np.array([0.07, 0.13, 0.13, 0.15, 0.15, 0.17, 0.15, 0.05])
    type_weights = [0.68, 0.14, 0.08, 0.09, 0.01]

    for index in range(1, n_works + 1):
        model = choose_model()
        model_name, family, zone, center_lat, center_lon, *_ = model
        year = int(RNG.choice(year_values, p=year_weights))
        type_code = str(RNG.choice(TYPE_CODES, p=type_weights))
        work_id = f"{type_code}_DEMO_{index:04d}_{year}"
        address = f"Endereço sintético {index:04d}"
        number = str(10 + index * 3)
        latitude = float(RNG.normal(center_lat, 0.016 if family == "V_TER" else 0.009))
        longitude = float(RNG.normal(center_lon, 0.019 if family == "V_TER" else 0.011))

        # Pequena parcela invertida para demonstrar a validação automática.
        if RNG.random() < 0.035:
            coord_x, coord_y = latitude, longitude
        else:
            coord_x, coord_y = longitude, latitude

        linked_models = [model_name]
        if RNG.random() < 0.085:
            same_family = [item[0] for item in MODELS if item[1] == family and item[0] != model_name]
            if same_family:
                linked_models.append(str(RNG.choice(same_family)))

        connection.execute(
            """
            INSERT INTO trabalhos VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                work_id,
                work_id,
                work_id,
                type_code,
                TYPE_LABELS[type_code],
                year,
                address,
                number,
                f"{address}, {number}",
                model_name if len(linked_models) == 1 else f"{len(linked_models)} modelos vinculados",
                len(linked_models),
                1,
                len(linked_models),
                1,
            ),
        )
        cursor = connection.execute(
            """
            INSERT INTO trabalho_imoveis
            (trabalho_id, endereco, numero, label, coord_x, coord_y)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (work_id, address, number, f"{address}, {number}", coord_x, coord_y),
        )
        property_id = int(cursor.lastrowid)
        for order, linked_model in enumerate(linked_models, start=1):
            connection.execute(
                "INSERT INTO trabalho_modelos VALUES (?, ?, ?)",
                (work_id, linked_model, order),
            )
            connection.execute(
                "INSERT INTO trabalho_imovel_modelos VALUES (?, ?)",
                (property_id, linked_model),
            )
    connection.commit()


def generate_catalog_and_samples() -> tuple[pd.DataFrame, pd.DataFrame]:
    catalog_records: list[dict[str, object]] = []
    sample_records: list[dict[str, object]] = []
    today = date(2026, 8, 13)
    for sequence, model in enumerate(MODELS, start=1):
        model_name, family, zone, center_lat, center_lon, start_year, end_year, r2 = model
        sample_size = int(RNG.integers(55, 340))
        excluded = int(RNG.integers(0, max(2, sample_size // 5)))
        data_start = date(start_year, int(RNG.integers(1, 7)), int(RNG.integers(1, 25)))
        data_end = date(end_year, int(RNG.integers(6, 13)), int(RNG.integers(1, 25)))
        catalog_records.append(
            {
                "modelo_nome": model_name,
                "arquivo": f"{model_name}.dai",
                "artifact_sha256": f"demo-{sequence:04d}",
                "versao_formato": 2,
                "data_inicial": data_start.isoformat(),
                "data_final": data_end.isoformat(),
                "n_modelo": sample_size,
                "n_completo": sample_size + excluded,
                "n_outliers": excluded,
                "pct_outliers": round(100 * excluded / (sample_size + excluded), 2),
                "variavel_alvo": "VTOTAL" if family == "V_TER" else "VUNIT",
                "preditoras_json": '["AREA", "RH", "ANO"]',
                "r2_ajustado": r2,
                "tipo_y": "total" if family in {"V_TER", "A_EDIF"} else "unitario",
                "coluna_area": "ATTOTAL" if family == "V_TER" else "APRIV",
                "equacao": "Equação sintética para demonstração",
                "artifact_mtime": today.isoformat(),
                "status": "demonstração",
                "zona_declarada": zone,
            }
        )
        for sample_index in range(sample_size):
            data_ref = data_start + timedelta(
                days=int(RNG.integers(0, max(1, (data_end - data_start).days + 1)))
            )
            spread = 0.020 if family == "V_TER" else 0.010
            sample_records.append(
                {
                    "modelo_nome": model_name,
                    "sample_id": f"{sequence:02d}-{sample_index + 1:04d}",
                    "latitude": float(RNG.normal(center_lat, spread)),
                    "longitude": float(RNG.normal(center_lon, spread * 1.15)),
                    "data_ref": data_ref.isoformat(),
                }
            )
    return pd.DataFrame(catalog_records), pd.DataFrame(sample_records)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    database_path = OUTPUT / "trabalhos_demo.sqlite3"
    if database_path.exists():
        database_path.unlink()
    connection = sqlite3.connect(database_path)
    try:
        create_schema(connection)
        generate_work_data(connection)
    finally:
        connection.close()

    catalog, samples = generate_catalog_and_samples()
    catalog.to_csv(OUTPUT / "catalogo_modelos_demo.csv", index=False)
    samples.to_csv(OUTPUT / "amostras_modelos_demo.csv.gz", index=False, compression="gzip")
    print(f"Demo criado em {OUTPUT}")


if __name__ == "__main__":
    main()

