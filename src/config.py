from __future__ import annotations

import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DEMO_DIR = ROOT_DIR / "data" / "demo"
PRIVATE_DIR = ROOT_DIR / "data" / "private"

# Em implantação pública, defina ALLOW_DAI_UPLOADS=false e use catálogos pré-extraídos.
ALLOW_DAI_UPLOADS = os.getenv("ALLOW_DAI_UPLOADS", "true").strip().casefold() in {
    "1",
    "true",
    "sim",
    "yes",
}

DEMO_DB = DEMO_DIR / "trabalhos_demo.sqlite3"
DEMO_CATALOG = DEMO_DIR / "catalogo_modelos_demo.csv"
DEMO_SAMPLES = DEMO_DIR / "amostras_modelos_demo.csv.gz"

# Limites amplos de Porto Alegre usados somente para detectar coordenadas invertidas.
POA_LONGITUDE_RANGE = (-51.40, -50.95)
POA_LATITUDE_RANGE = (-30.35, -29.80)

TIPOLOGIA_LABELS = {
    "AP": "Apartamento",
    "BOX": "Box/estacionamento",
    "CCOM": "Conjunto comercial",
    "DEM HAB": "Demolição/habitação",
    "DEP": "Depósito",
    "EDIF": "Edificação",
    "LCOM": "Loja/conjunto comercial",
    "LOCOM": "Locação comercial",
    "LOJA": "Loja",
    "PCOM": "Prédio comercial",
    "RCOND": "Residência em condomínio",
    "RECOND": "Residência em condomínio",
    "RES": "Residência",
    "SALA": "Sala comercial",
    "TCOND": "Terreno em condomínio",
    "TER": "Terreno",
}

TIPO_TRABALHO_LABELS = {
    "LA": "Laudo de Avaliação",
    "PT": "Parecer Técnico",
    "IT": "Informação Técnica",
    "PTF": "Parecer Técnico Fundamentado",
    "PIV": "Parecer Indicativo de Valor",
}

# Mantido para compatibilidade com notebooks que importam a configuração antiga.
SCORE_WEIGHTS = {
    "demanda": 0.35,
    "recencia": 0.25,
    "suporte": 0.25,
    "catalogo": 0.15,
}

TRIAGE_RULE_VERSION = "2.0"
RECENT_DEMAND_WINDOW_MONTHS = 12
DEMAND_REFERENCE_WORKS = 20
MIN_SPATIAL_SAMPLE = 3
MIN_SPATIAL_DIAGNOSTIC_WORKS = 10
SPATIAL_RELATIVE_RISK_START = 1.0
SPATIAL_RELATIVE_RISK_FULL = 3.0

# O risco de desempenho permanece sem valor até existir base de teste identificada.
TRIAGE_EVIDENCE_WEIGHTS = {
    "impacto": 0.40,
    "desempenho": 0.35,
    "suporte": 0.15,
    "operacional": 0.10,
}

HF_REPO_ID = os.getenv(
    "HF_REPO_ID",
    "gui-sparim/repositorio_mesa",
).strip()

HF_REPO_REVISION = os.getenv(
    "HF_REPO_REVISION",
    "main",
).strip()

# Confirmação administrativa para permitir a abertura dos .dai baixados.
TRUST_HF_DAI = os.getenv(
    "TRUST_HF_DAI",
    "false",
).strip().casefold() in {
    "1",
    "true",
    "sim",
    "yes",
}