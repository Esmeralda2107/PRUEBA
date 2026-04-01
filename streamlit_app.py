from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================
st.set_page_config(
    page_title="TFM_ GRUPO Y SITE SELECTION MANHATTAN",
    page_icon="📍",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "datos" / "maestro" / "MASTER_DATASET_MANHATTAN_ML.csv"
GEOJSON_DIR = BASE_DIR / "datos" / "crudos" / "zonas"


# =========================================================
# ESTILOS
# =========================================================
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 16px 18px;
        min-height: 132px;
    }
    .metric-title {
        font-size: 0.90rem;
        color: #475569;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 1.30rem;
        font-weight: 700;
        color: #0f172a;
        line-height: 1.25;
        word-wrap: break-word;
        overflow-wrap: anywhere;
    }
    .metric-sub {
        font-size: 0.85rem;
        color: #64748b;
        margin-top: 8px;
    }
    .scenario-box {
        background-color: #f1f5f9;
        border-left: 4px solid #0f766e;
        padding: 12px 14px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .summary-box {
        background-color: #f8fafc;
        border: 1px solid #cbd5e1;
        border-radius: 14px;
        padding: 16px;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# METADATOS DEL MODELO
# =========================================================
DIMENSIONS = {
    "DEMANDA": {
        "label": "Censo (Demanda)",
        "variables": {
            "POBLACION_KM2": {
                "label": "Población por km²",
                "weight": 30,
                "sense": "direct",
                "unit": "hab/km²",
            },
            "PORCENTAJE_HISPANOS": {
                "label": "Porcentaje hispanos",
                "weight": 30,
                "sense": "direct",
                "unit": "%",
            },
            "EDAD_MEDIANA": {
                "label": "Edad mediana",
                "weight": 10,
                "sense": "direct",
                "unit": "años",
            },
            "INGRESO_MEDIANO_HOGAR": {
                "label": "Ingreso mediano del hogar",
                "weight": 20,
                "sense": "direct",
                "unit": "USD",
            },
            "TAMANO_HOGAR_PROMEDIO": {
                "label": "Tamaño hogar promedio",
                "weight": 10,
                "sense": "direct",
                "unit": "personas/hogar",
            },
        },
    },
    "MOVILIDAD": {
        "label": "Movilidad",
        "variables": {
            "MOVILIDAD_PROMEDIO_DIARIA": {
                "label": "Movilidad promedio diaria",
                "weight": 70,
                "sense": "direct",
                "unit": "promedio diario",
            },
            "MOV_CANTIDAD_ESTACIONES": {
                "label": "Cantidad de estaciones",
                "weight": 30,
                "sense": "direct",
                "unit": "número",
            },
        },
    },
    "SEGURIDAD": {
        "label": "Seguridad",
        "variables": {
            "DELITO_PROPIEDAD_KM2": {
                "label": "Delito propiedad por km²",
                "weight": 40,
                "sense": "inverse",
                "unit": "incidentes/km²",
            },
            "DELITO_TRANSPORTE_KM2": {
                "label": "Delito transporte por km²",
                "weight": 40,
                "sense": "inverse",
                "unit": "incidentes/km²",
            },
            "DELITO_OTROS_KM2": {
                "label": "Otros delitos por km²",
                "weight": 20,
                "sense": "inverse",
                "unit": "incidentes/km²",
            },
        },
    },
    "PUNTOS_INTERES": {
        "label": "Puntos de interés",
        "variables": {
            "LUGARES_COMERCIO_KM2": {
                "label": "Lugares comercio por km²",
                "weight": 40,
                "sense": "direct",
                "unit": "lugares/km²",
            },
            "LUGARES_OFICINAS_KM2": {
                "label": "Lugares oficinas por km²",
                "weight": 40,
                "sense": "direct",
                "unit": "lugares/km²",
            },
            "LUGARES_RESIDENCIAL_KM2": {
                "label": "Lugares residencial por km²",
                "weight": 20,
                "sense": "direct",
                "unit": "lugares/km²",
            },
        },
    },
    "COMPETENCIA": {
        "label": "Competencia",
        "variables": {
            "COMPETENCIA_DIRECTA_KM2": {
                "label": "Competencia directa por km²",
                "weight": 70,
                "sense": "inverse",
                "unit": "competidores/km²",
            },
            "COMPETENCIA_INDIRECTA_KM2": {
                "label": "Competencia indirecta por km²",
                "weight": 30,
                "sense": "direct",
                "unit": "competidores/km²",
            },
        },
    },
    "COSTE": {
        "label": "Coste",
        "variables": {
            "ALQ_PRECIO_PIE2_ANUAL": {
                "label": "Precio alquiler pie² anual",
                "weight": 100,
                "sense": "inverse",
                "unit": "USD/pie²/año",
            },
        },
    },
}

SCENARIOS = {
    "Potencial de demanda": {
        "description": (
            "Prioriza la capacidad de atracción comercial de la zona. "
            "Las dimensiones principales son demanda y puntos de interés."
        ),
        "weights": {
            "DEMANDA": 35,
            "PUNTOS_INTERES": 35,
            "MOVILIDAD": 10,
            "SEGURIDAD": 10,
            "COSTE": 5,
            "COMPETENCIA": 5,
        },
        "main_dims": ["DEMANDA", "PUNTOS_INTERES"],
        "context_dims": ["MOVILIDAD", "SEGURIDAD", "COSTE", "COMPETENCIA"],
    },
    "Eficiencia y flujo": {
        "description": (
            "Prioriza condiciones urbanas relevantes para un modelo fast casual "
            "orientado al take-away. Las dimensiones principales son movilidad "
            "y puntos de interés."
        ),
        "weights": {
            "MOVILIDAD": 40,
            "PUNTOS_INTERES": 30,
            "DEMANDA": 10,
            "SEGURIDAD": 10,
            "COSTE": 5,
            "COMPETENCIA": 5,
        },
        "main_dims": ["MOVILIDAD", "PUNTOS_INTERES"],
        "context_dims": ["DEMANDA", "SEGURIDAD", "COSTE", "COMPETENCIA"],
    },
    "Viabilidad y riesgo": {
        "description": (
            "Enfatiza estabilidad operativa y económica de la implantación, "
            "además de la saturación competitiva del entorno."
        ),
        "weights": {
            "SEGURIDAD": 25,
            "COSTE": 25,
            "COMPETENCIA": 20,
            "DEMANDA": 10,
            "MOVILIDAD": 10,
            "PUNTOS_INTERES": 10,
        },
        "main_dims": ["SEGURIDAD", "COSTE", "COMPETENCIA"],
        "context_dims": ["DEMANDA", "MOVILIDAD", "PUNTOS_INTERES"],
    },
}


# =========================================================
# FUNCIONES AUXILIARES
# =========================================================
def metric_card(title, value, subtitle=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def clean_zone_id(value):
    if pd.isna(value):
        return None
    return str(value).strip().upper()


def score_0_100(series: pd.Series, sense: str = "direct") -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    s = s.fillna(s.median())

    min_v = s.min()
    max_v = s.max()

    if max_v == min_v:
        scored = pd.Series(50.0, index=s.index)
    else:
        if sense == "direct":
            scored = ((s - min_v) / (max_v - min_v)) * 100
        else:
            scored = ((max_v - s) / (max_v - min_v)) * 100

    return scored.round(2)


def classify_level(score):
    if score >= 67:
        return "Alto"
    if score < 34:
        return "Bajo"
    return "Neutro"


def detect_geojson_id_field(geojson_dict):
    features = geojson_dict.get("features", [])
    if not features:
        return None

    props = features[0].get("properties", {})
    candidates = [
        "NTA2020",
        "nta2020",
        "NTACode",
        "NTA_CODE",
        "nta_code",
        "NTA",
        "nta",
        "id",
        "ID",
    ]

    for cand in candidates:
        if cand in props:
            return cand

    if props:
        return list(props.keys())[0]

    return None


def extract_geojson_ids(geojson_dict, geo_field):
    ids = []
    for feature in geojson_dict.get("features", []):
        props = feature.get("properties", {})
        ids.append(clean_zone_id(props.get(geo_field)))
    return ids


@st.cache_data
def load_data():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"No existe el CSV maestro en: {CSV_PATH}")

    geojson_files = sorted(GEOJSON_DIR.glob("*.geojson"))
    if not geojson_files:
        raise FileNotFoundError(
            f"No se encontró ningún .geojson dentro de: {GEOJSON_DIR}"
        )

    df = pd.read_csv(CSV_PATH)

    with open(geojson_files[0], "r", encoding="utf-8") as f:
        geojson = json.load(f)

    return df, geojson, geojson_files[0].name


def validate_columns(df):
    required = ["ID_ZONA", "NOMBRE_ZONA"]
    for dim_meta in DIMENSIONS.values():
        required.extend(list(dim_meta["variables"].keys()))
    return [c for c in required if c not in df.columns]


def compute_dimension_scores(df):
    out = df.copy()

    for dim_key, dim_meta in DIMENSIONS.items():
        contrib_cols = []

        for var, var_meta in dim_meta["variables"].items():
            out[var] = pd.to_numeric(out[var], errors="coerce")
            out[var] = out[var].fillna(out[var].median())

            score_col = f"SCORE_VAR_{var}"
            contrib_col = f"CONTRIB_DIMVAR_{var}"

            out[score_col] = score_0_100(out[var], var_meta["sense"])
            out[contrib_col] = out[score_col] * (var_meta["weight"] / 100)
            contrib_cols.append(contrib_col)

        out[f"SCORE_DIM_{dim_key}"] = out[contrib_cols].sum(axis=1).round(2)

    return out


def normalize_bucket_weights(raw_weights, total_bucket):
    raw_sum = sum(raw_weights.values())
    if raw_sum == 0:
        equal = total_bucket / len(raw_weights)
        return {k: equal for k in raw_weights.keys()}

    normalized = {}
    for k, v in raw_weights.items():
        normalized[k] = round((v / raw_sum) * total_bucket, 2)

    diff = round(total_bucket - sum(normalized.values()), 2)
    last_key = list(normalized.keys())[-1]
    normalized[last_key] = round(normalized[last_key] + diff, 2)
    return normalized


def build_effective_scenario_weights(scenario_name, main_raw, context_raw):
    scenario = SCENARIOS[scenario_name]
    main_weights = normalize_bucket_weights(main_raw, 70)
    context_weights = normalize_bucket_weights(context_raw, 30)

    effective = {}
    for dim in scenario["main_dims"]:
        effective[dim] = main_weights[dim]
    for dim in scenario["context_dims"]:
        effective[dim] = context_weights[dim]

    return effective


def compute_scenario_scores(df, scenario_weights):
    out = df.copy()

    dim_contrib_cols = []

    for dim_key, dim_weight in scenario_weights.items():
        dim_score_col = f"SCORE_DIM_{dim_key}"
        dim_contrib_col = f"CONTRIB_SCEN_DIM_{dim_key}"

        out[dim_contrib_col] = out[dim_score_col] * (dim_weight / 100)
        dim_contrib_cols.append(dim_contrib_col)

        for var, var_meta in DIMENSIONS[dim_key]["variables"].items():
            var_score_col = f"SCORE_VAR_{var}"
            var_contrib_col = f"CONTRIB_SCEN_VAR_{var}"
            out[var_contrib_col] = (
                out[var_score_col]
                * (var_meta["weight"] / 100)
                * (dim_weight / 100)
            ).round(4)

    out["SCORE_ESCENARIO"] = out[dim_contrib_cols].sum(axis=1).round(2)
    out["RANK"] = out["SCORE_ESCENARIO"].rank(
        ascending=False, method="dense"
    ).astype(int)

    return out.sort_values("SCORE_ESCENARIO", ascending=False)


@st.cache_data
def compute_clusters(df, feature_cols):
    x = df[feature_cols].copy()
    for col in feature_cols:
        x[col] = pd.to_numeric(x[col], errors="coerce")
        x[col] = x[col].fillna(x[col].median())

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = km.fit_predict(x_scaled)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(x_scaled)

    return clusters, coords[:, 0], coords[:, 1]


def get_top_subdimensions(row, top_n=3):
    items = []
    for dim_meta in DIMENSIONS.values():
        for var, var_meta in dim_meta["variables"].items():
            items.append(
                {
                    "variable": var,
                    "label": var_meta["label"],
                    "dimension": next(
                        d["label"]
                        for d in DIMENSIONS.values()
                        if var in d["variables"]
                    ),
                    "score": row[f"SCORE_VAR_{var}"],
                    "contrib": row[f"CONTRIB_SCEN_VAR_{var}"],
                }
            )
    items = sorted(items, key=lambda x: x["contrib"], reverse=True)
    return items[:top_n]


def get_dimension_summary(row):
    summaries = []
    for dim_key, dim_meta in DIMENSIONS.items():
        dim_score = row[f"SCORE_DIM_{dim_key}"]
        level = classify_level(dim_score)

        best_var = None
        best_val = -1
        for var, var_meta in dim_meta["variables"].items():
            value = row[f"CONTRIB_DIMVAR_{var}"]
            if value > best_val:
                best_val = value
                best_var = (var, var_meta["label"], row[f"SCORE_VAR_{var}"])

        summaries.append(
            {
                "dimension": dim_meta["label"],
                "score": dim_score,
                "level": level,
                "best_subdim": best_var[1],
                "best_subdim_score": round(best_var[2], 2),
            }
        )
    return summaries


def build_multiindex_ranking(df, scenario_name):
    context_cols = [
        "RANK",
        "ID_ZONA",
        "NOMBRE_ZONA",
        "CLUSTER_K4",
        "SCORE_ESCENARIO",
    ]
    dim_cols = [
        "SCORE_DIM_DEMANDA",
        "SCORE_DIM_MOVILIDAD",
        "SCORE_DIM_SEGURIDAD",
        "SCORE_DIM_PUNTOS_INTERES",
        "SCORE_DIM_COMPETENCIA",
        "SCORE_DIM_COSTE",
    ]

    subdim_cols = []
    for dim_meta in DIMENSIONS.values():
        subdim_cols.extend(list(dim_meta["variables"].keys()))

    rank_df = df[context_cols + dim_cols + subdim_cols].copy()
    rank_df["ESCENARIO"] = scenario_name
    rank_df = rank_df[
        ["RANK", "ID_ZONA", "NOMBRE_ZONA", "CLUSTER_K4", "ESCENARIO", "SCORE_ESCENARIO"]
        + dim_cols
        + subdim_cols
    ]

    col_tuples = [
        ("Contexto", "Rank"),
        ("Contexto", "ID zona"),
        ("Contexto", "Zona"),
        ("Contexto", "Cluster K=4"),
        ("Contexto", "Escenario"),
        ("Contexto", "Score escenario"),
        ("Dimensiones", "Demanda"),
        ("Dimensiones", "Movilidad"),
        ("Dimensiones", "Seguridad"),
        ("Dimensiones", "Puntos de interés"),
        ("Dimensiones", "Competencia"),
        ("Dimensiones", "Coste"),
    ]

    for dim_meta in DIMENSIONS.values():
        for var, var_meta in dim_meta["variables"].items():
            col_tuples.append(("Subdimensiones", var_meta["label"]))

    rank_df.columns = pd.MultiIndex.from_tuples(col_tuples)
    return rank_df


# =========================================================
# CARGA Y PREPARACIÓN
# =========================================================
st.title("TFM_ GRUPO Y SITE SELECTION MANHATTAN")
st.caption("Aplicación interactiva para scoring multicriterio, escenarios de decisión y análisis territorial.")

try:
    df, geojson, geojson_filename = load_data()
except Exception as e:
    st.error(str(e))
    st.stop()

missing_cols = validate_columns(df)
if missing_cols:
    st.error(f"Faltan columnas necesarias en el CSV maestro: {missing_cols}")
    st.stop()

df["ID_ZONA"] = df["ID_ZONA"].apply(clean_zone_id)
df["NOMBRE_ZONA"] = df["NOMBRE_ZONA"].astype(str).str.strip()

geo_field = detect_geojson_id_field(geojson)
if geo_field is None:
    st.error("No se pudo detectar el campo ID del GeoJSON.")
    st.stop()

geojson_ids = extract_geojson_ids(geojson, geo_field)
geojson_id_set = set([x for x in geojson_ids if x is not None])

all_feature_cols = []
for dim_meta in DIMENSIONS.values():
    all_feature_cols.extend(list(dim_meta["variables"].keys()))

df = compute_dimension_scores(df)

clusters, pca1, pca2 = compute_clusters(df, all_feature_cols)
df["CLUSTER_K4"] = (clusters + 1).astype(int)
df["PCA_1"] = pca1
df["PCA_2"] = pca2


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("Escenario y filtros")

scenario_name = st.sidebar.selectbox(
    "Escenario de decisión",
    options=list(SCENARIOS.keys()),
)

scenario = SCENARIOS[scenario_name]

st.sidebar.markdown(
    f"""
    <div class="scenario-box">
        <strong>{scenario_name}</strong><br>
        {scenario["description"]}
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("### Reparto macro 70/30")
st.sidebar.caption(
    "Puedes ajustar la distribución interna, pero las dimensiones principales siempre suman 70 % y las de contexto 30 %."
)

default_main_raw = {}
default_context_raw = {}

for dim in scenario["main_dims"]:
    default_main_raw[dim] = scenario["weights"][dim] / 70 * 100

for dim in scenario["context_dims"]:
    default_context_raw[dim] = scenario["weights"][dim] / 30 * 100

st.sidebar.markdown("**Dimensiones principales (suman 70%)**")
main_raw = {}
for dim in scenario["main_dims"]:
    main_raw[dim] = st.sidebar.slider(
        f"{DIMENSIONS[dim]['label']} | reparto interno %",
        min_value=0,
        max_value=100,
        value=int(round(default_main_raw[dim])),
        step=5,
        key=f"main_{scenario_name}_{dim}",
    )

st.sidebar.markdown("**Dimensiones de contexto (suman 30%)**")
context_raw = {}
for dim in scenario["context_dims"]:
    context_raw[dim] = st.sidebar.slider(
        f"{DIMENSIONS[dim]['label']} | reparto interno %",
        min_value=0,
        max_value=100,
        value=int(round(default_context_raw[dim])),
        step=5,
        key=f"context_{scenario_name}_{dim}",
    )

effective_weights = build_effective_scenario_weights(
    scenario_name, main_raw, context_raw
)

weights_preview = pd.DataFrame(
    {
        "Dimensión": [DIMENSIONS[d]["label"] for d in effective_weights.keys()],
        "Peso efectivo (%)": list(effective_weights.values()),
        "Tipo": [
            "Principal" if d in scenario["main_dims"] else "Contexto"
            for d in effective_weights.keys()
        ],
    }
)
st.sidebar.dataframe(weights_preview, use_container_width=True, hide_index=True)

scenario_scored = compute_scenario_scores(df, effective_weights)

st.sidebar.markdown("### Filtros")

score_range = st.sidebar.slider(
    "Score del escenario (puntos 0–100)",
    min_value=0.0,
    max_value=100.0,
    value=(
        float(scenario_scored["SCORE_ESCENARIO"].min()),
        float(scenario_scored["SCORE_ESCENARIO"].max()),
    ),
)

rent_range = st.sidebar.slider(
    "Alquiler (USD/pie²/año)",
    min_value=float(scenario_scored["ALQ_PRECIO_PIE2_ANUAL"].min()),
    max_value=float(scenario_scored["ALQ_PRECIO_PIE2_ANUAL"].max()),
    value=(
        float(scenario_scored["ALQ_PRECIO_PIE2_ANUAL"].min()),
        float(scenario_scored["ALQ_PRECIO_PIE2_ANUAL"].max()),
    ),
)

comp_dir_range = st.sidebar.slider(
    "Competencia directa (competidores/km²)",
    min_value=float(scenario_scored["COMPETENCIA_DIRECTA_KM2"].min()),
    max_value=float(scenario_scored["COMPETENCIA_DIRECTA_KM2"].max()),
    value=(
        float(scenario_scored["COMPETENCIA_DIRECTA_KM2"].min()),
        float(scenario_scored["COMPETENCIA_DIRECTA_KM2"].max()),
    ),
)

mob_range = st.sidebar.slider(
    "Movilidad promedio diaria",
    min_value=float(scenario_scored["MOVILIDAD_PROMEDIO_DIARIA"].min()),
    max_value=float(scenario_scored["MOVILIDAD_PROMEDIO_DIARIA"].max()),
    value=(
        float(scenario_scored["MOVILIDAD_PROMEDIO_DIARIA"].min()),
        float(scenario_scored["MOVILIDAD_PROMEDIO_DIARIA"].max()),
    ),
)

income_range = st.sidebar.slider(
    "Ingreso mediano del hogar (USD)",
    min_value=float(scenario_scored["INGRESO_MEDIANO_HOGAR"].min()),
    max_value=float(scenario_scored["INGRESO_MEDIANO_HOGAR"].max()),
    value=(
        float(scenario_scored["INGRESO_MEDIANO_HOGAR"].min()),
        float(scenario_scored["INGRESO_MEDIANO_HOGAR"].max()),
    ),
)

cluster_options = sorted(scenario_scored["CLUSTER_K4"].dropna().unique().tolist())
selected_clusters = st.sidebar.multiselect(
    "Cluster K=4",
    options=cluster_options,
    default=cluster_options,
)

zone_options = sorted(scenario_scored["NOMBRE_ZONA"].dropna().unique().tolist())
selected_zones = st.sidebar.multiselect(
    "Zonas filtradas",
    options=zone_options,
    default=zone_options,
)

filtered = scenario_scored[
    scenario_scored["SCORE_ESCENARIO"].between(score_range[0], score_range[1])
    & scenario_scored["ALQ_PRECIO_PIE2_ANUAL"].between(rent_range[0], rent_range[1])
    & scenario_scored["COMPETENCIA_DIRECTA_KM2"].between(comp_dir_range[0], comp_dir_range[1])
    & scenario_scored["MOVILIDAD_PROMEDIO_DIARIA"].between(mob_range[0], mob_range[1])
    & scenario_scored["INGRESO_MEDIANO_HOGAR"].between(income_range[0], income_range[1])
    & scenario_scored["CLUSTER_K4"].isin(selected_clusters)
    & scenario_scored["NOMBRE_ZONA"].isin(selected_zones)
].copy()

if filtered.empty:
    st.warning("No hay zonas que cumplan los filtros actuales.")
    st.stop()

filtered = filtered.sort_values("SCORE_ESCENARIO", ascending=False).reset_index(drop=True)
filtered["RANK"] = filtered["SCORE_ESCENARIO"].rank(ascending=False, method="dense").astype(int)

best_zone = filtered.iloc[0]
top_subdims = get_top_subdimensions(best_zone, top_n=3)
top_subdim_text = " · ".join([x["label"] for x in top_subdims])


# =========================================================
# KPIs
# =========================================================
c1, c2, c3, c4 = st.columns([2.2, 1.1, 1.1, 1.9])

with c1:
    metric_card(
        "Mejor zona",
        best_zone["NOMBRE_ZONA"],
        f"ID: {best_zone['ID_ZONA']}",
    )

with c2:
    metric_card(
        "Score del escenario",
        f"{best_zone['SCORE_ESCENARIO']:.2f}",
        scenario_name,
    )

with c3:
    metric_card(
        "Zonas visibles",
        f"{len(filtered)}",
        "Tras aplicar filtros",
    )

with c4:
    metric_card(
        "Subdimensiones dominantes",
        top_subdim_text,
        "Top 3 variables con mayor aporte al score de la mejor zona",
    )


# =========================================================
# LECTURA RÁPIDA DE LA MEJOR ZONA
# =========================================================
dimension_summary = get_dimension_summary(best_zone)

summary_lines = []
for item in dimension_summary:
    summary_lines.append(
        f"- **{item['dimension']}**: {item['level']} "
        f"({item['score']:.1f}/100). "
        f"Subdimensión más fuerte: **{item['best_subdim']}** "
        f"({item['best_subdim_score']:.1f}/100)."
    )

dominant_lines = []
for item in top_subdims:
    dominant_lines.append(
        f"- **{item['label']}** ({item['dimension']}): "
        f"{item['score']:.1f}/100, aporte escenario {item['contrib']:.2f}."
    )

st.markdown('<div class="summary-box">', unsafe_allow_html=True)
st.markdown(f"### Lectura rápida de la mejor zona: **{best_zone['NOMBRE_ZONA']}**")
st.markdown(
    f"**Escenario activo:** {scenario_name}. "
    f"**Score final:** {best_zone['SCORE_ESCENARIO']:.2f}/100. "
    f"Esta lectura interpreta el comportamiento de la zona con los pesos actualmente aplicados."
)
st.markdown("**Resumen por dimensiones**")
st.markdown("\n".join(summary_lines))
st.markdown("**Subdimensiones más determinantes en este escenario**")
st.markdown("\n".join(dominant_lines))
st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["🗺️ Mapa", "🏆 Ranking", "📊 Gráficos", "📘 Metodología", "🧪 Diagnóstico"]
)


# =========================================================
# TAB MAPA
# =========================================================
with tab1:
    st.subheader("Mapa interactivo por escenario y dimensiones")

    map_options = ["SCORE_ESCENARIO"] + [f"SCORE_DIM_{d}" for d in DIMENSIONS.keys()]
    option_labels = {
        "SCORE_ESCENARIO": f"Score final - {scenario_name}",
        "SCORE_DIM_DEMANDA": "Censo (Demanda)",
        "SCORE_DIM_MOVILIDAD": "Movilidad",
        "SCORE_DIM_SEGURIDAD": "Seguridad",
        "SCORE_DIM_PUNTOS_INTERES": "Puntos de interés",
        "SCORE_DIM_COMPETENCIA": "Competencia",
        "SCORE_DIM_COSTE": "Coste",
    }

    map_metric = st.selectbox(
        "Variable a visualizar en el mapa",
        options=map_options,
        format_func=lambda x: option_labels[x],
    )

    map_df = filtered.copy()
    map_df["ID_ZONA"] = map_df["ID_ZONA"].apply(clean_zone_id)
    map_df = map_df[map_df["ID_ZONA"].isin(geojson_id_set)].copy()

    if map_df.empty:
        st.error("No hay coincidencias entre los IDs filtrados y el GeoJSON.")
        st.stop()

    fig_map = px.choropleth_mapbox(
        map_df,
        geojson=geojson,
        locations="ID_ZONA",
        featureidkey=f"properties.{geo_field}",
        color=map_metric,
        hover_name="NOMBRE_ZONA",
        hover_data={
            "ID_ZONA": True,
            "CLUSTER_K4": True,
            "SCORE_ESCENARIO": ":.2f",
            "RANK": True,
            "SCORE_DIM_DEMANDA": ":.2f",
            "SCORE_DIM_MOVILIDAD": ":.2f",
            "SCORE_DIM_SEGURIDAD": ":.2f",
            "SCORE_DIM_PUNTOS_INTERES": ":.2f",
            "SCORE_DIM_COMPETENCIA": ":.2f",
            "SCORE_DIM_COSTE": ":.2f",
        },
        mapbox_style="carto-positron",
        center={"lat": 40.7831, "lon": -73.9712},
        zoom=10.4,
        opacity=0.78,
    )

    fig_map.update_layout(
        height=720,
        margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_colorbar_title="Puntos",
    )

    st.plotly_chart(fig_map, use_container_width=True)
    st.caption(f"Join cartográfico: ID_ZONA ↔ properties.{geo_field}")


# =========================================================
# TAB RANKING
# =========================================================
with tab2:
    st.subheader("Top 10 zonas según filtros")

    ranking_view = build_multiindex_ranking(filtered.head(10), scenario_name)
    st.dataframe(
        ranking_view,
        use_container_width=True,
        hide_index=True,
    )

    ranking_download = build_multiindex_ranking(filtered, scenario_name).copy()
    ranking_download.columns = [
        f"{a} | {b}" for a, b in ranking_download.columns.to_list()
    ]
    csv_download = ranking_download.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        label="⬇️ Descargar ranking filtrado",
        data=csv_download,
        file_name="ranking_site_selection_manhattan.csv",
        mime="text/csv",
    )


# =========================================================
# TAB GRÁFICOS
# =========================================================
with tab3:
    st.subheader("Gráficos de apoyo a la decisión")

    # 1) Top 10 por score
    top10_chart = filtered.head(10).copy()
    top10_chart["score_txt"] = top10_chart["SCORE_ESCENARIO"].round(1)

    fig_top10 = px.bar(
        top10_chart.sort_values("SCORE_ESCENARIO", ascending=True),
        x="SCORE_ESCENARIO",
        y="NOMBRE_ZONA",
        orientation="h",
        text="score_txt",
        color="SCORE_ESCENARIO",
        title=f"Top 10 zonas por score - {scenario_name}",
    )
    fig_top10.update_layout(height=500, margin=dict(l=0, r=0, t=50, b=0))
    st.plotly_chart(fig_top10, use_container_width=True)

    col_a, col_b = st.columns(2)

    # 2) Clustering k=4
    with col_a:
        fig_cluster = px.scatter(
            filtered,
            x="PCA_1",
            y="PCA_2",
            color=filtered["CLUSTER_K4"].astype(str),
            size="SCORE_ESCENARIO",
            hover_name="NOMBRE_ZONA",
            title="Clustering K-means k=4 (provisional)",
            labels={"color": "Cluster"},
        )
        fig_cluster.update_layout(height=460, margin=dict(l=0, r=0, t=50, b=0))
        st.plotly_chart(fig_cluster, use_container_width=True)

    # 3) Dispersión útil
    with col_b:
        fig_scatter = px.scatter(
            filtered,
            x="ALQ_PRECIO_PIE2_ANUAL",
            y="MOVILIDAD_PROMEDIO_DIARIA",
            color="SCORE_ESCENARIO",
            size="SCORE_ESCENARIO",
            hover_name="NOMBRE_ZONA",
            title="Alquiler (USD/pie²/año) vs movilidad promedio diaria",
        )
        fig_scatter.update_layout(height=460, margin=dict(l=0, r=0, t=50, b=0))
        st.plotly_chart(fig_scatter, use_container_width=True)

    # 4) Comparativa por dimensiones
    st.markdown("### Comparativa dimensional del Top 5")

    dim_cols = [
        "SCORE_DIM_DEMANDA",
        "SCORE_DIM_MOVILIDAD",
        "SCORE_DIM_SEGURIDAD",
        "SCORE_DIM_PUNTOS_INTERES",
        "SCORE_DIM_COMPETENCIA",
        "SCORE_DIM_COSTE",
    ]

    top5_dims = filtered.head(5)[["NOMBRE_ZONA"] + dim_cols].copy()
    rename_dim_cols = {
        "SCORE_DIM_DEMANDA": "Demanda",
        "SCORE_DIM_MOVILIDAD": "Movilidad",
        "SCORE_DIM_SEGURIDAD": "Seguridad",
        "SCORE_DIM_PUNTOS_INTERES": "Puntos de interés",
        "SCORE_DIM_COMPETENCIA": "Competencia",
        "SCORE_DIM_COSTE": "Coste",
    }
    top5_dims = top5_dims.rename(columns=rename_dim_cols)

    dims_melt = top5_dims.melt(
        id_vars="NOMBRE_ZONA",
        var_name="Dimensión",
        value_name="Puntuación",
    )

    fig_dims = px.bar(
        dims_melt,
        x="Dimensión",
        y="Puntuación",
        color="NOMBRE_ZONA",
        barmode="group",
        title="Desempeño por dimensiones del Top 5",
    )
    fig_dims.update_layout(height=520, margin=dict(l=0, r=0, t=50, b=0))
    st.plotly_chart(fig_dims, use_container_width=True)


# =========================================================
# TAB METODOLOGÍA
# =========================================================
with tab4:
    st.subheader("Metodología implementada")

    st.markdown(
        """
**Estructura del modelo**
- Nivel micro: ponderación local de variables dentro de cada dimensión.
- Nivel macro: combinación de dimensiones según el escenario de decisión.
- Escala final: 0 a 100 puntos.

**Regla macro**
- Dimensiones principales: **70 %**
- Dimensiones de contexto: **30 %**
- La app permite ajustar el reparto interno, manteniendo fijo ese marco.

**Dimensiones**
- Censo (Demanda)
- Movilidad
- Seguridad
- Puntos de interés
- Competencia
- Coste
"""
    )

    st.markdown("### Pesos locales por dimensión")
    local_rows = []
    for dim_key, dim_meta in DIMENSIONS.items():
        for var, var_meta in dim_meta["variables"].items():
            local_rows.append(
                {
                    "Dimensión": dim_meta["label"],
                    "Subdimensión": var_meta["label"],
                    "Variable": var,
                    "Peso local (%)": var_meta["weight"],
                    "Sentido": "Directo" if var_meta["sense"] == "direct" else "Inverso",
                    "Unidad": var_meta["unit"],
                }
            )
    local_df = pd.DataFrame(local_rows)
    st.dataframe(local_df, use_container_width=True, hide_index=True)

    st.markdown("### Escenarios y pesos efectivos actuales")
    macro_rows = []
    for dim_key, weight in effective_weights.items():
        macro_rows.append(
            {
                "Escenario": scenario_name,
                "Dimensión": DIMENSIONS[dim_key]["label"],
                "Peso efectivo (%)": weight,
                "Tipo": "Principal" if dim_key in scenario["main_dims"] else "Contexto",
            }
        )
    macro_df = pd.DataFrame(macro_rows)
    st.dataframe(macro_df, use_container_width=True, hide_index=True)

    st.markdown("### Zonas por cluster (K-means k=4 provisional)")
    cluster_list_df = (
        df.sort_values(["CLUSTER_K4", "NOMBRE_ZONA"])
        .groupby("CLUSTER_K4")["NOMBRE_ZONA"]
        .apply(list)
        .reset_index()
    )

    for _, row in cluster_list_df.iterrows():
        cluster_id = row["CLUSTER_K4"]
        zone_list = row["NOMBRE_ZONA"]
        with st.expander(f"Cluster {cluster_id} | {len(zone_list)} zonas"):
            st.write(", ".join(zone_list))

    st.markdown("### Fórmulas")
    st.latex(r"ScoreVar_{i,j} = \left(\frac{x_{i,j}-\min(x_j)}{\max(x_j)-\min(x_j)}\right)\cdot 100")
    st.markdown("Para variables de sentido directo.")
    st.latex(r"ScoreVar_{i,j} = \left(\frac{\max(x_j)-x_{i,j}}{\max(x_j)-\min(x_j)}\right)\cdot 100")
    st.markdown("Para variables de sentido inverso.")
    st.latex(r"ScoreDim_{i,d} = \sum_j w_{j|d}\cdot ScoreVar_{i,j}")
    st.latex(r"ScoreEscenario_{i,s} = \sum_d w_{d|s}\cdot ScoreDim_{i,d}")

    st.info(
        "El gráfico de clustering y el listado por cluster se muestran como aproximación provisional con K-means k=4. "
        "Cuando dispongas del clustering definitivo, solo tendrás que sustituir esa asignación."
    )


# =========================================================
# TAB DIAGNÓSTICO
# =========================================================
with tab5:
    st.subheader("Diagnóstico técnico")

    csv_id_set = set(df["ID_ZONA"].dropna().unique())
    missing_in_geojson = sorted(csv_id_set - geojson_id_set)
    missing_in_csv = sorted(geojson_id_set - csv_id_set)

    st.write("Archivo GeoJSON cargado:", geojson_filename)
    st.write("Campo detectado en GeoJSON:", geo_field)
    st.write("Total IDs CSV:", len(csv_id_set))
    st.write("Total IDs GeoJSON:", len(geojson_id_set))
    st.write("IDs del CSV no encontrados en el GeoJSON:", missing_in_geojson[:25])
    st.write("IDs del GeoJSON no encontrados en el CSV:", missing_in_csv[:25])

    features = geojson.get("features", [])
    if features:
        st.write("Propiedades del primer feature:")
        st.json(features[0].get("properties", {}))
