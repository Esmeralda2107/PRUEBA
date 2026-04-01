from pathlib import Path
import json
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
    page_title="TFM GRUPO 7 SITE SELECTION MANHATTAN",
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
        font-size: 1.25rem;
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
        margin-top: 14px;
        margin-bottom: 10px;
    }
    .group-title {
        margin-top: 18px;
        margin-bottom: 8px;
        font-weight: 700;
        color: #0f172a;
        font-size: 1.03rem;
    }
    .custom-table table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.92rem;
    }
    .custom-table th, .custom-table td {
        border: 1px solid #e2e8f0;
        padding: 8px 10px;
        text-align: left !important;
        vertical-align: top;
    }
    .custom-table th {
        background-color: #f1f5f9;
        font-weight: 700;
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
            "POBLACION_KM2": {"label": "Población por km²", "weight": 30, "sense": "direct"},
            "PORCENTAJE_HISPANOS": {"label": "Porcentaje hispanos", "weight": 30, "sense": "direct"},
            "EDAD_MEDIANA": {"label": "Edad mediana", "weight": 10, "sense": "direct"},
            "INGRESO_MEDIANO_HOGAR": {"label": "Ingreso mediano del hogar", "weight": 20, "sense": "direct"},
            "TAMANO_HOGAR_PROMEDIO": {"label": "Tamaño hogar promedio", "weight": 10, "sense": "direct"},
        },
    },
    "MOVILIDAD": {
        "label": "Movilidad",
        "variables": {
            "MOVILIDAD_PROMEDIO_DIARIA": {"label": "Flujo de personas (promedio diario)", "weight": 70, "sense": "direct"},
            "MOV_CANTIDAD_ESTACIONES": {"label": "Cantidad de estaciones", "weight": 30, "sense": "direct"},
        },
    },
    "SEGURIDAD": {
        "label": "Seguridad",
        "variables": {
            "DELITO_PROPIEDAD_KM2": {"label": "Delito propiedad por km²", "weight": 40, "sense": "inverse"},
            "DELITO_TRANSPORTE_KM2": {"label": "Delito transporte por km²", "weight": 40, "sense": "inverse"},
            "DELITO_OTROS_KM2": {"label": "Otros delitos por km²", "weight": 20, "sense": "inverse"},
        },
    },
    "PUNTOS_INTERES": {
        "label": "Puntos de interés",
        "variables": {
            "LUGARES_COMERCIO_KM2": {"label": "Lugares comercio por km²", "weight": 40, "sense": "direct"},
            "LUGARES_OFICINAS_KM2": {"label": "Lugares oficinas por km²", "weight": 40, "sense": "direct"},
            "LUGARES_RESIDENCIAL_KM2": {"label": "Lugares residencial por km²", "weight": 20, "sense": "direct"},
        },
    },
    "COMPETENCIA": {
        "label": "Competencia",
        "variables": {
            "COMPETENCIA_DIRECTA_KM2": {"label": "Competencia directa por km²", "weight": 70, "sense": "inverse"},
            "COMPETENCIA_INDIRECTA_KM2": {"label": "Competencia indirecta por km²", "weight": 30, "sense": "direct"},
        },
    },
    "COSTE": {
        "label": "Coste",
        "variables": {
            "ALQ_PRECIO_PIE2_ANUAL": {"label": "Precio alquiler pie² anual", "weight": 100, "sense": "inverse"},
        },
    },
}

SCENARIOS = {
    "Potencial de demanda": {
        "description": (
            "Prioriza las dimensiones más vinculadas con la capacidad de atracción comercial de la zona. "
            "Bajo esta lógica, el 70 % del peso total se concentra en Censo (Demanda) y Puntos de interés, "
            "mientras que el 30 % restante se reparte entre Movilidad, Seguridad, Coste y Competencia como factores de contexto."
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
            "Da mayor peso a las condiciones urbanas más relevantes para un modelo fast casual orientado al take-away. "
            "En este escenario, el 70 % se concentra en Movilidad y Puntos de interés, "
            "mientras que el 30 % restante se distribuye entre Censo (Demanda), Seguridad, Coste y Competencia."
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
            "Enfatiza los factores que inciden con mayor fuerza en la estabilidad operativa y económica de la implantación, "
            "así como en la saturación competitiva del entorno. En este escenario, el 70 % del peso se concentra en Seguridad, "
            "Coste y Competencia, mientras que el 30 % restante se reparte entre Censo (Demanda), Movilidad y Puntos de interés."
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

CLUSTER_DESCRIPTORS = {
    "POBLACION_KM2": "alta densidad poblacional",
    "PORCENTAJE_HISPANOS": "alta población hispana",
    "EDAD_MEDIANA": "edad media madura",
    "INGRESO_MEDIANO_HOGAR": "alto ingreso",
    "TAMANO_HOGAR_PROMEDIO": "hogares más grandes",
    "MOVILIDAD_PROMEDIO_DIARIA": "alto flujo de personas",
    "MOV_CANTIDAD_ESTACIONES": "alta conectividad",
    "DELITO_PROPIEDAD_KM2": "menor delito patrimonial",
    "DELITO_TRANSPORTE_KM2": "menor delito en transporte",
    "DELITO_OTROS_KM2": "menor delincuencia general",
    "LUGARES_COMERCIO_KM2": "actividad comercial",
    "LUGARES_OFICINAS_KM2": "concentración de oficinas",
    "LUGARES_RESIDENCIAL_KM2": "entorno residencial",
    "COMPETENCIA_DIRECTA_KM2": "baja competencia directa",
    "COMPETENCIA_INDIRECTA_KM2": "alta competencia indirecta",
    "ALQ_PRECIO_PIE2_ANUAL": "coste de alquiler contenido",
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
        return pd.Series(50.0, index=s.index)

    if sense == "direct":
        out = ((s - min_v) / (max_v - min_v)) * 100
    else:
        out = ((max_v - s) / (max_v - min_v)) * 100

    return out.round(2)


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
    candidates = ["NTA2020", "nta2020", "NTACode", "NTA_CODE", "nta_code", "NTA", "nta", "id", "ID"]

    for cand in candidates:
        if cand in props:
            return cand

    return list(props.keys())[0] if props else None


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
        raise FileNotFoundError(f"No se encontró ningún .geojson dentro de: {GEOJSON_DIR}")

    df = pd.read_csv(CSV_PATH)

    with open(geojson_files[0], "r", encoding="utf-8") as f:
        geojson = json.load(f)

    return df, geojson


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
                out[var_score_col] * (var_meta["weight"] / 100) * (dim_weight / 100)
            ).round(4)

    out["SCORE_ESCENARIO"] = out[dim_contrib_cols].sum(axis=1).round(2)
    out["RANK"] = out["SCORE_ESCENARIO"].rank(ascending=False, method="dense").astype(int)
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


def build_cluster_names(df):
    score_cols = [f"SCORE_VAR_{v}" for dim in DIMENSIONS.values() for v in dim["variables"].keys()]
    cluster_names = {}

    overall_means = df[score_cols].mean()

    for cluster_id in sorted(df["CLUSTER_K4"].unique()):
        sub = df[df["CLUSTER_K4"] == cluster_id]
        cluster_means = sub[score_cols].mean()
        lift = (cluster_means - overall_means).sort_values(ascending=False)

        top_vars = []
        for col in lift.index:
            raw_var = col.replace("SCORE_VAR_", "")
            if raw_var in CLUSTER_DESCRIPTORS:
                top_vars.append(CLUSTER_DESCRIPTORS[raw_var])
            if len(top_vars) == 2:
                break

        if len(top_vars) == 0:
            cluster_names[cluster_id] = f"Cluster {cluster_id}"
        elif len(top_vars) == 1:
            cluster_names[cluster_id] = top_vars[0].capitalize()
        else:
            cluster_names[cluster_id] = f"{top_vars[0].capitalize()} y {top_vars[1]}"

    return cluster_names


def get_top_subdimensions(row, top_n=3):
    items = []
    for dim_key, dim_meta in DIMENSIONS.items():
        for var, var_meta in dim_meta["variables"].items():
            items.append(
                {
                    "label": var_meta["label"],
                    "dimension": dim_meta["label"],
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

        best_label = None
        best_contrib = -1
        for var, var_meta in dim_meta["variables"].items():
            contrib = row[f"CONTRIB_SCEN_VAR_{var}"]
            if contrib > best_contrib:
                best_contrib = contrib
                best_label = var_meta["label"]

        summaries.append(
            {
                "dimension": dim_meta["label"],
                "score": dim_score,
                "level": level,
                "best_subdim": best_label,
            }
        )
    return summaries


def fmt_num(x, decimals=2):
    if pd.isna(x):
        return ""
    return f"{x:,.{decimals}f}"


def fmt_int(x):
    if pd.isna(x):
        return ""
    return f"{int(x):,}"


def render_html_table(df):
    html = '<div class="custom-table">' + df.to_html(index=False, escape=False) + "</div>"
    st.markdown(html, unsafe_allow_html=True)


def build_grouped_context(df, scenario_name):
    out = df[["RANK", "ID_ZONA", "NOMBRE_ZONA", "CLUSTER_NAME", "SCORE_ESCENARIO"]].head(10).copy()
    out["ESCENARIO"] = scenario_name
    out = out.rename(columns={
        "RANK": "Rank",
        "ID_ZONA": "ID zona",
        "NOMBRE_ZONA": "Zona",
        "CLUSTER_NAME": "Cluster",
        "SCORE_ESCENARIO": "Score escenario",
        "ESCENARIO": "Escenario",
    })

    out["Rank"] = out["Rank"].apply(fmt_int)
    out["Score escenario"] = out["Score escenario"].apply(lambda x: fmt_num(x, 2))
    return out


def build_grouped_dimensions(df):
    out = df[[
        "RANK",
        "NOMBRE_ZONA",
        "SCORE_DIM_DEMANDA",
        "SCORE_DIM_MOVILIDAD",
        "SCORE_DIM_SEGURIDAD",
        "SCORE_DIM_PUNTOS_INTERES",
        "SCORE_DIM_COMPETENCIA",
        "SCORE_DIM_COSTE",
    ]].head(10).copy()

    out = out.rename(columns={
        "RANK": "Rank",
        "NOMBRE_ZONA": "Zona",
        "SCORE_DIM_DEMANDA": "Demanda",
        "SCORE_DIM_MOVILIDAD": "Movilidad",
        "SCORE_DIM_SEGURIDAD": "Seguridad",
        "SCORE_DIM_PUNTOS_INTERES": "Puntos de interés",
        "SCORE_DIM_COMPETENCIA": "Competencia",
        "SCORE_DIM_COSTE": "Coste",
    })

    out["Rank"] = out["Rank"].apply(fmt_int)
    for col in ["Demanda", "Movilidad", "Seguridad", "Puntos de interés", "Competencia", "Coste"]:
        out[col] = out[col].apply(lambda x: fmt_num(x, 2))

    return out


def build_grouped_subdimensions(df):
    cols = ["RANK", "NOMBRE_ZONA"]
    rename_map = {"RANK": "Rank", "NOMBRE_ZONA": "Zona"}

    for dim_meta in DIMENSIONS.values():
        for var, var_meta in dim_meta["variables"].items():
            cols.append(var)
            rename_map[var] = var_meta["label"]

    out = df[cols].head(10).copy().rename(columns=rename_map)

    out["Rank"] = out["Rank"].apply(fmt_int)

    for col in out.columns:
        if col not in ["Rank", "Zona"]:
            out[col] = out[col].apply(lambda x: fmt_num(x, 2))

    return out


def sequential_bucket_sliders(sidebar, dims, labels, total, min_each, defaults, key_prefix):
    """
    Genera sliders secuenciales.
    La última dimensión se ajusta automáticamente para cerrar el total.
    """
    values = {}
    remaining_total = total
    remaining_dims = len(dims)

    for i, dim in enumerate(dims):
        label = labels[dim]

        if i == len(dims) - 1:
            values[dim] = remaining_total
            sidebar.info(f"{label}: {remaining_total}% (ajuste automático)")
        else:
            min_allowed = min_each
            max_allowed = remaining_total - min_each * (remaining_dims - 1)
            default_val = int(round(defaults.get(dim, min_allowed)))
            default_val = max(min_allowed, min(default_val, max_allowed))

            val = sidebar.slider(
                label,
                min_value=min_allowed,
                max_value=max_allowed,
                value=default_val,
                step=1,
                key=f"{key_prefix}_{dim}",
            )
            values[dim] = val
            remaining_total -= val
            remaining_dims -= 1

    return values


# =========================================================
# CARGA Y PREPARACIÓN
# =========================================================
st.title("TFM GRUPO 7 SITE SELECTION MANHATTAN")
st.caption("Aplicación interactiva para scoring multicriterio, escenarios de decisión y análisis territorial.")

try:
    df, geojson = load_data()
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

cluster_names = build_cluster_names(df)
df["CLUSTER_NAME"] = df["CLUSTER_K4"].map(cluster_names)


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

st.sidebar.markdown("### Distribución automática 70/30")
st.sidebar.caption(
    "Las dimensiones principales siempre suman 70% y las de contexto 30%. "
    "Las últimas dimensiones de cada bloque se ajustan automáticamente."
)

default_main_raw = {}
default_context_raw = {}

for dim in scenario["main_dims"]:
    default_main_raw[dim] = scenario["weights"][dim]

for dim in scenario["context_dims"]:
    default_context_raw[dim] = scenario["weights"][dim]

st.sidebar.markdown("**Dimensiones principales (mínimo 15%)**")
main_weights = sequential_bucket_sliders(
    sidebar=st.sidebar,
    dims=scenario["main_dims"],
    labels={k: DIMENSIONS[k]["label"] for k in scenario["main_dims"]},
    total=70,
    min_each=15,
    defaults=default_main_raw,
    key_prefix=f"main_{scenario_name}",
)

st.sidebar.markdown("**Dimensiones de contexto (mínimo 5%)**")
context_weights = sequential_bucket_sliders(
    sidebar=st.sidebar,
    dims=scenario["context_dims"],
    labels={k: DIMENSIONS[k]["label"] for k in scenario["context_dims"]},
    total=30,
    min_each=5,
    defaults=default_context_raw,
    key_prefix=f"context_{scenario_name}",
)

effective_weights = {**main_weights, **context_weights}
scenario_scored = compute_scenario_scores(df, effective_weights)

st.sidebar.markdown("### Filtros")

score_range = st.sidebar.slider(
    "Score del escenario (0–100)",
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

competition_range = st.sidebar.slider(
    "Competencia directa (competidores/km²)",
    min_value=float(scenario_scored["COMPETENCIA_DIRECTA_KM2"].min()),
    max_value=float(scenario_scored["COMPETENCIA_DIRECTA_KM2"].max()),
    value=(
        float(scenario_scored["COMPETENCIA_DIRECTA_KM2"].min()),
        float(scenario_scored["COMPETENCIA_DIRECTA_KM2"].max()),
    ),
)

flow_range = st.sidebar.slider(
    "Flujo de personas (promedio diario)",
    min_value=float(scenario_scored["MOVILIDAD_PROMEDIO_DIARIA"].min()),
    max_value=float(scenario_scored["MOVILIDAD_PROMEDIO_DIARIA"].max()),
    value=(
        float(scenario_scored["MOVILIDAD_PROMEDIO_DIARIA"].min()),
        float(scenario_scored["MOVILIDAD_PROMEDIO_DIARIA"].max()),
    ),
)

selected_zones = st.sidebar.multiselect(
    "Zonas",
    options=sorted(scenario_scored["NOMBRE_ZONA"].dropna().unique().tolist()),
    default=sorted(scenario_scored["NOMBRE_ZONA"].dropna().unique().tolist()),
)

selected_clusters = st.sidebar.multiselect(
    "Clusters",
    options=sorted(df["CLUSTER_NAME"].dropna().unique().tolist()),
    default=sorted(df["CLUSTER_NAME"].dropna().unique().tolist()),
)

filtered = scenario_scored[
    scenario_scored["SCORE_ESCENARIO"].between(score_range[0], score_range[1])
    & scenario_scored["ALQ_PRECIO_PIE2_ANUAL"].between(rent_range[0], rent_range[1])
    & scenario_scored["COMPETENCIA_DIRECTA_KM2"].between(competition_range[0], competition_range[1])
    & scenario_scored["MOVILIDAD_PROMEDIO_DIARIA"].between(flow_range[0], flow_range[1])
    & scenario_scored["NOMBRE_ZONA"].isin(selected_zones)
    & scenario_scored["CLUSTER_NAME"].isin(selected_clusters)
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
# KPIS
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
        "Top 3 variables con mayor aporte",
    )


# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3, tab4 = st.tabs(
    ["🗺️ Mapa", "🏆 Ranking", "📊 Gráficos", "📘 Metodología"]
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
            "CLUSTER_NAME": True,
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

    # Descripción debajo del mapa
    dimension_summary = get_dimension_summary(best_zone)
    summary_lines = []
    for item in dimension_summary:
        summary_lines.append(
            f"- **{item['dimension']}**: {item['level']} ({item['score']:.1f}/100). "
            f"Subdimensión más representativa: **{item['best_subdim']}**."
        )

    st.markdown('<div class="summary-box">', unsafe_allow_html=True)
    st.markdown(f"### Descripción de la mejor zona: **{best_zone['NOMBRE_ZONA']}**")
    st.markdown(
        f"En el escenario **{scenario_name}**, esta zona obtiene un score de "
        f"**{best_zone['SCORE_ESCENARIO']:.2f}/100** con la distribución actual de pesos."
    )
    st.markdown("**Resumen por dimensiones**")
    st.markdown("\n".join(summary_lines))
    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# TAB RANKING
# =========================================================
with tab2:
    st.subheader("Top 10 zonas según filtros")

    st.markdown('<div class="group-title">Contexto</div>', unsafe_allow_html=True)
    render_html_table(build_grouped_context(filtered, scenario_name))

    st.markdown('<div class="group-title">Dimensiones</div>', unsafe_allow_html=True)
    render_html_table(build_grouped_dimensions(filtered))

    st.markdown('<div class="group-title">Subdimensiones</div>', unsafe_allow_html=True)
    render_html_table(build_grouped_subdimensions(filtered))

    csv_download = filtered.to_csv(index=False).encode("utf-8-sig")
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

    with col_a:
        fig_cluster = px.scatter(
            filtered,
            x="PCA_1",
            y="PCA_2",
            color="CLUSTER_NAME",
            size="SCORE_ESCENARIO",
            hover_name="NOMBRE_ZONA",
            title="Clusters K-means k=4",
            labels={
                "PCA_1": "Componente 1 del clustering",
                "PCA_2": "Componente 2 del clustering",
                "CLUSTER_NAME": "Cluster",
            }
        )
        fig_cluster.update_layout(height=460, margin=dict(l=0, r=0, t=50, b=0))
        st.plotly_chart(fig_cluster, use_container_width=True)

    with col_b:
        fig_scatter = px.scatter(
            filtered,
            x="ALQ_PRECIO_PIE2_ANUAL",
            y="MOVILIDAD_PROMEDIO_DIARIA",
            color="SCORE_ESCENARIO",
            size="SCORE_ESCENARIO",
            hover_name="NOMBRE_ZONA",
            title="Alquiler vs flujo de personas",
            labels={
                "ALQ_PRECIO_PIE2_ANUAL": "Alquiler (USD/pie²/año)",
                "MOVILIDAD_PROMEDIO_DIARIA": "Flujo de personas (promedio diario)",
            }
        )
        fig_scatter.update_layout(height=460, margin=dict(l=0, r=0, t=50, b=0))
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("### Comparativa dimensional del Top 5")

    top5_dims = filtered.head(5)[[
        "NOMBRE_ZONA",
        "SCORE_DIM_DEMANDA",
        "SCORE_DIM_MOVILIDAD",
        "SCORE_DIM_SEGURIDAD",
        "SCORE_DIM_PUNTOS_INTERES",
        "SCORE_DIM_COMPETENCIA",
        "SCORE_DIM_COSTE",
    ]].copy()

    top5_dims = top5_dims.rename(columns={
        "NOMBRE_ZONA": "Zona",
        "SCORE_DIM_DEMANDA": "Demanda",
        "SCORE_DIM_MOVILIDAD": "Movilidad",
        "SCORE_DIM_SEGURIDAD": "Seguridad",
        "SCORE_DIM_PUNTOS_INTERES": "Puntos de interés",
        "SCORE_DIM_COMPETENCIA": "Competencia",
        "SCORE_DIM_COSTE": "Coste",
    })

    dims_melt = top5_dims.melt(
        id_vars="Zona",
        var_name="Dimensión",
        value_name="Puntuación",
    )

    fig_dims = px.bar(
        dims_melt,
        x="Dimensión",
        y="Puntuación",
        color="Zona",
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

**Sentido de las variables**
- **Sentido directo**: valores altos representan una condición más favorable para el negocio.
- **Sentido inverso**: valores altos representan una condición menos favorable, por lo que reciben menor puntuación.

**Regla macro**
- Dimensiones principales: **70 %**
- Dimensiones de contexto: **30 %**
- La app permite modificar solo el reparto interno dentro de cada bloque, respetando siempre esa restricción.

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
                    "Peso local (%)": f"{var_meta['weight']}%",
                    "Sentido": "Directo" if var_meta["sense"] == "direct" else "Inverso",
                }
            )
    render_html_table(pd.DataFrame(local_rows))

    st.markdown("### Zonas por cluster")
    cluster_list_df = (
        df.sort_values(["CLUSTER_K4", "NOMBRE_ZONA"])
        .groupby(["CLUSTER_K4", "CLUSTER_NAME"])["NOMBRE_ZONA"]
        .apply(list)
        .reset_index()
    )

    for _, row in cluster_list_df.iterrows():
        with st.expander(f"{row['CLUSTER_NAME']} | {len(row['NOMBRE_ZONA'])} zonas"):
            st.write(", ".join(row["NOMBRE_ZONA"]))

    st.markdown("### Lectura del gráfico de clustering")
    st.markdown(
        """
- **Componente 1 del clustering** y **Componente 2 del clustering** son ejes sintéticos obtenidos mediante PCA.
- Su función es **visualizar** en dos dimensiones la proximidad entre zonas.
- No representan una variable única del dataset, sino una combinación de variables utilizadas para facilitar la lectura del agrupamiento.
"""
    )

    st.markdown("### Fórmulas")
    st.latex(r"ScoreVar_{i,j} = \left(\frac{x_{i,j}-\min(x_j)}{\max(x_j)-\min(x_j)}\right)\cdot 100")
    st.markdown("Para variables de sentido directo.")
    st.latex(r"ScoreVar_{i,j} = \left(\frac{\max(x_j)-x_{i,j}}{\max(x_j)-\min(x_j)}\right)\cdot 100")
    st.markdown("Para variables de sentido inverso.")
    st.latex(r"ScoreDim_{i,d} = \sum_j w_{j|d}\cdot ScoreVar_{i,j}")
    st.latex(r"ScoreEscenario_{i,s} = \sum_d w_{d|s}\cdot ScoreDim_{i,d}")
