from pathlib import Path
import json
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
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
    .custom-table thead tr:nth-child(1) th {
        text-align: center !important;
        background-color: #e2e8f0;
    }
    .custom-table thead tr:nth-child(2) th {
        text-align: left !important;
        background-color: #f1f5f9;
    }
    .chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin: 8px 0 14px 0;
    }
    .chip {
        background: #f1f5f9;
        color: #0f172a;
        border: 1px solid #cbd5e1;
        border-radius: 999px;
        padding: 6px 10px;
        font-size: 0.85rem;
        white-space: nowrap;
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
            "PORCENTAJE_HISPANOS": {"label": "Porcentaje hispanos", "weight": 15, "sense": "direct"},
            "EDAD_MEDIANA": {"label": "Edad mediana", "weight": 15, "sense": "direct"},
            "INGRESO_MEDIANO_HOGAR": {"label": "Ingreso mediano del hogar", "weight": 25, "sense": "direct"},
            "TAMANO_HOGAR_PROMEDIO": {"label": "Tamaño hogar promedio", "weight": 15, "sense": "direct"},
        },
    },
    "MOVILIDAD": {
        "label": "Movilidad",
        "variables": {
            "MOVILIDAD_PROMEDIO_DIARIA": {"label": "Movilidad promedio diaria", "weight": 75, "sense": "direct"},
            "MOV_CANTIDAD_ESTACIONES": {"label": "Cantidad de estaciones", "weight": 25, "sense": "direct"},
        },
    },
    "SEGURIDAD": {
        "label": "Seguridad",
        "variables": {
            "DELITO_PROPIEDAD_KM2": {"label": "Delito propiedad por km²", "weight": 45, "sense": "inverse"},
            "DELITO_TRANSPORTE_KM2": {"label": "Delito transporte por km²", "weight": 35, "sense": "inverse"},
            "DELITO_OTROS_KM2": {"label": "Otros delitos por km²", "weight": 20, "sense": "inverse"},
        },
    },
    "PUNTOS_INTERES": {
        "label": "Puntos de interés",
        "variables": {
            "LUGARES_COMERCIO_KM2": {"label": "Lugares comercio por km²", "weight": 35, "sense": "direct"},
            "LUGARES_OFICINAS_KM2": {"label": "Lugares oficinas por km²", "weight": 45, "sense": "direct"},
            "LUGARES_RESIDENCIAL_KM2": {"label": "Lugares residencial por km²", "weight": 20, "sense": "direct"},
        },
    },
    "COMPETENCIA": {
        "label": "Competencia",
        "variables": {
            "COMPETENCIA_DIRECTA_KM2": {"label": "Competencia directa por km²", "weight": 90, "sense": "inverse"},
            "COMPETENCIA_INDIRECTA_KM2": {"label": "Competencia indirecta por km²", "weight": 10, "sense": "direct"},
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
        "description": "Prioriza las dimensiones más vinculadas con la capacidad de atracción comercial de la zona.",
        "weights": {
            "DEMANDA": 35,
            "PUNTOS_INTERES": 25,
            "MOVILIDAD": 15,
            "SEGURIDAD": 10,
            "COSTE": 10,
            "COMPETENCIA": 5,
        },
        "main_dims": ["DEMANDA", "PUNTOS_INTERES"],
        "context_dims": ["MOVILIDAD", "SEGURIDAD", "COSTE", "COMPETENCIA"],
    },
    "Eficiencia y flujo": {
        "description": "Da mayor peso a las condiciones urbanas más relevantes para un modelo fast casual orientado al take-away.",
        "weights": {
            "MOVILIDAD": 35,
            "PUNTOS_INTERES": 25,
            "DEMANDA": 15,
            "SEGURIDAD": 10,
            "COSTE": 10,
            "COMPETENCIA": 5,
        },
        "main_dims": ["MOVILIDAD", "PUNTOS_INTERES"],
        "context_dims": ["DEMANDA", "SEGURIDAD", "COSTE", "COMPETENCIA"],
    },
    "Viabilidad y riesgo": {
        "description": "Enfatiza los factores que inciden con mayor fuerza en la estabilidad operativa y económica de la implantación, así como en la saturación competitiva del entorno.",
        "weights": {
            "SEGURIDAD": 20,
            "COSTE": 25,
            "COMPETENCIA": 15,
            "DEMANDA": 15,
            "MOVILIDAD": 10,
            "PUNTOS_INTERES": 15,
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
    "MOVILIDAD_PROMEDIO_DIARIA": "alta movilidad",
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

CLUSTER_LABELS = {
    1: "Cluster A",
    2: "Cluster B",
    3: "Cluster C",
    4: "Cluster D",
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


def render_chips(items):
    html = '<div class="chip-row">'
    for item in items:
        html += f'<span class="chip">{item}</span>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def clean_zone_id(value):
    if pd.isna(value):
        return None
    return str(value).strip().upper()


def score_0_100_percentile(series: pd.Series, sense: str = "direct") -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    s = s.fillna(s.median())

    p5 = s.quantile(0.05)
    p95 = s.quantile(0.95)

    if pd.isna(p5) or pd.isna(p95) or p95 == p5:
        return pd.Series(50.0, index=s.index)

    s_clipped = s.clip(lower=p5, upper=p95)

    if sense == "direct":
        out = ((s_clipped - p5) / (p95 - p5)) * 100
    else:
        out = ((p95 - s_clipped) / (p95 - p5)) * 100

    return out.clip(0, 100).round(2)


def classify_level(score):
    if score >= 85:
        return "muy alta"
    if score >= 70:
        return "alta"
    if score >= 40:
        return "media"
    if score >= 25:
        return "baja"
    return "muy baja"


def classify_level_plural(score):
    if score >= 85:
        return "muy altas"
    if score >= 70:
        return "altas"
    if score >= 40:
        return "medias"
    if score >= 25:
        return "bajas"
    return "muy bajas"


def score_icon(score):
    if score >= 85:
        return "🟢"
    if score >= 70:
        return "🟩"
    if score >= 40:
        return "🟡"
    if score >= 25:
        return "🟠"
    return "🔴"


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

            out[score_col] = score_0_100_percentile(out[var], var_meta["sense"])
            out[contrib_col] = out[score_col] * (var_meta["weight"] / 100)
            contrib_cols.append(contrib_col)

        out[f"SCORE_DIM_{dim_key}"] = out[contrib_cols].sum(axis=1).round(2)

    return out


def compute_clusters(df, feature_cols, n_clusters=4):
    X = df[feature_cols].copy()
    X = X.apply(pd.to_numeric, errors="coerce").fillna(X.median())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    k = min(n_clusters, len(df))
    if k < 2:
        return pd.Series([0] * len(df), index=df.index)

    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = model.fit_predict(X_scaled)
    return pd.Series(labels, index=df.index)


def build_cluster_names(df):
    cluster_names = {}

    for cluster_id in sorted(df["CLUSTER_K4"].dropna().unique()):
        sub = df[df["CLUSTER_K4"] == cluster_id]
        feature_scores = {}
        for dim_meta in DIMENSIONS.values():
            for var in dim_meta["variables"]:
                feature_scores[var] = sub[f"SCORE_VAR_{var}"].mean()

        top_vars = sorted(feature_scores, key=feature_scores.get, reverse=True)[:2]
        descriptors = [CLUSTER_DESCRIPTORS.get(v, v) for v in top_vars]
        suffix = " + ".join(descriptors) if descriptors else "perfil mixto"
        cluster_names[cluster_id] = f"{CLUSTER_LABELS.get(cluster_id, f'Cluster {cluster_id}')} · {suffix}"

    return cluster_names


def competition_summary_text(row):
    direct = row["SCORE_VAR_COMPETENCIA_DIRECTA_KM2"]
    indirect = row["SCORE_VAR_COMPETENCIA_INDIRECTA_KM2"]

    better = "competencia indirecta" if indirect >= direct else "competencia directa"
    worse = "competencia directa" if indirect >= direct else "competencia indirecta"

    dim_score = row["SCORE_DIM_COMPETENCIA"]

    if dim_score >= 70:
        intro = "Lo que indica una presión competitiva relativamente más moderada dentro del modelo."
    elif dim_score >= 40:
        intro = "Lo que indica una presión competitiva intermedia dentro del modelo."
    else:
        intro = "Lo que indica una presión competitiva relativamente más exigente dentro del modelo."

    return f"{intro} La dimensión muestra mejor desempeño en {better} que en {worse}."


def cost_summary_text(row):
    score = row["SCORE_VAR_ALQ_PRECIO_PIE2_ANUAL"]

    if score >= 70:
        return "Lo que indica un nivel de alquiler relativamente más bajo dentro del conjunto analizado."
    if score >= 40:
        return "Lo que indica un nivel de alquiler intermedio dentro del conjunto analizado."
    return "Lo que indica un nivel de alquiler relativamente más alto dentro del conjunto analizado."


def demand_summary_text(row):
    score = row["SCORE_DIM_DEMANDA"]
    return f"La demanda presenta un nivel {classify_level(score)} dentro del conjunto analizado, con un score de {score:.2f}/100."


def mobility_summary_text(row):
    score = row["SCORE_DIM_MOVILIDAD"]
    return f"La movilidad presenta un nivel {classify_level(score)} dentro del conjunto analizado, con un score de {score:.2f}/100."


def security_summary_text(row):
    score = row["SCORE_DIM_SEGURIDAD"]
    return f"La seguridad presenta un nivel {classify_level(score)} dentro del conjunto analizado, con un score de {score:.2f}/100."


def poi_summary_text(row):
    score = row["SCORE_DIM_PUNTOS_INTERES"]
    return f"Los puntos de interés presentan un nivel {classify_level(score)} dentro del conjunto analizado, con un score de {score:.2f}/100."


def compute_scenario_score(df, scenario_weights):
    out = df.copy()
    contrib_cols = []
    for dim_key, weight in scenario_weights.items():
        col = f"CONTRIB_SCEN_{dim_key}"
        out[col] = out[f"SCORE_DIM_{dim_key}"] * (weight / 100)
        contrib_cols.append(col)

    out["SCORE_ESCENARIO"] = out[contrib_cols].sum(axis=1).round(2)
    out["CATEGORIA_ESCENARIO"] = out["SCORE_ESCENARIO"].apply(classify_level)
    return out


def compute_feasible_bounds(total, dims_count, min_each, max_each):
    lower = max(min_each, total - (dims_count - 1) * max_each)
    upper = min(max_each, total - (dims_count - 1) * min_each)
    return int(lower), int(upper)


def allocate_remaining_weights(dims, target, base_weights, min_each, max_each):
    if not dims:
        return {}

    weights = {d: int(min_each) for d in dims}
    remaining = int(target) - int(min_each) * len(dims)

    if remaining < 0:
        raise ValueError("No existe una asignación factible con esos parámetros.")

    priorities = {}
    for d in dims:
        priorities[d] = max(int(base_weights.get(d, min_each)) - int(min_each), 1)

    while remaining > 0:
        eligible = [d for d in dims if weights[d] < max_each]
        if not eligible:
            break
        chosen = max(
            eligible,
            key=lambda d: priorities[d] / (weights[d] - min_each + 1),
        )
        weights[chosen] += 1
        remaining -= 1

    return weights


def rebalance_weights(weights, selected_dim, new_value, total, min_each, max_each):
    out = {k: int(v) for k, v in weights.items()}
    out[selected_dim] = int(new_value)

    other_dims = [d for d in out if d != selected_dim]
    remaining_total = int(total) - int(new_value)

    allocated = allocate_remaining_weights(
        dims=other_dims,
        target=remaining_total,
        base_weights={d: out[d] for d in other_dims},
        min_each=min_each,
        max_each=max_each,
    )

    for d, v in allocated.items():
        out[d] = v

    return out


def build_filter_defaults(scenario_scored):
    all_clusters = scenario_scored["CLUSTER_FILTER"].dropna().unique().tolist()
    return {
        "filter_score_range": (
            float(scenario_scored["SCORE_ESCENARIO"].min()),
            float(scenario_scored["SCORE_ESCENARIO"].max()),
        ),
        "filter_cost_range": (
            float(scenario_scored["ALQ_PRECIO_PIE2_ANUAL"].min()),
            float(scenario_scored["ALQ_PRECIO_PIE2_ANUAL"].max()),
        ),
        "filter_competition_range": (
            float(scenario_scored["COMPETENCIA_DIRECTA_KM2"].min()),
            float(scenario_scored["COMPETENCIA_DIRECTA_KM2"].max()),
        ),
        "filter_mobility_range": (
            float(scenario_scored["MOVILIDAD_PROMEDIO_DIARIA"].min()),
            float(scenario_scored["MOVILIDAD_PROMEDIO_DIARIA"].max()),
        ),
        "filter_zones": sorted(scenario_scored["NOMBRE_ZONA"].dropna().unique().tolist()),
        "filter_clusters": sorted(all_clusters),
    }


def reset_filters_callback(defaults):
    for k, v in defaults.items():
        st.session_state[k] = v


def sync_filter_state_with_scenario(scenario_name, defaults):
    key = "active_scenario_for_filters"
    if st.session_state.get(key) != scenario_name:
        reset_filters_callback(defaults)
        st.session_state[key] = scenario_name


def initialize_weight_state(state_key, defaults):
    if state_key not in st.session_state:
        st.session_state[state_key] = defaults.copy()


def render_dimension_detail_table(row, dim_key):
    rows = []
    for var, var_meta in DIMENSIONS[dim_key]["variables"].items():
        rows.append(
            {
                "Variable": var_meta["label"],
                "Campo": var,
                "Valor": row[var],
                "Score 0-100": row[f"SCORE_VAR_{var}"],
                "Peso local (%)": var_meta["weight"],
                "Contribución": row[f"CONTRIB_DIMVAR_{var}"],
            }
        )
    return pd.DataFrame(rows)


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

clusters = compute_clusters(df, all_feature_cols)
df["CLUSTER_K4"] = (clusters + 1).astype(int)

cluster_names = build_cluster_names(df)
df["CLUSTER_DESC"] = df["CLUSTER_K4"].map(cluster_names)
df["CLUSTER_FILTER"] = df["CLUSTER_K4"].map(CLUSTER_LABELS)


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

st.sidebar.markdown("### Ajuste de pesos")
st.sidebar.caption(
    "En cada escenario, las dimensiones principales concentran el 60% del peso total y las dimensiones de contexto el 40%. "
    "Puedes modificar varias dimensiones de forma acumulativa dentro de cada bloque, y la aplicación reajusta automáticamente las demás para conservar esa lógica."
)

# PRINCIPALES
st.sidebar.markdown("**Dimensiones principales**")
main_dims = scenario["main_dims"]
main_defaults = {d: scenario["weights"][d] for d in main_dims}
main_weights_key = f"main_weights_{scenario_name}"
initialize_weight_state(main_weights_key, main_defaults)

main_min = 10
main_max = 60

main_selected = st.sidebar.selectbox(
    "Dimensión principal a modificar",
    options=main_dims,
    format_func=lambda x: DIMENSIONS[x]["label"],
    key=f"main_selected_{scenario_name}",
)

main_lower, main_upper = compute_feasible_bounds(
    total=60,
    dims_count=len(main_dims),
    min_each=main_min,
    max_each=main_max,
)

current_main_weights = st.session_state[main_weights_key].copy()
main_current_value = int(current_main_weights[main_selected])
main_current_value = max(main_lower, min(main_current_value, main_upper))

if main_lower == main_upper:
    main_selected_value = main_lower
    st.sidebar.caption(
        f"Peso fijo para {DIMENSIONS[main_selected]['label']}: {main_selected_value}%"
    )
else:
    main_selected_value = st.sidebar.slider(
        f"Peso de {DIMENSIONS[main_selected]['label']} (%)",
        min_value=main_lower,
        max_value=main_upper,
        value=main_current_value,
        step=1,
    )

main_weights = rebalance_weights(
    current_main_weights,
    selected_dim=main_selected,
    new_value=main_selected_value,
    total=60,
    min_each=main_min,
    max_each=main_max,
)
st.session_state[main_weights_key] = main_weights
render_chips([f"{DIMENSIONS[d]['label']}: {w}%" for d, w in main_weights.items()])

# CONTEXTO
st.sidebar.markdown("**Dimensiones de contexto**")
context_dims = scenario["context_dims"]
context_defaults = {d: scenario["weights"][d] for d in context_dims}
context_weights_key = f"context_weights_{scenario_name}"
initialize_weight_state(context_weights_key, context_defaults)

context_min = 5
context_max = 25

context_selected = st.sidebar.selectbox(
    "Dimensión de contexto a modificar",
    options=context_dims,
    format_func=lambda x: DIMENSIONS[x]["label"],
    key=f"context_selected_{scenario_name}",
)

context_lower, context_upper = compute_feasible_bounds(
    total=40,
    dims_count=len(context_dims),
    min_each=context_min,
    max_each=context_max,
)

current_context_weights = st.session_state[context_weights_key].copy()
context_current_value = int(current_context_weights[context_selected])
context_current_value = max(context_lower, min(context_current_value, context_upper))

if context_lower == context_upper:
    context_selected_value = context_lower
    st.sidebar.caption(
        f"Peso fijo para {DIMENSIONS[context_selected]['label']}: {context_selected_value}%"
    )
else:
    context_selected_value = st.sidebar.slider(
        f"Peso de {DIMENSIONS[context_selected]['label']} (%)",
        min_value=context_lower,
        max_value=context_upper,
        value=context_current_value,
        step=1,
    )

context_weights = rebalance_weights(
    current_context_weights,
    selected_dim=context_selected,
    new_value=context_selected_value,
    total=40,
    min_each=context_min,
    max_each=context_max,
)
st.session_state[context_weights_key] = context_weights
render_chips([f"{DIMENSIONS[d]['label']}: {w}%" for d, w in context_weights.items()])

scenario_weights = {}
scenario_weights.update(main_weights)
scenario_weights.update(context_weights)

scenario_scored = compute_scenario_score(df, scenario_weights)

defaults = build_filter_defaults(scenario_scored)
sync_filter_state_with_scenario(scenario_name, defaults)

st.sidebar.markdown("### Filtros")
st.sidebar.button(
    "Restablecer filtros",
    on_click=reset_filters_callback,
    args=(defaults,),
)

score_min, score_max = defaults["filter_score_range"]
cost_min, cost_max = defaults["filter_cost_range"]
comp_min, comp_max = defaults["filter_competition_range"]
mob_min, mob_max = defaults["filter_mobility_range"]

selected_score_range = st.sidebar.slider(
    "Rango score escenario",
    min_value=float(score_min),
    max_value=float(score_max),
    value=st.session_state.get("filter_score_range", defaults["filter_score_range"]),
)

selected_cost_range = st.sidebar.slider(
    "Rango coste (ALQ_PRECIO_PIE2_ANUAL)",
    min_value=float(cost_min),
    max_value=float(cost_max),
    value=st.session_state.get("filter_cost_range", defaults["filter_cost_range"]),
)

selected_comp_range = st.sidebar.slider(
    "Rango competencia directa",
    min_value=float(comp_min),
    max_value=float(comp_max),
    value=st.session_state.get("filter_competition_range", defaults["filter_competition_range"]),
)

selected_mob_range = st.sidebar.slider(
    "Rango movilidad promedio diaria",
    min_value=float(mob_min),
    max_value=float(mob_max),
    value=st.session_state.get("filter_mobility_range", defaults["filter_mobility_range"]),
)

selected_zones = st.sidebar.multiselect(
    "Zonas",
    options=defaults["filter_zones"],
    default=st.session_state.get("filter_zones", defaults["filter_zones"]),
)

selected_clusters = st.sidebar.multiselect(
    "Clusters",
    options=defaults["filter_clusters"],
    default=st.session_state.get("filter_clusters", defaults["filter_clusters"]),
)

filtered = scenario_scored.copy()
filtered = filtered[
    (filtered["SCORE_ESCENARIO"] >= selected_score_range[0])
    & (filtered["SCORE_ESCENARIO"] <= selected_score_range[1])
    & (filtered["ALQ_PRECIO_PIE2_ANUAL"] >= selected_cost_range[0])
    & (filtered["ALQ_PRECIO_PIE2_ANUAL"] <= selected_cost_range[1])
    & (filtered["COMPETENCIA_DIRECTA_KM2"] >= selected_comp_range[0])
    & (filtered["COMPETENCIA_DIRECTA_KM2"] <= selected_comp_range[1])
    & (filtered["MOVILIDAD_PROMEDIO_DIARIA"] >= selected_mob_range[0])
    & (filtered["MOVILIDAD_PROMEDIO_DIARIA"] <= selected_mob_range[1])
    & (filtered["NOMBRE_ZONA"].isin(selected_zones))
    & (filtered["CLUSTER_FILTER"].isin(selected_clusters))
].copy()

filtered = filtered.sort_values("SCORE_ESCENARIO", ascending=False).reset_index(drop=True)
filtered["RANK"] = filtered.index + 1

if filtered.empty:
    st.warning("No hay resultados con la combinación actual de pesos y filtros.")
    st.stop()


# =========================================================
# CABECERA DE RESULTADOS
# =========================================================
best_zone = filtered.iloc[0]
coverage = filtered["ID_ZONA"].isin(geojson_id_set).mean() * 100 if len(filtered) > 0 else 0

col1, col2, col3, col4 = st.columns(4)
with col1:
    metric_card("Zonas visibles", f"{len(filtered)}", "Tras aplicar pesos y filtros")
with col2:
    metric_card("Mejor zona", best_zone["NOMBRE_ZONA"], f"Score: {best_zone['SCORE_ESCENARIO']:.2f}")
with col3:
    metric_card("Promedio escenario", f"{filtered['SCORE_ESCENARIO'].mean():.2f}", f"Categoría media: {classify_level(filtered['SCORE_ESCENARIO'].mean())}")
with col4:
    metric_card("Cobertura GeoJSON", f"{coverage:.1f}%", "IDs de zona con geometría detectada")


# =========================================================
# TABS
# =========================================================
tabs = st.tabs(["Ranking y mapa", "Top 5", "Ficha de zona", "Metodología"])

with tabs[0]:
    st.subheader("Ranking de zonas")

    ranking_cols = [
        "RANK",
        "NOMBRE_ZONA",
        "SCORE_ESCENARIO",
        "CATEGORIA_ESCENARIO",
        "SCORE_DIM_DEMANDA",
        "SCORE_DIM_MOVILIDAD",
        "SCORE_DIM_SEGURIDAD",
        "SCORE_DIM_PUNTOS_INTERES",
        "SCORE_DIM_COMPETENCIA",
        "SCORE_DIM_COSTE",
        "CLUSTER_FILTER",
    ]
    st.dataframe(
        filtered[ranking_cols].rename(
            columns={
                "NOMBRE_ZONA": "Zona",
                "SCORE_ESCENARIO": "Score escenario",
                "CATEGORIA_ESCENARIO": "Categoría",
                "SCORE_DIM_DEMANDA": "Demanda",
                "SCORE_DIM_MOVILIDAD": "Movilidad",
                "SCORE_DIM_SEGURIDAD": "Seguridad",
                "SCORE_DIM_PUNTOS_INTERES": "Puntos de interés",
                "SCORE_DIM_COMPETENCIA": "Competencia",
                "SCORE_DIM_COSTE": "Coste",
                "CLUSTER_FILTER": "Cluster",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Mapa territorial")
    map_df = filtered[filtered["ID_ZONA"].isin(geojson_id_set)].copy()

    if map_df.empty:
        st.info("No hay coincidencias entre las zonas filtradas y los IDs detectados en el GeoJSON.")
    else:
        fig_map = px.choropleth_mapbox(
            map_df,
            geojson=geojson,
            locations="ID_ZONA",
            featureidkey=f"properties.{geo_field}",
            color="SCORE_ESCENARIO",
            hover_name="NOMBRE_ZONA",
            hover_data={
                "ID_ZONA": True,
                "SCORE_ESCENARIO": ':.2f',
                "SCORE_DIM_DEMANDA": ':.2f',
                "SCORE_DIM_MOVILIDAD": ':.2f',
                "SCORE_DIM_SEGURIDAD": ':.2f',
                "SCORE_DIM_PUNTOS_INTERES": ':.2f',
                "SCORE_DIM_COMPETENCIA": ':.2f',
                "SCORE_DIM_COSTE": ':.2f',
            },
            mapbox_style="carto-positron",
            zoom=10.5,
            center={"lat": 40.7831, "lon": -73.9712},
            opacity=0.7,
        )
        fig_map.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=700)
        st.plotly_chart(fig_map, use_container_width=True)

with tabs[1]:
    st.subheader("Comparativa dimensional del Top 5")

    top5 = filtered.head(5).copy()
    dim_cols = [
        "SCORE_DIM_DEMANDA",
        "SCORE_DIM_MOVILIDAD",
        "SCORE_DIM_SEGURIDAD",
        "SCORE_DIM_PUNTOS_INTERES",
        "SCORE_DIM_COMPETENCIA",
        "SCORE_DIM_COSTE",
    ]
    melted = top5.melt(
        id_vars=["NOMBRE_ZONA"],
        value_vars=dim_cols,
        var_name="Dimensión",
        value_name="Score",
    )
    melted["Dimensión"] = (
        melted["Dimensión"]
        .str.replace("SCORE_DIM_", "", regex=False)
        .str.replace("_", " ", regex=False)
        .str.title()
    )

    fig_top = px.bar(
        melted,
        x="NOMBRE_ZONA",
        y="Score",
        color="Dimensión",
        barmode="group",
        title="Top 5 por score escenario",
    )
    fig_top.update_layout(
        xaxis_title="Zona",
        yaxis_title="Score 0-100",
        legend_title="Dimensión",
        height=520,
    )
    st.plotly_chart(fig_top, use_container_width=True)

    st.subheader("Pesos efectivos del escenario")
    weights_df = pd.DataFrame(
        [
            {"Dimensión": DIMENSIONS[d]["label"], "Peso global (%)": w}
            for d, w in scenario_weights.items()
        ]
    ).sort_values("Peso global (%)", ascending=False)
    st.dataframe(weights_df, use_container_width=True, hide_index=True)

with tabs[2]:
    st.subheader("Ficha detallada de zona")

    selected_zone_name = st.selectbox(
        "Selecciona una zona",
        options=filtered["NOMBRE_ZONA"].tolist(),
    )
    row = filtered[filtered["NOMBRE_ZONA"] == selected_zone_name].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Zona", row["NOMBRE_ZONA"], f"Rank: {int(row['RANK'])}")
    with c2:
        metric_card("Score escenario", f"{row['SCORE_ESCENARIO']:.2f}", f"{score_icon(row['SCORE_ESCENARIO'])} {classify_level(row['SCORE_ESCENARIO'])}")
    with c3:
        metric_card("Cluster", row["CLUSTER_FILTER"], row["CLUSTER_DESC"])
    with c4:
        metric_card("ID zona", row["ID_ZONA"], "Identificador geográfico")

    st.markdown(
        f"""
        <div class="summary-box">
            <strong>Lectura general</strong><br>
            La zona <strong>{row['NOMBRE_ZONA']}</strong> obtiene un score de <strong>{row['SCORE_ESCENARIO']:.2f}/100</strong>,
            lo que la sitúa en una categoría <strong>{classify_level(row['SCORE_ESCENARIO'])}</strong> dentro del escenario
            <strong>{scenario_name}</strong>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    summaries = {
        "DEMANDA": demand_summary_text(row),
        "MOVILIDAD": mobility_summary_text(row),
        "SEGURIDAD": security_summary_text(row),
        "PUNTOS_INTERES": poi_summary_text(row),
        "COMPETENCIA": competition_summary_text(row),
        "COSTE": cost_summary_text(row),
    }

    for dim_key in ["DEMANDA", "MOVILIDAD", "SEGURIDAD", "PUNTOS_INTERES", "COMPETENCIA", "COSTE"]:
        st.markdown(f"### {DIMENSIONS[dim_key]['label']}")
        metric_card(
            "Score dimensión",
            f"{row[f'SCORE_DIM_{dim_key}']:.2f}",
            f"{score_icon(row[f'SCORE_DIM_{dim_key}'])} {classify_level(row[f'SCORE_DIM_{dim_key}'])}",
        )
        st.write(summaries[dim_key])
        st.dataframe(render_dimension_detail_table(row, dim_key), use_container_width=True, hide_index=True)

with tabs[3]:
    st.subheader("Metodología resumida")

    st.markdown(
        """
        **Estandarización de variables**
        - Cada variable técnica se transforma a una escala 0–100 mediante percentiles acotados entre p5 y p95.
        - Las variables de beneficio usan sentido directo.
        - Las variables de coste o riesgo usan sentido inverso.

        **Interpretación de categorías**
        - **Muy alta**: 85 a 100
        - **Alta**: 70 a 84.99
        - **Media**: 40 a 69.99
        - **Baja**: 25 a 39.99
        - **Muy baja**: 0 a 24.99

        **Lógica de escenarios**
        - Las dimensiones principales concentran el 60% del peso total.
        - Las dimensiones de contexto concentran el 40% del peso total.
        - El ajuste lateral reequilibra automáticamente el resto de pesos dentro de cada bloque.

        **Pesos locales por dimensión**
        """
    )

    for dim_key, dim_meta in DIMENSIONS.items():
        st.markdown(f"**{dim_meta['label']}**")
        dim_df = pd.DataFrame(
            [
                {
                    "Variable técnica": var,
                    "Etiqueta": var_meta["label"],
                    "Peso local (%)": var_meta["weight"],
                    "Sentido": "Directo" if var_meta["sense"] == "direct" else "Inverso",
                }
                for var, var_meta in dim_meta["variables"].items()
            ]
        )
        st.dataframe(dim_df, use_container_width=True, hide_index=True)

    st.markdown("**Escenarios disponibles**")
    scen_df = pd.DataFrame(
        [
            {
                "Escenario": scen_name,
                "Descripción": scen_meta["description"],
                "Dimensiones principales": ", ".join([DIMENSIONS[d]["label"] for d in scen_meta["main_dims"]]),
                "Dimensiones contexto": ", ".join([DIMENSIONS[d]["label"] for d in scen_meta["context_dims"]]),
            }
            for scen_name, scen_meta in SCENARIOS.items()
        ]
    )
    st.dataframe(scen_df, use_container_width=True, hide_index=True)
