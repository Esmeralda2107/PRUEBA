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
            "PORCENTAJE_HISPANOS": {"label": "Porcentaje hispanos", "weight": 20, "sense": "direct"},
            "EDAD_MEDIANA": {"label": "Edad mediana", "weight": 15, "sense": "direct"},
            "INGRESO_MEDIANO_HOGAR": {"label": "Ingreso mediano del hogar", "weight": 20, "sense": "direct"},
            "TAMANO_HOGAR_PROMEDIO": {"label": "Tamaño hogar promedio", "weight": 15, "sense": "direct"},
        },
    },
    "MOVILIDAD": {
        "label": "Movilidad",
        "variables": {
            "MOVILIDAD_PROMEDIO_DIARIA": {"label": "Movilidad promedio diaria", "weight": 60, "sense": "direct"},
            "MOV_CANTIDAD_ESTACIONES": {"label": "Cantidad de estaciones", "weight": 40, "sense": "direct"},
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
            "COMPETENCIA_DIRECTA_KM2": {"label": "Competencia directa por km²", "weight": 60, "sense": "inverse"},
            "COMPETENCIA_INDIRECTA_KM2": {"label": "Competencia indirecta por km²", "weight": 40, "sense": "direct"},
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
    if score >= 50:
        return "media"
    if score >= 25:
        return "baja"
    return "muy baja"


def classify_level_plural(score):
    if score >= 85:
        return "muy altas"
    if score >= 70:
        return "altas"
    if score >= 50:
        return "medias"
    if score >= 25:
        return "bajas"
    return "muy bajas"


def score_icon(score):
    if score >= 85:
        return "🟢"
    if score >= 70:
        return "🟩"
    if score >= 50:
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

    return clusters


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
            cluster_names[cluster_id] = "sin rasgo dominante claro"
        elif len(top_vars) == 1:
            cluster_names[cluster_id] = top_vars[0]
        else:
            cluster_names[cluster_id] = f"{top_vars[0]} y {top_vars[1]}"

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
                    "var": var,
                }
            )
    items = sorted(items, key=lambda x: x["contrib"], reverse=True)
    return items[:top_n]


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


def render_html_table_multiindex(df):
    html = '<div class="custom-table">' + df.to_html(index=False, escape=False) + "</div>"
    st.markdown(html, unsafe_allow_html=True)


def build_grouped_context(df, scenario_name):
    out = df[[
        "RANK", "ID_ZONA", "NOMBRE_ZONA", "CLUSTER_FILTER", "CLUSTER_DESC", "SCORE_ESCENARIO"
    ]].head(10).copy()
    out["ESCENARIO"] = scenario_name
    out = out.rename(columns={
        "RANK": "Rank",
        "ID_ZONA": "ID zona",
        "NOMBRE_ZONA": "Zona",
        "CLUSTER_FILTER": "Cluster",
        "CLUSTER_DESC": "Nombre del cluster",
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
        out[col] = out[col].apply(
            lambda x: f"{score_icon(float(x))} puntuación {classify_level(float(x))} | {fmt_num(float(x), 1)}"
        )

    return out


def build_grouped_subdimensions(df):
    data_cols = []
    col_tuples = [("", "Rank"), ("", "Zona")]

    for dim_key, dim_meta in DIMENSIONS.items():
        for var, var_meta in dim_meta["variables"].items():
            data_cols.append(var)
            col_tuples.append((dim_meta["label"], var_meta["label"]))

    base = df[["RANK", "NOMBRE_ZONA"] + data_cols].head(10).copy()
    base["RANK"] = base["RANK"].apply(fmt_int)

    for col in data_cols:
        base[col] = base[col].apply(lambda x: fmt_num(x, 2))

    base.columns = pd.MultiIndex.from_tuples(col_tuples)
    return base


def compute_feasible_bounds(total, dims_count, min_each, max_each):
    lower = max(min_each, total - (dims_count - 1) * max_each)
    upper = min(max_each, total - (dims_count - 1) * min_each)
    return lower, upper


def allocate_remaining(selected_dim, selected_value, dims, total, min_each, max_each, base_weights):
    remaining_dims = [d for d in dims if d != selected_dim]
    remaining_total = total - selected_value

    if not remaining_dims:
        return {selected_dim: selected_value}

    weights = {d: min_each for d in remaining_dims}
    extra_capacity = {d: max_each - min_each for d in remaining_dims}
    base_pref = {d: max(base_weights[d] - min_each, 1) for d in remaining_dims}

    remaining_points = remaining_total - len(remaining_dims) * min_each
    assigned_extra = {d: 0 for d in remaining_dims}

    while remaining_points > 0:
        eligible = [d for d in remaining_dims if assigned_extra[d] < extra_capacity[d]]
        if not eligible:
            break
        chosen = max(eligible, key=lambda d: base_pref[d] / (assigned_extra[d] + 1))
        assigned_extra[chosen] += 1
        remaining_points -= 1

    final_weights = {selected_dim: selected_value}
    for d in remaining_dims:
        final_weights[d] = min_each + assigned_extra[d]

    diff = total - sum(final_weights.values())
    if diff != 0:
        adjustable = [d for d in remaining_dims if min_each <= final_weights[d] + diff <= max_each]
        if adjustable:
            final_weights[adjustable[-1]] += diff

    return final_weights


def top_vars_for_dimension(row, dim_key, n=2):
    items = []
    for var, var_meta in DIMENSIONS[dim_key]["variables"].items():
        items.append(
            {
                "label": var_meta["label"],
                "score": row[f"SCORE_VAR_{var}"],
            }
        )
    items = sorted(items, key=lambda x: x["score"], reverse=True)
    return items[:n]


def demand_summary_text(row):
    top_items = top_vars_for_dimension(row, "DEMANDA", n=2)
    score1 = top_items[0]["score"]
    score2 = top_items[1]["score"]
    level1 = classify_level(score1)
    level2 = classify_level(score2)

    if level1 == level2:
        return (
            f"Destacan {top_items[0]['label'].lower()} y {top_items[1]['label'].lower()}, "
            f"ambos con puntuación {level1}."
        )

    return (
        f"Destacan {top_items[0]['label'].lower()} y {top_items[1]['label'].lower()}, "
        f"con puntuaciones {level1} y {level2}, respectivamente."
    )


def mobility_summary_text(row):
    mov = row["SCORE_VAR_MOVILIDAD_PROMEDIO_DIARIA"]
    est = row["SCORE_VAR_MOV_CANTIDAD_ESTACIONES"]
    level_mov = classify_level(mov)
    level_est = classify_level(est)

    if level_mov == level_est:
        return (
            f"La movilidad promedio diaria y la cantidad de estaciones presentan "
            f"puntuaciones {classify_level_plural((mov + est) / 2)} dentro de esta dimensión."
        )

    return (
        f"La movilidad promedio diaria presenta una puntuación {level_mov}, "
        f"mientras que la cantidad de estaciones muestra una puntuación {level_est} dentro de esta dimensión."
    )


def security_summary_text(row):
    top_item = top_vars_for_dimension(row, "SEGURIDAD", n=1)[0]
    return (
        f"Lo que indica una menor exposición relativa al riesgo dentro del conjunto analizado. "
        f"El mejor desempeño relativo se observa en {top_item['label'].lower()}."
    )


def poi_summary_text(row):
    top_item = top_vars_for_dimension(row, "PUNTOS_INTERES", n=1)[0]
    return (
        f"El principal aporte dentro de la dimensión proviene de {top_item['label'].lower()}, "
        f"con puntuación {classify_level(top_item['score'])}."
    )


def competition_summary_text(row):
    direct = row["SCORE_VAR_COMPETENCIA_DIRECTA_KM2"]
    indirect = row["SCORE_VAR_COMPETENCIA_INDIRECTA_KM2"]

    better = "competencia indirecta" if indirect >= direct else "competencia directa"
    worse = "competencia directa" if indirect >= direct else "competencia indirecta"

    dim_score = row["SCORE_DIM_COMPETENCIA"]

    if dim_score >= 70:
        intro = "Lo que indica una presión competitiva relativamente más moderada dentro del modelo."
    elif dim_score >= 50:
        intro = "Lo que indica una presión competitiva intermedia dentro del modelo."
    else:
        intro = "Lo que indica una presión competitiva relativamente más exigente dentro del modelo."

    return f"{intro} La dimensión muestra mejor desempeño en {better} que en {worse}."


def cost_summary_text(row):
    score = row["SCORE_VAR_ALQ_PRECIO_PIE2_ANUAL"]

    if score >= 70:
        return "Lo que indica un nivel de alquiler relativamente más bajo dentro del conjunto analizado."
    if score >= 50:
        return "Lo que indica un nivel de alquiler intermedio dentro del conjunto analizado."
    return "Lo que indica un nivel de alquiler relativamente más alto dentro del conjunto analizado."


def dimension_summary_line(row, dim_key):
    dim_label = DIMENSIONS[dim_key]["label"]
    dim_score = row[f"SCORE_DIM_{dim_key}"]
    level = classify_level(dim_score)

    if dim_key == "DEMANDA":
        detail = demand_summary_text(row)
    elif dim_key == "MOVILIDAD":
        detail = mobility_summary_text(row)
    elif dim_key == "SEGURIDAD":
        detail = security_summary_text(row)
    elif dim_key == "PUNTOS_INTERES":
        detail = poi_summary_text(row)
    elif dim_key == "COMPETENCIA":
        detail = competition_summary_text(row)
    elif dim_key == "COSTE":
        detail = cost_summary_text(row)
    else:
        detail = ""

    return f"- **{dim_label}**: puntuación {level} ({dim_score:.1f}/100). {detail}"


def get_filter_defaults(scenario_scored, all_clusters):
    return {
        "filter_score_range": (
            float(scenario_scored["SCORE_ESCENARIO"].min()),
            float(scenario_scored["SCORE_ESCENARIO"].max()),
        ),
        "filter_rent_range": (
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


def update_weight_group(weights_key, slider_key, selected_dim, dims, total, min_each, max_each):
    current_weights = st.session_state[weights_key].copy()
    selected_value = int(st.session_state[slider_key])

    new_weights = allocate_remaining(
        selected_dim=selected_dim,
        selected_value=selected_value,
        dims=dims,
        total=total,
        min_each=min_each,
        max_each=max_each,
        base_weights=current_weights,
    )
    st.session_state[weights_key] = new_weights


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

main_min = 20
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

current_main_weights = st.session_state[main_weights_key]
main_current_value = int(current_main_weights[main_selected])
main_current_value = max(main_lower, min(main_current_value, main_upper))

main_slider_key = f"main_slider_{scenario_name}_{main_selected}"
st.session_state[main_slider_key] = main_current_value

st.sidebar.slider(
    f"Peso de {DIMENSIONS[main_selected]['label']} (%)",
    min_value=main_lower,
    max_value=main_upper,
    step=1,
    key=main_slider_key,
    on_change=update_weight_group,
    args=(main_weights_key, main_slider_key, main_selected, main_dims, 60, main_min, main_max),
)

current_main_weights = st.session_state[main_weights_key]
for d in main_dims:
    if d != main_selected:
        st.sidebar.info(f"{DIMENSIONS[d]['label']}: {current_main_weights[d]}% (ajuste automático)")

# CONTEXTO
st.sidebar.markdown("**Dimensiones de contexto**")
context_dims = scenario["context_dims"]
context_defaults = {d: scenario["weights"][d] for d in context_dims}
context_weights_key = f"context_weights_{scenario_name}"
initialize_weight_state(context_weights_key, context_defaults)

context_min = 5
context_max = 15

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

current_context_weights = st.session_state[context_weights_key]
context_current_value = int(current_context_weights[context_selected])
context_current_value = max(context_lower, min(context_current_value, context_upper))

context_slider_key = f"context_slider_{scenario_name}_{context_selected}"
st.session_state[context_slider_key] = context_current_value

st.sidebar.slider(
    f"Peso de {DIMENSIONS[context_selected]['label']} (%)",
    min_value=context_lower,
    max_value=context_upper,
    step=1,
    key=context_slider_key,
    on_change=update_weight_group,
    args=(context_weights_key, context_slider_key, context_selected, context_dims, 40, context_min, context_max),
)

current_context_weights = st.session_state[context_weights_key]
for d in context_dims:
    if d != context_selected:
        st.sidebar.info(f"{DIMENSIONS[d]['label']}: {current_context_weights[d]}% (ajuste automático)")

effective_weights = {**st.session_state[main_weights_key], **st.session_state[context_weights_key]}
scenario_scored = compute_scenario_scores(df, effective_weights)

# FILTROS
st.sidebar.markdown("### Filtros")

all_cluster_options = sorted(df["CLUSTER_FILTER"].dropna().unique().tolist())
filter_defaults = get_filter_defaults(scenario_scored, all_cluster_options)
sync_filter_state_with_scenario(scenario_name, filter_defaults)

st.sidebar.button(
    "Restablecer filtros",
    use_container_width=True,
    on_click=reset_filters_callback,
    args=(filter_defaults,),
)

score_range = st.sidebar.slider(
    "Score del escenario (0–100)",
    min_value=0.0,
    max_value=100.0,
    key="filter_score_range",
)

rent_range = st.sidebar.slider(
    "Alquiler (USD/pie²/año)",
    min_value=float(scenario_scored["ALQ_PRECIO_PIE2_ANUAL"].min()),
    max_value=float(scenario_scored["ALQ_PRECIO_PIE2_ANUAL"].max()),
    key="filter_rent_range",
)

competition_range = st.sidebar.slider(
    "Competencia directa (competidores/km²)",
    min_value=float(scenario_scored["COMPETENCIA_DIRECTA_KM2"].min()),
    max_value=float(scenario_scored["COMPETENCIA_DIRECTA_KM2"].max()),
    key="filter_competition_range",
)

mobility_range = st.sidebar.slider(
    "Movilidad (promedio diario de personas)",
    min_value=float(scenario_scored["MOVILIDAD_PROMEDIO_DIARIA"].min()),
    max_value=float(scenario_scored["MOVILIDAD_PROMEDIO_DIARIA"].max()),
    key="filter_mobility_range",
)

selected_zones = st.sidebar.multiselect(
    "Zonas",
    options=sorted(scenario_scored["NOMBRE_ZONA"].dropna().unique().tolist()),
    key="filter_zones",
)

selected_clusters = st.sidebar.multiselect(
    "Clusters",
    options=all_cluster_options,
    key="filter_clusters",
)

filtered = scenario_scored[
    scenario_scored["SCORE_ESCENARIO"].between(score_range[0], score_range[1])
    & scenario_scored["ALQ_PRECIO_PIE2_ANUAL"].between(rent_range[0], rent_range[1])
    & scenario_scored["COMPETENCIA_DIRECTA_KM2"].between(competition_range[0], competition_range[1])
    & scenario_scored["MOVILIDAD_PROMEDIO_DIARIA"].between(mobility_range[0], mobility_range[1])
    & scenario_scored["NOMBRE_ZONA"].isin(selected_zones)
    & scenario_scored["CLUSTER_FILTER"].isin(selected_clusters)
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
# RESUMEN DE FILTROS ACTIVOS
# =========================================================
st.markdown("### Filtros activos")
st.caption("Resumen del escenario y de las restricciones actualmente aplicadas al análisis.")

excluded_clusters = [c for c in all_cluster_options if c not in selected_clusters]
excluded_clusters_text = ", ".join(excluded_clusters) if excluded_clusters else "ninguno"

chips = [
    f"Escenario: {scenario_name}",
    f"Score mínimo: {fmt_num(score_range[0], 0)}",
    f"Renta máxima: {fmt_num(rent_range[1], 2)}",
    f"Competencia directa máxima: {fmt_num(competition_range[1], 2)}",
    f"Clusters excluidos: {excluded_clusters_text}",
    f"{len(filtered)} zonas visibles",
]
render_chips(chips)


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
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["🗺️ Mapa", "🏆 Ranking", "📊 Gráficos", "📘 Metodología", "⚠️ Limitaciones del modelo"]
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
            "CLUSTER_FILTER": True,
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

    summary_lines = [
        dimension_summary_line(best_zone, "DEMANDA"),
        dimension_summary_line(best_zone, "MOVILIDAD"),
        dimension_summary_line(best_zone, "SEGURIDAD"),
        dimension_summary_line(best_zone, "PUNTOS_INTERES"),
        dimension_summary_line(best_zone, "COMPETENCIA"),
        dimension_summary_line(best_zone, "COSTE"),
    ]

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
    render_html_table_multiindex(build_grouped_subdimensions(filtered))

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
        cluster_dim = (
            filtered.groupby(["CLUSTER_FILTER", "CLUSTER_DESC"], as_index=False)[[
                "SCORE_DIM_DEMANDA",
                "SCORE_DIM_MOVILIDAD",
                "SCORE_DIM_SEGURIDAD",
                "SCORE_DIM_PUNTOS_INTERES",
                "SCORE_DIM_COMPETENCIA",
                "SCORE_DIM_COSTE",
            ]]
            .mean()
        )

        heat = cluster_dim.melt(
            id_vars=["CLUSTER_FILTER", "CLUSTER_DESC"],
            var_name="Dimensión",
            value_name="Puntuación promedio",
        )

        heat["Dimensión"] = heat["Dimensión"].replace({
            "SCORE_DIM_DEMANDA": "Demanda",
            "SCORE_DIM_MOVILIDAD": "Movilidad",
            "SCORE_DIM_SEGURIDAD": "Seguridad",
            "SCORE_DIM_PUNTOS_INTERES": "Puntos de interés",
            "SCORE_DIM_COMPETENCIA": "Competencia",
            "SCORE_DIM_COSTE": "Coste",
        })

        heat["Cluster"] = heat["CLUSTER_FILTER"] + " — " + heat["CLUSTER_DESC"]

        fig_cluster = px.density_heatmap(
            heat,
            x="Dimensión",
            y="Cluster",
            z="Puntuación promedio",
            text_auto=".1f",
            color_continuous_scale="Tealgrn",
            title="Perfil promedio por dimensiones de cada cluster",
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
            title="Alquiler vs movilidad",
            labels={
                "ALQ_PRECIO_PIE2_ANUAL": "Alquiler (USD/pie²/año)",
                "MOVILIDAD_PROMEDIO_DIARIA": "Movilidad (promedio diario de personas)",
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

**Metodología de normalización**
- Para la construcción del scoring, las variables se reexpresan en una escala común de 0 a 100 puntos.
- Con el fin de reducir la influencia de valores extremos, se utilizan como límites de referencia los **percentiles 5 y 95** de la distribución de cada variable.
- Los valores inferiores al percentil 5 se asignan al límite inferior de la escala y los valores superiores al percentil 95 se asignan al límite superior.
- **Sentido directo**: valores más altos reciben puntuaciones más cercanas a 100.
- **Sentido inverso**: valores más altos reciben puntuaciones más cercanas a 0.

**Regla macro**
- Dimensiones principales: **60 %**
- Dimensiones de contexto: **40 %**
- La app permite modificar varias dimensiones de forma acumulativa dentro de cada bloque y reajusta automáticamente las demás para conservar la estructura.

**Interpretación de categorías**
- **Muy alta**: 85 a 100
- **Alta**: 70 a 84.99
- **Media**: 50 a 69.99
- **Baja**: 25 a 49.99
- **Muy baja**: 0 a 24.99

Estas categorías se aplican sobre la **puntuación transformada (0–100)** y no sobre el valor bruto de la variable.  
Por eso, una variable o dimensión con puntuación alta significa **mejor desempeño relativo dentro del modelo de scoring**, no necesariamente un valor bruto alto en el dato original.  
En dimensiones de sentido inverso, una puntuación alta indica una condición relativamente más favorable dentro del conjunto analizado:
- **Seguridad**: menor exposición relativa al riesgo.
- **Coste**: menor nivel relativo de alquiler.
- **Competencia**: menor presión competitiva relativa.

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
        .groupby(["CLUSTER_K4", "CLUSTER_FILTER", "CLUSTER_DESC"])["NOMBRE_ZONA"]
        .apply(list)
        .reset_index()
    )

    for _, row in cluster_list_df.iterrows():
        with st.expander(f"{row['CLUSTER_FILTER']} — {row['CLUSTER_DESC']} | {len(row['NOMBRE_ZONA'])} zonas"):
            st.write(", ".join(row["NOMBRE_ZONA"]))

    st.markdown("### Lectura del gráfico de clusters")
    st.markdown(
        """
- El gráfico muestra el **perfil promedio por dimensiones de cada cluster**.
- Cada fila representa un cluster y cada columna una dimensión.
- El valor y el color indican la puntuación media de ese cluster en esa dimensión.
- Su objetivo es facilitar una lectura más interpretable del agrupamiento territorial.
"""
    )

    st.markdown("### Fórmulas")
    st.latex(r"ScoreVar_{i,j} = \left(\frac{x^{clip}_{i,j}-P5_j}{P95_j-P5_j}\right)\cdot 100")
    st.markdown("Para variables de sentido directo.")

    st.latex(r"ScoreVar_{i,j} = \left(\frac{P95_j-x^{clip}_{i,j}}{P95_j-P5_j}\right)\cdot 100")
    st.markdown("Para variables de sentido inverso.")

    st.latex(r"ScoreDim_{i,d} = \sum_j w_{j|d}\cdot ScoreVar_{i,j}")

    st.latex(r"ScoreEscenario_{i,s} = \sum_d w_{d|s}\cdot ScoreDim_{i,d}")
    
    st.markdown("### Leyenda de fórmulas")
    st.markdown(
            """
- **i**: zona analizada o NTA.
 - **j**: variable o subdimensión.
- **d**: dimensión.
 - **s**: escenario.
- **xᵢⱼ**: valor bruto observado de la variable *j* en la zona *i*.
- **xᶜˡⁱᵖᵢⱼ**: valor de la variable *j* en la zona *i*, acotado entre los percentiles 5 y 95.
- **P5ⱼ**: percentil 5 de la distribución de la variable *j*.
- **P95ⱼ**: percentil 95 de la distribución de la variable *j*.
- **wⱼ|d**: peso local de la variable *j* dentro de la dimensión *d*.
- **w_d|s**: peso macro de la dimensión *d* dentro del escenario *s*.
    """
     )


# =========================================================
# TAB LIMITACIONES
# =========================================================
with tab5:
    st.subheader("Limitaciones del modelo")

    st.markdown(
        """
- El análisis se limita a Manhattan y a las NTA como unidad territorial, por lo que no evalúa ubicaciones puntuales.
- El modelo simplifica la realidad y se basa en variables proxy según la disponibilidad de datos.
- Los resultados son relativos al conjunto de zonas analizadas, por lo que un score alto no implica una condición óptima absoluta.
- La recomendación depende de los pesos asignados en cada escenario, por lo que el ranking puede variar.
- El modelo depende de la calidad, actualización y homogeneidad de las fuentes utilizadas.
- No incorpora factores cualitativos y operativos clave, por lo que no sustituye la validación en campo.
- La recomendación obtenida no garantiza el éxito comercial del negocio.
"""
    )
