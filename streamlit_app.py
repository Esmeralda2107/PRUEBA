from pathlib import Path
import json

import pandas as pd
import plotly.express as px
import streamlit as st


# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================
st.set_page_config(
    page_title="TFM Site Selection Manhattan",
    page_icon="📍",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "datos" / "maestro" / "MASTER_DATASET_MANHATTAN_ML.csv"
GEOJSON_DIR = BASE_DIR / "datos" / "crudos" / "zonas"


# =========================================================
# FUNCIONES AUXILIARES
# =========================================================
def clean_zone_id(value):
    """Normaliza IDs territoriales para que el join no falle por formato."""
    if pd.isna(value):
        return None
    return str(value).strip().upper()


def normalize(series: pd.Series, inverse: bool = False) -> pd.Series:
    """Min-max normalization; invierte el criterio si es de coste."""
    s = pd.to_numeric(series, errors="coerce")
    s = s.fillna(s.median())

    min_v = s.min()
    max_v = s.max()

    if max_v == min_v:
        norm = pd.Series(0.5, index=s.index)
    else:
        norm = (s - min_v) / (max_v - min_v)

    return 1 - norm if inverse else norm


@st.cache_data
def load_data():
    """Carga CSV maestro y el primer GeoJSON encontrado en datos/crudos/zonas."""
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


def detect_geojson_id_field(geojson_dict):
    """
    Intenta detectar automáticamente qué campo dentro de properties
    funciona como ID para enlazar con ID_ZONA.
    """
    features = geojson_dict.get("features", [])
    if not features:
        return None

    props = features[0].get("properties", {})
    candidates = [
        "NTA2020", "nta2020",
        "NTACode", "nta_code", "NTA_CODE",
        "NTA", "nta",
        "id", "ID"
    ]

    for cand in candidates:
        if cand in props:
            return cand

    # Si no encuentra ninguno, devuelve el primer campo disponible
    if props:
        return list(props.keys())[0]

    return None


def extract_geojson_ids(geojson_dict, geo_field):
    """Extrae y limpia todos los IDs del GeoJSON para validar el join."""
    ids = []
    for feature in geojson_dict.get("features", []):
        props = feature.get("properties", {})
        ids.append(clean_zone_id(props.get(geo_field)))
    return ids


def compute_score(df: pd.DataFrame, criteria_meta: dict, weights: dict) -> pd.DataFrame:
    """Calcula score multicriterio ponderado y ranking."""
    out = df.copy()

    for col in weights.keys():
        out[col] = pd.to_numeric(out[col], errors="coerce")
        out[col] = out[col].fillna(out[col].median())

    total_weight = sum(weights.values())
    if total_weight == 0:
        total_weight = 1

    contrib_cols = []

    for col, weight in weights.items():
        inverse = criteria_meta[col]["type"] == "cost"
        out[f"{col}_NORM"] = normalize(out[col], inverse=inverse)
        out[f"CONTRIB_{col}"] = out[f"{col}_NORM"] * (weight / total_weight)
        contrib_cols.append(f"CONTRIB_{col}")

    out["SCORE"] = out[contrib_cols].sum(axis=1)
    out["RANK"] = out["SCORE"].rank(ascending=False, method="dense").astype(int)

    return out.sort_values("SCORE", ascending=False)


# =========================================================
# METADATOS DEL MODELO
# =========================================================
CRITERIA = {
    "POBLACION_KM2": {"label": "Densidad de población", "type": "benefit"},
    "INGRESO_MEDIANO_HOGAR": {"label": "Ingreso mediano del hogar", "type": "benefit"},
    "MOV_CANTIDAD_ESTACIONES": {"label": "Cantidad de estaciones", "type": "benefit"},
    "MOVILIDAD_PROMEDIO_DIARIA": {"label": "Movilidad promedio diaria", "type": "benefit"},
    "ALQ_PRECIO_PIE2_ANUAL": {"label": "Precio alquiler pie² anual", "type": "cost"},
    "DELITO_PROPIEDAD_KM2": {"label": "Delito propiedad por km²", "type": "cost"},
    "COMPETENCIA_DIRECTA_KM2": {"label": "Competencia directa por km²", "type": "cost"},
}

PRESETS = {
    "Balanceado": {
        "POBLACION_KM2": 20,
        "INGRESO_MEDIANO_HOGAR": 20,
        "MOV_CANTIDAD_ESTACIONES": 15,
        "MOVILIDAD_PROMEDIO_DIARIA": 15,
        "ALQ_PRECIO_PIE2_ANUAL": 10,
        "DELITO_PROPIEDAD_KM2": 10,
        "COMPETENCIA_DIRECTA_KM2": 10,
    },
    "Máxima demanda": {
        "POBLACION_KM2": 30,
        "INGRESO_MEDIANO_HOGAR": 25,
        "MOV_CANTIDAD_ESTACIONES": 15,
        "MOVILIDAD_PROMEDIO_DIARIA": 20,
        "ALQ_PRECIO_PIE2_ANUAL": 5,
        "DELITO_PROPIEDAD_KM2": 3,
        "COMPETENCIA_DIRECTA_KM2": 2,
    },
    "Control de coste": {
        "POBLACION_KM2": 15,
        "INGRESO_MEDIANO_HOGAR": 15,
        "MOV_CANTIDAD_ESTACIONES": 10,
        "MOVILIDAD_PROMEDIO_DIARIA": 10,
        "ALQ_PRECIO_PIE2_ANUAL": 30,
        "DELITO_PROPIEDAD_KM2": 10,
        "COMPETENCIA_DIRECTA_KM2": 10,
    },
    "Seguridad primero": {
        "POBLACION_KM2": 15,
        "INGRESO_MEDIANO_HOGAR": 15,
        "MOV_CANTIDAD_ESTACIONES": 10,
        "MOVILIDAD_PROMEDIO_DIARIA": 10,
        "ALQ_PRECIO_PIE2_ANUAL": 10,
        "DELITO_PROPIEDAD_KM2": 30,
        "COMPETENCIA_DIRECTA_KM2": 10,
    },
}


# =========================================================
# CARGA DE DATOS
# =========================================================
st.title("📍 Site Selection Manhattan")
st.caption("Aplicación interactiva para el TFM: mapa, filtros, scoring multicriterio, ranking y gráficos.")

try:
    df, geojson, geojson_filename = load_data()
except Exception as e:
    st.error(str(e))
    st.stop()

required_cols = ["ID_ZONA", "NOMBRE_ZONA"] + list(CRITERIA.keys())
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Faltan columnas en el CSV maestro: {missing}")
    st.stop()

# Limpieza de IDs y nombres
df["ID_ZONA"] = df["ID_ZONA"].apply(clean_zone_id)
df["NOMBRE_ZONA"] = df["NOMBRE_ZONA"].astype(str).str.strip()

# Detección del campo territorial en el GeoJSON
geo_field = detect_geojson_id_field(geojson)
if geo_field is None:
    st.error("No se pudo detectar el campo ID dentro del GeoJSON.")
    st.stop()

geojson_ids = extract_geojson_ids(geojson, geo_field)
geojson_id_set = set([x for x in geojson_ids if x is not None])
csv_id_set = set(df["ID_ZONA"].dropna().unique())

missing_in_geojson = sorted(csv_id_set - geojson_id_set)
missing_in_csv = sorted(geojson_id_set - csv_id_set)


# =========================================================
# BARRA LATERAL
# =========================================================
st.sidebar.header("⚙️ Configuración del modelo")

scenario = st.sidebar.selectbox("Escenario base", list(PRESETS.keys()))
default_weights = PRESETS[scenario]

selected_criteria = st.sidebar.multiselect(
    "Criterios activos",
    options=list(CRITERIA.keys()),
    default=list(default_weights.keys()),
    format_func=lambda x: CRITERIA[x]["label"],
)

weights = {}
st.sidebar.markdown("### Pesos")
for col in selected_criteria:
    weights[col] = st.sidebar.slider(
        CRITERIA[col]["label"],
        min_value=0,
        max_value=100,
        value=int(default_weights.get(col, 10)),
        step=5,
    )

if sum(weights.values()) == 0:
    st.sidebar.warning("Asigna al menos un peso mayor que 0.")
    st.stop()

scored = compute_score(df, CRITERIA, weights)

st.sidebar.markdown("### Filtros")
score_min = st.sidebar.slider("Score mínimo", 0.0, 1.0, 0.0, 0.01)

rent_min = float(scored["ALQ_PRECIO_PIE2_ANUAL"].min())
rent_max = float(scored["ALQ_PRECIO_PIE2_ANUAL"].max())
rent_range = st.sidebar.slider(
    "Rango alquiler",
    min_value=rent_min,
    max_value=rent_max,
    value=(rent_min, rent_max),
)

income_min = float(scored["INGRESO_MEDIANO_HOGAR"].min())
income_max = float(scored["INGRESO_MEDIANO_HOGAR"].max())
income_range = st.sidebar.slider(
    "Rango ingreso mediano",
    min_value=income_min,
    max_value=income_max,
    value=(income_min, income_max),
)

crime_max = float(scored["DELITO_PROPIEDAD_KM2"].max())
crime_limit = st.sidebar.slider(
    "Máximo delito propiedad por km²",
    min_value=0.0,
    max_value=crime_max,
    value=crime_max,
)

zone_options = sorted(scored["NOMBRE_ZONA"].dropna().unique().tolist())
selected_zones = st.sidebar.multiselect(
    "Filtrar zonas concretas",
    options=zone_options,
    default=zone_options,
)

filtered = scored[
    (scored["SCORE"] >= score_min)
    & (scored["ALQ_PRECIO_PIE2_ANUAL"].between(rent_range[0], rent_range[1]))
    & (scored["INGRESO_MEDIANO_HOGAR"].between(income_range[0], income_range[1]))
    & (scored["DELITO_PROPIEDAD_KM2"] <= crime_limit)
    & (scored["NOMBRE_ZONA"].isin(selected_zones))
].copy()

if filtered.empty:
    st.warning("No hay zonas que cumplan los filtros actuales.")
    st.stop()

filtered = filtered.sort_values("SCORE", ascending=False).reset_index(drop=True)
filtered["RANK"] = filtered["SCORE"].rank(ascending=False, method="dense").astype(int)

best_zone = filtered.iloc[0]
criterion_star = max(
    selected_criteria,
    key=lambda c: best_zone.get(f"CONTRIB_{c}", 0)
)


# =========================================================
# KPIS
# =========================================================
c1, c2, c3, c4 = st.columns(4)
c1.metric("Mejor zona", best_zone["NOMBRE_ZONA"])
c2.metric("Score máximo", f"{best_zone['SCORE']:.3f}")
c3.metric("Zonas visibles", f"{len(filtered)}")
c4.metric("Factor dominante", CRITERIA[criterion_star]["label"])


# =========================================================
# DIAGNÓSTICO DEL JOIN
# =========================================================
with st.expander("Diagnóstico mapa / join CSV ↔ GeoJSON"):
    st.write("Archivo GeoJSON cargado:", geojson_filename)
    st.write("Campo detectado en GeoJSON para el join:", geo_field)
    st.write("Ejemplo IDs CSV:", sorted(list(csv_id_set))[:10])
    st.write("Ejemplo IDs GeoJSON:", sorted(list(geojson_id_set))[:10])
    st.write("Total IDs CSV:", len(csv_id_set))
    st.write("Total IDs GeoJSON:", len(geojson_id_set))
    st.write("IDs del CSV que NO están en el GeoJSON:", missing_in_geojson[:20])
    st.write("IDs del GeoJSON que NO están en el CSV:", missing_in_csv[:20])

    features = geojson.get("features", [])
    if features:
        st.write("Propiedades del primer feature del GeoJSON:")
        st.json(features[0].get("properties", {}))


# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Mapa", "🏆 Ranking", "📊 Gráficos", "📘 Metodología"])


with tab1:
    st.subheader("Mapa interactivo de site selection")

    map_metric = st.selectbox(
        "Variable a pintar en el mapa",
        options=["SCORE"] + selected_criteria,
        format_func=lambda x: "Score final" if x == "SCORE" else CRITERIA[x]["label"],
    )

    map_df = filtered.copy()
    map_df["ID_ZONA"] = map_df["ID_ZONA"].apply(clean_zone_id)

    matched = map_df["ID_ZONA"].isin(geojson_id_set)
    unmatched_rows = map_df.loc[~matched, ["ID_ZONA", "NOMBRE_ZONA"]]

    if unmatched_rows.shape[0] > 0:
        st.warning("Estas zonas no encuentran polígono en el GeoJSON:")
        st.dataframe(unmatched_rows, use_container_width=True, hide_index=True)

    map_df = map_df.loc[matched].copy()

    if map_df.empty:
        st.error(
            "Después de validar IDs, no queda ninguna zona con geometría asociada. "
            "Revisa el campo territorial del GeoJSON y el formato de ID_ZONA."
        )
        st.stop()

    # Plotly documenta que el GeoJSON debe enlazarse con locations + featureidkey;
    # aquí lo hacemos usando el campo detectado automáticamente dentro de properties.
    fig_map = px.choropleth_mapbox(
        map_df,
        geojson=geojson,
        locations="ID_ZONA",
        featureidkey=f"properties.{geo_field}",
        color=map_metric,
        hover_name="NOMBRE_ZONA",
        hover_data={
            "ID_ZONA": True,
            "SCORE": ":.3f",
            "RANK": True,
            "POBLACION_KM2": ":.2f",
            "INGRESO_MEDIANO_HOGAR": ":,.0f",
            "MOV_CANTIDAD_ESTACIONES": ":.0f",
            "MOVILIDAD_PROMEDIO_DIARIA": ":,.0f",
            "ALQ_PRECIO_PIE2_ANUAL": ":.2f",
            "DELITO_PROPIEDAD_KM2": ":.2f",
            "COMPETENCIA_DIRECTA_KM2": ":.2f",
        },
        mapbox_style="carto-positron",
        center={"lat": 40.7831, "lon": -73.9712},
        zoom=10.4,
        opacity=0.72,
    )

    fig_map.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=700,
        coloraxis_colorbar_title="Valor",
    )

    st.plotly_chart(fig_map, use_container_width=True)
    st.caption(f"Join usado en el mapa: ID_ZONA ↔ properties.{geo_field}")


with tab2:
    st.subheader("Ranking de ubicaciones")

    show_cols = ["RANK", "ID_ZONA", "NOMBRE_ZONA", "SCORE"] + selected_criteria
    st.dataframe(
        filtered[show_cols],
        use_container_width=True,
        hide_index=True,
    )

    csv_download = filtered[show_cols].to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="⬇️ Descargar ranking en CSV",
        data=csv_download,
        file_name="ranking_site_selection.csv",
        mime="text/csv",
    )


with tab3:
    st.subheader("Visualizaciones")

    top10 = filtered.head(10).copy()
    top10["SCORE_TXT"] = top10["SCORE"].round(3)

    fig_bar = px.bar(
        top10.sort_values("SCORE", ascending=True),
        x="SCORE",
        y="NOMBRE_ZONA",
        orientation="h",
        text="SCORE_TXT",
        color="SCORE",
        title="Top 10 zonas por score",
    )
    fig_bar.update_layout(height=500, margin=dict(l=0, r=0, t=50, b=0))
    st.plotly_chart(fig_bar, use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        fig_scatter = px.scatter(
            filtered,
            x="ALQ_PRECIO_PIE2_ANUAL",
            y="MOVILIDAD_PROMEDIO_DIARIA",
            size="SCORE",
            color="SCORE",
            hover_name="NOMBRE_ZONA",
            title="Alquiler vs movilidad",
        )
        fig_scatter.update_layout(height=450, margin=dict(l=0, r=0, t=50, b=0))
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_b:
        fig_hist = px.histogram(
            filtered,
            x="SCORE",
            nbins=12,
            title="Distribución del score",
        )
        fig_hist.update_layout(height=450, margin=dict(l=0, r=0, t=50, b=0))
        st.plotly_chart(fig_hist, use_container_width=True)


with tab4:
    st.subheader("Metodología del modelo")

    st.markdown(
        """
**Lógica del score**

Se calcula un score multicriterio ponderado:

\\[
Score_i = \\sum_j w_j \\cdot x_{ij}^{norm}
\\]

- Los criterios **benefit** premian valores altos.
- Los criterios **cost** se invierten, para que valores bajos sean mejores.
- Los pesos se normalizan automáticamente según la suma total elegida.

**Criterios benefit**
- Densidad de población
- Ingreso mediano
- Cantidad de estaciones
- Movilidad promedio diaria

**Criterios cost**
- Alquiler
- Delito de propiedad
- Competencia directa

**Lectura**
- Un score más alto implica una zona más favorable según la ponderación seleccionada.
- El ranking cambia en tiempo real al mover filtros o pesos.
"""
    )
