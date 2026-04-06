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
- Los valores inferiores al percentil 5 se acotan al percentil 5 y los valores superiores al percentil 95 se acotan al percentil 95 antes de calcular la puntuación.
- **Sentido directo**: valores más altos reciben puntuaciones más cercanas a 100.
- **Sentido inverso**: valores más altos reciben puntuaciones más cercanas a 0.

**Regla macro**
- Dimensiones principales: **60 %**
- Dimensiones de contexto: **40 %**
- La app permite modificar varias dimensiones de forma acumulativa dentro de cada bloque y reajusta automáticamente las demás para conservar esa lógica.

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
