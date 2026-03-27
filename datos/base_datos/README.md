# 🗂️ Base de Datos

Este directorio reúne las **entidades estructuradas** del proyecto en la etapa de base de datos, resultado del procesamiento e integración territorial aplicado sobre las fuentes limpias.

Los archivos contenidos en esta carpeta corresponden a una versión ya consolidada de la información, organizada según la lógica del modelo de datos del proyecto y preparada para su implementación en SQL y para la posterior construcción del dataset maestro.

Las entidades se organizan por dimensión temática:

- **`alquiler/`**: tabla estructurada con la información de precio de alquiler comercial por zona.
- **`censo/`**: tabla consolidada con los indicadores sociodemográficos relevantes para el análisis.
- **`competencia/`**: tabla estructurada con los establecimientos clasificados como competencia dentro del área de estudio.
- **`seguridad/`**: tabla con los registros delictivos ya organizados para su agregación y análisis territorial.
- **`movilidad/`**: tabla con la información de estaciones y métricas agregadas de movilidad.
- **`lugares_interes/`**: tabla estructurada de equipamientos y puntos de interés con utilidad analítica.
- **`zonas/`**: tabla territorial base con los identificadores y atributos espaciales de referencia.

Los archivos contenidos en este directorio son el **producto de la ejecución de los notebooks de integración** desarrollados en Python y organizados en la carpeta **`notebooks/02_integracion/`** del repositorio.

Esta carpeta representa la etapa en la que las fuentes ya limpias pasan a una estructura común y consistente, sirviendo como insumo directo para la implementación del modelo relacional y para la posterior construcción del **dataset maestro**, desarrollada en la carpeta **`notebooks/03_dataset_maestro/`**.
