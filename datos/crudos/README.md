# 🗂️ Datos Crudos

Este directorio reúne las **fuentes originales** utilizadas en el proyecto como punto de partida para la construcción del sistema de datos del TFM. Su contenido corresponde a la etapa **cruda**, por lo que los archivos aquí almacenados conservan, en la medida de lo posible, su estructura de origen y no han sido sometidos todavía al proceso completo de limpieza o transformación analítica.

Las fuentes crudas se organizan por dimensión temática:

- **`censo/`**: información sociodemográfica base de Manhattan, incluyendo población, edad, origen hispano, estructura del hogar, empleo, ingresos, ocupación de vivienda y renta residencial.
- **`competencia/`**: registros originales de establecimientos gastronómicos e inspecciones, utilizados para construir la dimensión de competencia.
- **`seguridad/`**: registros crudos de incidentes y denuncias policiales empleados en la dimensión de seguridad urbana.
- **`movilidad/`**: archivos originales del sistema de transporte, utilizados para estimar intensidad de movilidad y accesibilidad territorial.
- **`lugares_interes/`**: base original de equipamientos y puntos de interés urbanos relevantes para el análisis espacial.
- **`zonas/`**: cartografía base en formato **GeoJSON** con las unidades territoriales de Manhattan empleadas para la integración espacial del resto de fuentes.

Los archivos contenidos en este directorio constituyen las **entradas de los notebooks de limpieza** desarrollados en Python y organizados en la carpeta **`notebooks/01_limpieza/`** del repositorio.

Esta carpeta representa la **primera capa de datos del proyecto**. A partir de estos archivos se desarrollan las etapas posteriores de:

- limpieza y depuración,
- reestructuración de variables,
- homologación de formatos,
- integración territorial,
- y consolidación del dataset maestro.
