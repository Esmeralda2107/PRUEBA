# 🧠 Notebooks de Dataset Maestro

Este directorio contiene los **notebooks dedicados a la construcción y preparación del dataset maestro** del proyecto.

Su función principal es tomar las entidades ya organizadas en **`datos/base_datos/`** y combinarlas en una única estructura analítica, que permita caracterizar de forma conjunta las zonas de Manhattan según las distintas dimensiones del estudio.

En esta fase se desarrollan procesos como:

- la unión de entidades temáticas,
- la validación de consistencia del conjunto integrado,
- la generación de versiones sucesivas del dataset maestro,
- la creación de métricas derivadas,
- la depuración final para análisis,
- y la estandarización de variables para etapas posteriores del proyecto.

Los notebooks incluidos en esta carpeta son:

- **`11_Consolidacion_Master_Dataset_ML.ipynb`**: consolidación de las entidades integradas en una primera versión del dataset maestro.
- **`12_Preparacion_y_Analisis_Estrategico.ipynb`**: preparación analítica final del dataset, generación de métricas derivadas y construcción de las versiones finales para análisis.

Los resultados de esta fase se almacenan en **`datos/maestro/`** y constituyen la salida final de la etapa de construcción de datos, así como el insumo principal para las siguientes fases de **clustering, scoring y escenarios**.
