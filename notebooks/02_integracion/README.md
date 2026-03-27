# 🔗 Notebooks de Integración

Este directorio contiene los **notebooks de integración** utilizados en la fase intermedia del pipeline de datos del proyecto.

Su función principal es tomar las fuentes ya depuradas en **`datos/limpios/`** y transformarlas en entidades estructuradas y coherentes entre sí, preparadas para su uso dentro de la lógica de base de datos y para la construcción posterior del dataset maestro.

En esta fase se desarrollan procesos como:

- la consolidación de entidades temáticas,
- la integración territorial de las distintas fuentes,
- la homologación mediante una unidad espacial común,
- la generación de tablas estructuradas,
- y la preparación de la información para su implementación en la base de datos del proyecto.

El notebook incluido en esta carpeta es:

- **`10_ETL_Manhattan_Master_Process.ipynb`**: integración territorial de las distintas fuentes del proyecto, consolidación de entidades temáticas y generación de las tablas estructuradas que alimentan la base de datos.

Los resultados de esta fase se almacenan en **`datos/base_datos/`** y constituyen la base común a partir de la cual se construye posteriormente el dataset maestro.
