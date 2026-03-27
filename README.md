# TFM Grupo 7 Big Data 2026

Repositorio técnico del **Trabajo de Fin de Máster del Grupo 7** del Máster en **Big Data & Business Intelligence**, orientado al desarrollo de un **sistema de Location Intelligence** para el análisis territorial de Manhattan.

Este repositorio reúne la **memoria técnica y la trazabilidad computacional** del proyecto, incluyendo la organización de las fuentes, los notebooks de procesamiento en Python, la implementación de la base de datos en SQL y la construcción del dataset maestro.

## 📂 Estructura del repositorio

- **`datos/`**: organización de los datos según su etapa de procesamiento.
  - **`crudos/`**: fuentes originales del proyecto.
  - **`limpios/`**: salidas del proceso de limpieza.
  - **`base_datos/`**: entidades estructuradas para su implementación en SQL.
  - **`maestro/`**: versiones finales del dataset maestro.
- **`notebooks/`**: notebooks de Python empleados en las fases de limpieza, integración y construcción del dataset maestro.
- **`scripts/`**: espacio reservado para las fases posteriores del proyecto en R, incluyendo clustering, scoring y escenarios.
- **`sql/`**: script de implementación de la base de datos y modelo relacional.
- **`resultados/`**: carpeta destinada a almacenar salidas y resultados de fases analíticas posteriores.
- **`anexos/`**: documentación complementaria vinculada al TFM.

## 🧭 Alcance

La presente versión del repositorio documenta y almacena los siguientes materiales desarrollados del TFM:

- recopilación de fuentes,
- limpieza y transformación inicial de datos,
- integración territorial,
- estructuración de entidades para base de datos,
- y consolidación del dataset maestro.

Las fases posteriores de **clustering, scoring y escenarios** se incorporarán en etapas siguientes del proyecto.

## ⚙️ Requisitos previos

Para reproducir localmente esta parte del proyecto se recomienda contar con:

- **Python 3.10 o superior**
- **Jupyter Notebook** o **JupyterLab**
- **MySQL Server 8.x**
- **MySQL Workbench** como entorno recomendado para la parte SQL

Las librerías necesarias para la ejecución de los notebooks se encuentran recogidas en el archivo **`requirements.txt`** del repositorio.

## ▶️ Orden general de ejecución del pipeline en Python

La ejecución de los notebooks sigue la lógica del flujo de datos del proyecto:

### 1. Limpieza

Ejecutar los notebooks de la carpeta:

**`notebooks/01_limpieza/`**

Estos notebooks toman como entrada los archivos de **`datos/crudos/`** y generan las salidas correspondientes en **`datos/limpios/`**.

### 2. Integración

Ejecutar los notebooks de la carpeta:

**`notebooks/02_integracion/`**

Este proceso toma como entrada los archivos de **`datos/limpios/`** y genera las entidades estructuradas almacenadas en **`datos/base_datos/`**.

### 3. Dataset maestro

Ejecutar los notebooks de la carpeta:

**`notebooks/03_dataset_maestro/`**

Estos notebooks toman como entrada los archivos de **`datos/base_datos/`** y generan las distintas versiones del dataset maestro en **`datos/maestro/`**.

## 🗄️ Reproducción de la base de datos SQL

La implementación SQL del proyecto se encuentra en la carpeta **`sql/`** y se apoya en las tablas contenidas en **`datos/base_datos/`**.

La carpeta **`sql/`** incluye además el archivo **`Modelo Relacion - Entidad.mwb`**, editable en MySQL Workbench.

## 🔗 Relación entre carpetas del proyecto

La trazabilidad general del repositorio sigue esta secuencia:

**`datos/crudos/`** → **`notebooks/01_limpieza/`** → **`datos/limpios/`**  
**`datos/limpios/`** → **`notebooks/02_integracion/`** → **`datos/base_datos/`**  
**`datos/base_datos/`** → **`notebooks/03_dataset_maestro/`** → **`datos/maestro/`**

## 📌 Nota final

Este repositorio ha sido organizado con fines de **trazabilidad, documentación técnica y reproducibilidad local** del proyecto. Los README incluidos en cada carpeta explican con mayor detalle el contenido y la función de cada etapa dentro del flujo de trabajo.
