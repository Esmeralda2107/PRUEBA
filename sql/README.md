 # 🗄️ Implementación SQL del Proyecto

Este directorio reúne los archivos necesarios para implementar localmente la **base de datos relacional** del proyecto en **MySQL Server**, utilizando como interfaz recomendada **MySQL Workbench**.

La finalidad de esta carpeta es permitir la **reproducción local** de la estructura SQL desarrollada para el TFM, a partir de las tablas ya consolidadas en **`datos/base_datos/`**. El modelo implementado responde a una lógica de **esquema estrella**, en la que la tabla **`ZONAS`** actúa como entidad central y permite relacionar el resto de dimensiones del sistema.

## 📂 Contenido de esta carpeta

- **`Estructura_DB_TFM_Manhattan.sql`**: script principal de creación de la base de datos.
  - La **primera parte** del archivo crea la base de datos, las tablas y sus relaciones.
  - La **segunda parte** contiene consultas de verificación que deben ejecutarse **solo después** de haber cargado todas las tablas.
- **`Modelo Relacion - Entidad.mwb`**: archivo editable del modelo relacional desarrollado en **MySQL Workbench**.

## ⚙️ Requisitos previos

Antes de ejecutar esta parte del proyecto, se recomienda tener instalado:

- **MySQL Server 8.x**
- **MySQL Workbench** (o cualquier cliente SQL compatible con MySQL)
- El repositorio descargado y organizado con la estructura indicada

> Este flujo fue planteado para trabajar localmente con **MySQL Server** y **MySQL Workbench**, por lo que ese entorno es el recomendado para su reproducción.

## 🚀 Guía de ejecución paso a paso

### 1. Crear la base de datos y las tablas

1. Abrir **MySQL Workbench**.
2. Conectarse al servidor local de **MySQL Server**.
3. Abrir el archivo:

   **`sql/Estructura_DB_TFM_Manhattan.sql`**

4. Ejecutar **únicamente la primera parte del script**, correspondiente a:
   - creación de la base de datos,
   - creación de las tablas,
   - y definición de las claves primarias y foráneas.

> **Importante:**  
> No ejecutar todavía la segunda parte del script, ya que contiene las consultas de verificación de carga.

### 2. Seleccionar la base de datos creada

Después de ejecutar la primera parte del script:

1. Actualizar la lista de esquemas en MySQL Workbench.
2. Ubicar la base de datos creada:

   **`TFM_SiteSelection_Manhattan`**

3. Seleccionarla como esquema activo antes de importar los archivos.


### 3. Preparar las tablas para la carga

Los datos que alimentan esta base se encuentran en la carpeta:

**`datos/base_datos/`**

Antes de importarlos, se recomienda verificar que:

- cada archivo CSV corresponda con su tabla SQL,
- los nombres de las columnas coincidan con los atributos definidos en la base de datos,
- y los tipos de datos sean coherentes con la estructura del script.

## 📥 Orden de carga recomendado

Para mantener la integridad de las relaciones entre tablas, los archivos deben cargarse en el siguiente orden:

1. **`ZONAS.csv`**
2. **`CENSO.csv`**
3. **`COSTO_ALQUILER.csv`**
4. **`MOVILIDAD.csv`**
5. **`SEGURIDAD.csv`**
6. **`RESTAURANTES.csv`**
7. **`LUGARES_INTERES.csv`**

> **Nota importante:**  
> La tabla **`ZONAS`** debe cargarse primero, ya que constituye la **unidad primaria del modelo** y permite la relación entre el resto de tablas mediante el campo **`ID_ZONA`**.

## 4. Importar los archivos CSV en MySQL Workbench

Para cada tabla:

1. Localizar la tabla correspondiente dentro de la base de datos.
2. Hacer clic derecho sobre la tabla.
3. Seleccionar **Table Data Import Wizard**.
4. Elegir el archivo CSV correspondiente desde la carpeta **`datos/base_datos/`**.
5. Confirmar el delimitador de columnas.
6. Verificar que los nombres de las columnas del archivo coincidan con los atributos de la tabla SQL.
7. Ejecutar la importación.

Se recomienda revisar cada tabla después de la carga para confirmar que los registros fueron importados correctamente.

## 5. Ejecutar las consultas de verificación

Una vez cargadas todas las tablas:

1. Volver al archivo:

   **`sql/Estructura_DB_TFM_Manhattan.sql`**

2. Ejecutar ahora la **segunda parte del script**, correspondiente a las consultas de verificación.

Estas consultas permiten:

- comprobar que las tablas contienen registros,
- validar la carga de datos,
- y verificar que la estructura general del modelo fue implementada correctamente.

## 🧩 Estructura lógica y finalidad del modelo

La base de datos implementa un **modelo estrella**, en el que **`ZONAS`** funciona como tabla principal y el resto de tablas se relacionan con ella a través de **`ID_ZONA`**.

Este diseño permite integrar información territorial de distintas dimensiones en una estructura relacional común, facilitando la consulta consistente de los datos por zona, la validación de relaciones entre tablas y su explotación analítica posterior.

Como apoyo complementario, la carpeta **`anexos/`** incluye el archivo **`esquema_modelo_logico_tfm.pdf`**, que presenta una representación visual del desarrollo del modelo lógico del sistema.

En conjunto, esta carpeta documenta la capa SQL del proyecto y sirve como soporte técnico para la implementación local del modelo de datos desarrollado en el TFM.
