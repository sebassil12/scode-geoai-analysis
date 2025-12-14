# Proyecto de Clasificación Geoespacial con Machine Learning

Este repositorio contiene el código fuente y la documentación para la clasificación de cobertura terrestre utilizando imágenes satelitales procesadas en Google Earth Engine (GEE) y modelos de Machine Learning (Random Forest) en Python.

---

## 🛠️ Parte 1: Documentación Técnica

### Estructura del Proyecto

El proyecto está organizado en módulos funcionales para generación de datos, entrenamiento y ejecución:

*   **`ml_project/notebooks/training_model_rf.ipynb`**: Notebook principal que orquesta el flujo de trabajo. Contiene la lógica de conexión con GEE, extracción de datos y ejecución del pipeline de entrenamiento.
*   **`ml_project/notebooks/training_utils.py`**: Librería de utilidades críticas. Incluye:
    *   `FeatureEngineer`: Transformador personalizado para ingeniería de características (polinomios, interacciones).
    *   `train_and_evaluate_model_optimized`: Función robusta que implementa `RandomizedSearchCV`, optimización de umbrales y manejo de desbalanceo con `SMOTETomek`.
*   **`ml_project/notebooks/create_matrix.ipynb`**: Generación y visualización de la Matriz de Confusión para evaluar el rendimiento del modelo.
*   **`ml_project/notebooks/create_graph_importance_features.ipynb`**: Visualización de la importancia de variables (Feature Importance), crucial para la interpretabilidad.
*   **`ml_project/notebooks/export_geotiff.ipynb`**: Script para exportar mosaicos o imágenes procesadas desde GEE en formato GeoTIFF para su posterior clasificación local.
*   **`ml_project/execution/`**: Carpeta con scripts para la puesta en producción.
    *   `classification_tiff.py`: Script para cargar el modelo entrenado y clasificar nuevas imágenes GeoTIFF.

### Requisitos e Instalación

Este proyecto utiliza **Conda** para la gestión de entornos. Todas las dependencias necesarias se encuentran en el archivo `environment.yml`.

1.  **Clonar el repositorio:**
    ```bash
    git clone <url-del-repositorio>
    cd scode-geoai-analysis
    ```

2.  **Crear el entorno virtual:**
    ```bash
    conda env create -f environment.yml
    ```

3.  **Activar el entorno:**
    ```bash
    conda activate geo-ml
    ```

### Guía de Reproducción

Para replicar los resultados del estudio, siga este orden de ejecución:

1.  **Preparación de Datos (GEE)**:
    Asegúrese de tener acceso a Google Earth Engine. Ejecute `ml_project/notebooks/export_geotiff.ipynb` si necesita descargar nuevas imágenes para inferencia.

2.  **Entrenamiento del Modelo**:
    Ejecute `ml_project/notebooks/training_model_rf.ipynb`. Este notebook utilizará las funciones optimizadas de `training_utils.py` para:
    *   Preprocesar los datos.
    *   Generar nuevas características (Feature Engineering).
    *   Optimizar hiperparámetros del Random Forest buscando maximizar el F1-Score.
    *   Guardar el modelo entrenado (`.pkl`) y sus metadatos.

3.  **Evaluación**:
    *   Ejecute `ml_project/notebooks/create_matrix.ipynb` para inspeccionar los errores de clasificación.
    *   Ejecute `ml_project/notebooks/create_graph_importance_features.ipynb` para crear gráficos que muestren qué bandas espectrales o índices tienen mayor peso.

4.  **Clasificación (Inferencia)**:
    Utilice los scripts en `ml_project/execution/` para generar mapas temáticos a partir de los GeoTIFFs exportados.

---
