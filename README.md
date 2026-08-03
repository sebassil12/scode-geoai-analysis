# GridCanopy — Motor de Riesgo de Vegetación con Sentinel-1 SAR

Pipeline de Machine Learning para clasificar cobertura vegetal en corredores de transmisión
eléctrica a partir de radar Sentinel-1 GRD. Se usa SAR en lugar de óptico porque el radar
atraviesa la nubosidad permanente de la sierra ecuatoriana, donde Sentinel-2 rara vez entrega
una escena limpia.

**Clases:** `1` Vegetación densa · `2` Vegetación baja/dispersa · `3` Suelo desnudo ·
`4` Construido. El valor `0` es *nodata*.

---

## Estructura

```
ml_project/
├── config.py                     corredores, fechas, parámetros S1, clases, rutas
├── core/
│   ├── gee_extractor.py          adquisición y pre-proceso Sentinel-1 en GEE
│   └── geotiff_io.py             toda la E/S ráster y las exportaciones desde GEE
├── features/
│   └── sar_engineering.py        índices SAR + GLCM (GEE) y polinomios + MI (sklearn)
├── modeling/
│   ├── train.py                  registro de modelos, tuning y serialización
│   ├── benchmark.py              campeón vs. retador (RF / XGBoost / LightGBM)
│   ├── evaluate.py               métricas compartidas, matriz de confusión, PR
│   └── wrappers.py               adaptadores que acaban dentro del pickle
├── inference/
│   ├── classify.py               inferencia local por bloques sobre GeoTIFF
│   └── vectorizer.py             ráster clasificado → GeoJSON / PostGIS + áreas
└── test_pipeline.py              auto-verificación de extremo a extremo (sin red)
```

Reglas que sostienen la estructura:

* **`config.py` es la única fuente de verdad.** Ningún otro módulo contiene rutas, fechas ni
  números mágicos. Las variables de entorno (`DATA_DIR`, `MODEL_DIR`, `COMPOSITE_START`, …)
  sobrescriben los valores por defecto.
* **`geotiff_io.py` es el único módulo que abre un ráster.** El clasificador y el vectorizador
  trabajan sobre arrays; el tamaño de bloque, el perfil COG y la convención de *nodata* se
  deciden en un solo sitio.
* **`train.py` define cómo se afina un modelo; `benchmark.py` sólo los enfrenta.** Así un
  modelo se tunea igual esté compitiendo o entrenándose para producción.

---

## Instalación

```bash
conda env create -f environment.yml
conda activate geo-ml
pip install xgboost lightgbm imbalanced-learn   # requeridos por benchmark.py
```

---

## Ejecución secuencial

```bash
# 1. EXTRACCIÓN — muestras de entrenamiento (etiquetas ESA WorldCover) y composite SAR
python -m ml_project.core.geotiff_io --corridor cuenca-molleturo --output data/raw/ --mode samples
python -m ml_project.core.geotiff_io --corridor cuenca-molleturo --output data/raw/

# 2. BENCHMARK — entrena RF, XGBoost y LightGBM y serializa el campeón
python -m ml_project.modeling.benchmark --data data/raw/ --output models/

# 3. ENTRENAMIENTO FINAL — opcional, para reentrenar sólo el modelo ganador
python -m ml_project.modeling.train --data data/raw/ --model xgboost --output models/champion.joblib

# 4. INFERENCIA — ráster clasificado + polígonos de riesgo + estadísticas
python -m ml_project.inference.classify \
    --input data/raw/cuenca-molleturo.tif \
    --model models/champion.joblib \
    --geojson risk.geojson --stats stats.json \
    --corridor cuenca-molleturo --max-distance-m 30

# 5. VECTORIZACIÓN aislada — sobre un ráster ya clasificado
python -m ml_project.inference.vectorizer --input corridor_classified.tif --geojson risk.geojson

# Verificación (no requiere GEE ni red)
python -m ml_project.test_pipeline
```

El benchmark deja en `models/`: `champion.joblib`, `champion_metadata.json`,
`benchmark_report.json`, y por cada candidato una matriz de confusión, una curva
precisión-recall y su `metrics.json`.

---

## Selección de modelo

`benchmark.py` afina cada candidato con el mismo presupuesto de búsqueda
(`RandomizedSearchCV`, 5-fold estratificado) y los evalúa sobre **una sola** partición
train/test creada antes de construir ningún modelo — repartir por modelo dejaría que una
partición afortunada, y no un mejor algoritmo, eligiera al campeón.

Se ordena por **F1 macro** en holdout, con **PR-AUC macro** como desempate. F1 lidera porque
una orden de trabajo emitida en falso y un riesgo no detectado cuestan ambos dinero real, así
que precisión y exhaustividad pesan igual.

Balanceo de clases:

| `--balance` | Comportamiento | Dependencia |
|---|---|---|
| `class_weight` (por defecto) | RF `balanced_subsample`, LightGBM `balanced`. XGBoost no tiene equivalente multiclase y se avisa por log. | ninguna |
| `smotetomek` | Remuestreo dentro de cada fold, uniforme para los tres modelos. | `imbalanced-learn` |

---

## Notas técnicas

* `COPERNICUS/S1_GRD` en GEE ya viene con órbita precisa, eliminación de ruido térmico y de
  borde, calibración radiométrica a sigma0 (dB) y corrección de terreno contra SRTM. El
  pipeline sólo añade filtro de speckle y composición mediana; reimplementar lo demás sería
  deshacer trabajo ya hecho.
* El filtro de speckle promedia en **potencia lineal**, no en dB: la media de logaritmos no es
  el logaritmo de la media y sesga el resultado a la baja.
* `VV_VH_ratio` y `RVI` también se calculan en potencia lineal. Un cociente de logaritmos no
  significa nada.
* Las texturas GLCM usan ventana 3×3. Ventanas mayores difuminan el borde entre la franja de
  servidumbre y el bosque, que es justo lo que el modelo debe encontrar.
* Las etiquetas de entrenamiento provienen de ESA WorldCover v200 remapeado. Es un proxy
  aproximado de un levantamiento de campo: sirve como línea base, no como exactitud validada.
* `ZeroIndexedClassifier` (en `modeling/wrappers.py`) existe porque XGBoost exige etiquetas
  `0..n-1`. Remapear globalmente filtraría esa restricción al config, a los metadatos y a la
  inferencia; el envoltorio la contiene para que la clase `1` siga significando "Vegetación
  densa" en todas partes.
* **`wrappers.py` no tiene bloque `__main__` a propósito.** Una clase definida en un módulo
  ejecutado con `python -m` se serializa como `__main__.NombreDeClase`, y cualquier otro punto
  de entrada falla al cargar el modelo con `AttributeError`. Todo lo que acabe dentro de un
  pickle debe vivir en un módulo que sólo se importe. `test_pipeline.py` lo verifica lanzando
  las CLI en subprocesos reales.
* `SEARCH_N_JOBS=-1` con `MODEL_N_JOBS=1`: paralelizar a la vez en la búsqueda y en el
  estimador sobresuscribe la máquina (16 núcleos → 16×16 hilos compitiendo) y el benchmark
  acaba siendo más lento que en un solo hilo.

---

## Calidad de código

```bash
ruff check ml_project/        # lint
ruff format ml_project/       # formato (compatible con Black, línea de 100)
python -m ml_project.test_pipeline
```

La configuración de `ruff` y `black` vive en `pyproject.toml`.
