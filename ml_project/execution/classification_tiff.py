import os
import glob
import joblib
import numpy as np
import pandas as pd
import rasterio
from rasterio.merge import merge
from rasterio.plot import show
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = RobustScaler()
        self.selector = None 
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        X_eng = self._engineer_features(X.copy())
        self.feature_names_in_ = X_eng.columns.tolist()
        self.scaler.fit(X_eng)
        return self

    def transform(self, X):
        X_eng = self._engineer_features(X.copy())
        if self.feature_names_in_:
            missing = set(self.feature_names_in_) - set(X_eng.columns)
            for c in missing:
                X_eng[c] = 0
            X_eng = X_eng[self.feature_names_in_]
        X_scaled = self.scaler.transform(X_eng)
        return pd.DataFrame(X_scaled, columns=X_eng.columns, index=X_eng.index)

    def _engineer_features(self, X):
        if 'NDVI_2023' in X.columns and 'NDVI_2022' in X.columns:
            X['NDVI_Delta_1yr'] = X['NDVI_2023'] - X['NDVI_2022']
        if 'NDVI_2024' in X.columns and 'NDVI_2020' in X.columns:
            X['NDVI_Delta_LongTerm'] = X['NDVI_2024'] - X['NDVI_2020']
        if 'NDVI_2024' in X.columns and 'NDVI_2023' in X.columns:
            X['NDVI_Ratio'] = X['NDVI_2024'] / (X['NDVI_2023'] + 0.01)
        if 'NIR' in X.columns and 'SWIR2' in X.columns:
            X['NBR'] = (X['NIR'] - X['SWIR2']) / (X['NIR'] + X['SWIR2'] + 0.0001)
        if 'elevation' in X.columns and 'slope' in X.columns:
            X['elev_slope_ratio'] = X['elevation'] / (X['slope'] + 1)
        return X

# ==========================================
# 2. CONFIGURACIÓN
# ==========================================

# La lista de bandas DEBE estar en el mismo orden que exportaste de GEE.
BANDS_ORDER = [
    'Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2',
    'NDVI', 'EVI', 'SAVI', 'NDWI',
    'NBR', 'MNDWI', 'GNDVI', 'NDMI', 'BSI', 'EVI2',
    'NDVI_2024', 'NDVI_2023', 'NDVI_2022',
    'NDVI_diff_2024_2023', 'NDVI_diff_2023_2022', 'NDVI_diff_2024_2022', 'NDVI_diff_2024_2020',
    'NDVI_ratio_2024_2023', 'NDVI_mean_3yr', 'NDVI_trend',
    'elevation', 'slope',
    'NIR_entropy', 'NIR_contrast', 'NIR_homogeneity', 'NIR_variance',
    'Red_entropy', 'Red_contrast', 'Red_homogeneity', 'Red_variance',
    'NDVI_entropy', 'NDVI_contrast', 'NDVI_homogeneity', 'NDVI_variance'
]

MODEL_PATH = '/home/sebastian/Documents/Programming/Python/scode-geoai-analysis/ml_project/notebooks/deforestation_model_optimized.pkl'
INPUT_PATTERN = 'cuenca_features_2024_para_prediccion*.tif'
OUTPUT_DIR = 'predicciones'

# ==========================================
# 3. LÓGICA DE PREDICCIÓN
# ==========================================

def predict_raster(tiff_path, model, bands):
    print(f"🔄 Procesando: {os.path.basename(tiff_path)}...")
    
    with rasterio.open(tiff_path) as src:
        # Leer datos (Bandas, Alto, Ancho)
        img_data = src.read()
        profile = src.profile
        
        # Crear máscara de datos válidos (donde no sea todo 0 o NaN)
        # Asumimos que si la primera banda es 0 o NaN, es fondo.
        # Ajusta esto si tus datos válidos pueden ser 0.
        if np.issubdtype(img_data.dtype, np.floating):
             mask = ~np.isnan(img_data[0])
        else:
             mask = (img_data[0] != 0) # O el valor de NoData que use GEE

        # Preparar datos para Scikit-Learn (Flattening)
        # Transformamos de (Bandas, H, W) -> (Píxeles, Bandas)
        c, h, w = img_data.shape
        reshaped_data = img_data.reshape(c, -1).T
        
        # Convertir a DataFrame con nombres correctos
        df = pd.DataFrame(reshaped_data, columns=bands)
        
        # Filtrar solo píxeles válidos para no predecir basura (ahorra memoria y tiempo)
        valid_pixels_idx = mask.reshape(-1)
        df_valid = df[valid_pixels_idx].copy()
        
        # Array vacío para resultados (lleno de 0 o 255 para NoData)
        prediction_flat = np.zeros(h * w, dtype=np.uint8) + 255 
        
        if len(df_valid) > 0:
            # PREDECIR
            print(f"   🔮 Prediciendo {len(df_valid)} píxeles...")
            preds = model.predict(df_valid)
            
            # Rellenar resultados
            prediction_flat[valid_pixels_idx] = preds
            
        # Volver a forma de imagen (1, H, W)
        prediction_img = prediction_flat.reshape(1, h, w)
        
        # Actualizar perfil para guardar (1 banda, uint8)
        profile.update(count=1, dtype=rasterio.uint8, nodata=255)
        
        # Guardar resultado temporal
        out_name = os.path.join(OUTPUT_DIR, "pred_" + os.path.basename(tiff_path))
        with rasterio.open(out_name, 'w', **profile) as dst:
            dst.write(prediction_img)
            
        return out_name

# ==========================================
# 4. EJECUCIÓN PRINCIPAL
# ==========================================

if __name__ == "__main__":
    # Crear carpeta de salida
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. Cargar Modelo
    print("📂 Cargando modelo...")
    try:
        model = joblib.load(MODEL_PATH)
        print("✅ Modelo cargado exitosamente.")
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        exit()

    # 2. Encontrar archivos TIFF
    tiff_files = glob.glob(INPUT_PATTERN)
    if not tiff_files:
        print("❌ No se encontraron archivos TIFF con el patrón especificado.")
        exit()
        
    print(f"📄 Archivos encontrados: {len(tiff_files)}")

    # 3. Predecir cada tile individualmente
    predicted_files = []
    for tiff in tiff_files:
        try:
            out_file = predict_raster(tiff, model, BANDS_ORDER)
            predicted_files.append(out_file)
        except Exception as e:
            print(f"⚠️ Error procesando {tiff}: {e}")

    # 4. Hacer el Mosaico (Collage)
    print("\n🧩 Uniendo (mosaicking) las predicciones...")
    
    src_files_to_mosaic = []
    for fp in predicted_files:
        src = rasterio.open(fp)
        src_files_to_mosaic.append(src)
    
    if src_files_to_mosaic:
        mosaic, out_trans = merge(src_files_to_mosaic)
        
        # Copiar metadatos del primer archivo
        out_meta = src_files_to_mosaic[0].meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans,
            "count": 1,
            "dtype": rasterio.uint8,
            "nodata": 255,
            "compress": "lzw" # Compresión para ahorrar espacio
        })
        
        final_output = "MAPA_FINAL_DEFORESTACION_2024.tif"
        with rasterio.open(final_output, "w", **out_meta) as dest:
            dest.write(mosaic)
            
        # Cerrar archivos abiertos
        for src in src_files_to_mosaic:
            src.close()
            
        print(f"\n🎉 ¡ÉXITO! Mapa final guardado como: {final_output}")
        
        # Visualización rápida
        plt.figure(figsize=(10, 10))
        plt.imshow(mosaic[0], cmap='viridis', vmin=0, vmax=2)
        plt.title("Resultado Final del Mosaico")
        plt.colorbar(ticks=[0, 1, 2], label='Clase')
        plt.show()
        
    else:
        print("❌ No se generaron predicciones para unir.")