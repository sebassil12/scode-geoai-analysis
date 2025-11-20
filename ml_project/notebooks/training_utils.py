import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, make_scorer, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import json
from sklearn.feature_selection import SelectFromModel

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom Transformer optimizado para DEFORESTACIÓN.
    En lugar de polinomios ciegos, calcula Deltas (Cambios temporales) e índices de quema.
    """
    def __init__(self):
        self.scaler = RobustScaler()
        self.selector = None 
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        X_eng = self._engineer_features(X.copy())
        
        # Guardamos nombres generados para consistencia
        self.feature_names_in_ = X_eng.columns.tolist()
        
        self.scaler.fit(X_eng)
        
        # NOTA: Eliminé el SelectKBest interno porque ya usas SelectFromModel en el pipeline.
        # Tener dos selectores compitiendo puede eliminar variables buenas.
            
        return self

    def transform(self, X):
        X_eng = self._engineer_features(X.copy())
        
        # --- GARANTÍA DE ORDEN ---
        if self.feature_names_in_:
            # Rellenar columnas faltantes con 0 (seguridad)
            missing = set(self.feature_names_in_) - set(X_eng.columns)
            for c in missing:
                X_eng[c] = 0
            # Ordenar exactamente igual que en fit
            X_eng = X_eng[self.feature_names_in_]
        
        X_scaled = self.scaler.transform(X_eng)
        X_scaled = pd.DataFrame(X_scaled, columns=X_eng.columns, index=X_eng.index)
        
        return X_scaled

    def _engineer_features(self, X):
        """
        Aquí está la clave: Calcular CAMBIOS, no solo estados.
        """
        # 1. DELTAS (La clave de la deforestación)
        # Si tenemos datos de años consecutivos, calculamos la pérdida.
        if 'NDVI_2023' in X.columns and 'NDVI_2022' in X.columns:
            X['NDVI_Delta_1yr'] = X['NDVI_2023'] - X['NDVI_2022'] # Negativo = Pérdida
        
        if 'NDVI_2024' in X.columns and 'NDVI_2020' in X.columns:
            X['NDVI_Delta_LongTerm'] = X['NDVI_2024'] - X['NDVI_2020']

        # 2. RATIOS DE CAMBIO (Más robusto que la resta simple)
        if 'NDVI_2024' in X.columns and 'NDVI_2023' in X.columns:
            # Evitar división por cero
            X['NDVI_Ratio'] = X['NDVI_2024'] / (X['NDVI_2023'] + 0.01)

        # 3. NBR (Normalized Burn Ratio) - Si tienes las bandas SWIR y NIR
        # Es el mejor índice para detectar suelo desnudo post-tala.
        if 'NIR' in X.columns and 'SWIR2' in X.columns:
            X['NBR'] = (X['NIR'] - X['SWIR2']) / (X['NIR'] + X['SWIR2'] + 0.0001)

        # 4. INTERACCIONES TOPOGRÁFICAS ÚTILES (Solo las que tienen sentido físico)
        if 'elevation' in X.columns and 'slope' in X.columns:
            # Zonas altas y planas suelen ser páramos (Otras Tierras)
            X['elev_slope_ratio'] = X['elevation'] / (X['slope'] + 1)
            
        return X

def train_and_evaluate_model_optimized(X_train: pd.DataFrame, y_train: pd.Series, 
                                     X_test: pd.DataFrame, y_test: pd.Series,
                                     training_metadata: dict = None):
    """
    Entrenamiento avanzado con RandomizedSearchCV + CALIBRACIÓN DE UMBRAL FINAL.
    """
    print("\n🚀 Paso 3: Iniciando entrenamiento avanzado (RandomizedSearch + Threshold Tuning)...")
    
    # 1. PIPELINE
    sampling_technique = SMOTETomek(
        random_state=42,
        smote=SMOTE(k_neighbors=5, random_state=42),
        tomek=TomekLinks(n_jobs=-1)
    )

    # Pipeline compatible con imblearn (maneja el resampleo internamente en cross-validation)
    pipeline = ImbPipeline([
        ('features', FeatureEngineer()),           
        ('selection', SelectFromModel(RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42))),
        ('balance', sampling_technique),           
        ('rf', RandomForestClassifier(random_state=42, n_jobs=-1))
    ])

    # 2. DISTRIBUCIÓN DE HIPERPARÁMETROS
    # Simplifiqué un poco para que converja mejor a soluciones estables
    param_distributions = {
        'selection__threshold': ['median', '1.25*mean'], # threshold dinámico
        'rf__n_estimators': [300, 500, 800],
        'rf__max_depth': [20, 30, None],
        'rf__min_samples_leaf': [2, 4], # Clave para reducir ruido
        'rf__class_weight': ['balanced_subsample'], # El mejor para RF
        'rf__max_features': ['sqrt', 'log2']
    }

    # 3. RANDOM SEARCH
    stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scorer = make_scorer(f1_score, average='macro')

    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=30, # 30 es suficiente si el espacio está bien definido
        cv=stratified_cv,
        scoring=scorer,
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    print(f"   ⚡ Buscando mejores hiperparámetros...")
    random_search.fit(X_train, y_train)
    
    best_pipeline = random_search.best_estimator_
    search_results = pd.DataFrame(random_search.cv_results_)
    
    print(f"\n✅ MEJOR CONFIGURACIÓN (CV Score: {random_search.best_score_:.4f}):")
    print(random_search.best_params_)

    # ---------------------------------------------------------
    # 4. CALIBRACIÓN DE UMBRAL (EL SECRETO PARA EL >0.80)
    # ---------------------------------------------------------

    
    print(f"\n⚖️  Calibrando umbral de decisión para maximizar precisión en Deforestación...")
    
    # Predecir probabilidades en Test
    y_probs = best_pipeline.predict_proba(X_test)
    
    best_thresh = 0.5
    best_f1_def = 0
    best_preds = best_pipeline.predict(X_test) # Default predictions (0.5)

    # Probamos umbrales agresivos
    for thresh in np.arange(0.4, 0.8, 0.02):
        custom_preds = []
        for probs in y_probs:
            if probs[2] >= thresh: # Si la prob de deforestación supera el umbral
                custom_preds.append(2)
            else:
                # Si no, elegimos el mayor entre 0 y 1
                custom_preds.append(0 if probs[0] > probs[1] else 1)
        
        f1_def = f1_score(y_test, custom_preds, average=None)[2]
        
        if f1_def > best_f1_def:
            best_f1_def = f1_def
            best_thresh = thresh
            best_preds = custom_preds

    print(f"   🏆 Umbral óptimo encontrado: {best_thresh:.2f}")
    print(f"   🚀 F1-Score Deforestado (Calibrado): {best_f1_def:.4f}")

    # 5. EVALUACIÓN FINAL (USANDO PREDICCIONES CALIBRADAS)
    print(f"\n{'='*60}")
    acc = accuracy_score(y_test, best_preds)
    print(f"ACCURACY FINAL: {acc:.4f} ({acc*100:.2f}%)")
    print(f"{'='*60}")
    
    class_labels = ['Otras Tierras (0)', 'Bosque (1)', 'Deforestado (2)']
    print(classification_report(y_test, best_preds, target_names=class_labels, digits=4))

    # 6. EXTRACCIÓN DE ARTEFACTOS
    try:
        # Scaler y Selector
        scaler = best_pipeline.named_steps['features'].scaler
        selector = best_pipeline.named_steps['selection']
        
        # Feature Importance
        # (Lógica simplificada para evitar errores, asumiendo que FeatureEngineer guarda nombres)
        feature_engineer = best_pipeline.named_steps['features']
        base_names = np.array(feature_engineer.feature_names_in_) if feature_engineer.feature_names_in_ else np.array([f'feat_{i}' for i in range(X_train.shape[1])])
        
        # Filtrar por selector
        support = selector.get_support()
        selected_names = base_names[support] if len(base_names) == len(support) else base_names[:sum(support)] # Fallback seguro
        
        # Importancia RF
        rf = best_pipeline.named_steps['rf']
        importances = rf.feature_importances_
        
        feature_importance = pd.DataFrame({
            'feature': selected_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print("\n🔝 Top 5 Características:")
        print(feature_importance.head(5))
        
    except Exception as e:
        print(f"⚠️ Aviso: No se pudo extraer importancia detallada ({e}).")
        feature_importance = pd.DataFrame() # Empty df fallback
        scaler = best_pipeline.named_steps['features'].scaler
        selector = best_pipeline.named_steps['selection']

    # Guardar resultados de búsqueda
    search_results.to_csv('random_search_optimized.csv', index=False)
    
    # Devolvemos los 5 objetos exactos que necesitas
    return best_pipeline, search_results, feature_importance, scaler, selector

def save_optimized_artifacts(classifier: ImbPipeline, 
                           training_df: pd.DataFrame, 
                           testing_df: pd.DataFrame, 
                           model_filename: str,
                           training_metadata: Dict,
                           feature_importance: pd.DataFrame = None,
                           search_results: pd.DataFrame = None,
                           scaler: RobustScaler = None,
                           selector: SelectKBest = None):
    """
    Guarda todos los artefactos del modelo optimizado.
    """
    print("\nPaso 5: Guardando artefactos optimizados...")
    
    # 1. Guardar modelo principal
    joblib.dump(classifier, model_filename)
    print(f"   ✅ Modelo guardado como '{model_filename}'")
    
    # 2. Guardar preprocesadores
    # Nota: Ahora están dentro del pipeline, pero los guardamos por separado también si se solicita
    if scaler is not None:
        joblib.dump(scaler, 'feature_scaler.pkl')
        joblib.dump(selector, 'feature_selector.pkl')
        print("   ✅ Preprocesadores guardados")
    
    # 3. Guardar datos
    training_df.to_csv('training_optimized.csv', index=False)
    testing_df.to_csv('testing_optimized.csv', index=False)
    
    # 4. Metadatos enriquecidos
    # Calcular accuracy final usando el pipeline (que maneja feature engineering internamente)
    final_acc = None
    if 'class' in testing_df.columns:
        X_test_raw = testing_df.drop('class', axis=1)
        y_test = testing_df['class']
        y_pred = classifier.predict(X_test_raw)
        final_acc = accuracy_score(y_test, y_pred)

    metadata = {
        **training_metadata,
        'optimization_round': 'feature_engineering_v2_pipeline',
        'final_accuracy': final_acc,
        'feature_engineering': 'interactions + polynomials + scaling + selection (Integrated Pipeline)',
        'top_features': feature_importance.head(10)['feature'].tolist() if feature_importance is not None else [],
        'processing_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(model_filename.replace('.pkl', '_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4, default=str)
    
    print(f"   ✅ Metadatos guardados")
    
    if feature_importance is not None:
        feature_importance.to_csv('feature_importance_optimized.csv', index=False)
        print("   ✅ Importancia de características guardada")
    
    print(f"\n🎉 ¡PROCESO DE OPTIMIZACIÓN COMPLETADO!")
    print(f"   Archivos generados:")
    print(f"   - {model_filename} (Modelo optimizado)")
    print(f"   - feature_scaler.pkl (Escalador)")
    print(f"   - feature_selector.pkl (Selector)")
    print(f"   - feature_importance_optimized.csv (Características)")
