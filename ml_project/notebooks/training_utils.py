import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks
from imblearn.pipeline import Pipeline
import joblib
import json

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom Transformer for advanced feature engineering.
    Encapsulates interaction terms, polynomials, and scaling.
    """
    def __init__(self):
        self.important_features = ['elevation', 'NDVI_mean_3yr', 'Red_contrast', 'NDVI_contrast', 'slope']
        self.scaler = RobustScaler()
        self.selector = None 
        self.selected_features_names = None

    def fit(self, X, y=None):
        # 1. Generate engineered features for a sample to fit the scaler and selector
        X_eng = self._engineer_features(X.copy())
        
        # 2. Fit Scaler
        self.scaler.fit(X_eng)
        # NOTA: Scikit-learn guarda feature_names_in_ automáticamente al hacer fit
        
        X_scaled = self.scaler.transform(X_eng)
        X_scaled = pd.DataFrame(X_scaled, columns=X_eng.columns, index=X_eng.index)
        
        # 3. Fit Selector (if y is provided)
        if y is not None:
            k = min(30, X_scaled.shape[1])
            self.selector = SelectKBest(score_func=f_classif, k=k)
            self.selector.fit(X_scaled, y)
            self.selected_features_names = X_scaled.columns[self.selector.get_support()].tolist()
        else:
            self.selected_features_names = X_scaled.columns.tolist()
            
        return self

    def transform(self, X):
        # 1. Generate features
        X_eng = self._engineer_features(X.copy())
        
        # --- CORRECCIÓN DE ORDEN DE COLUMNAS ---
        # Verificamos qué columnas espera el scaler y reordenamos X_eng para que coincida
        if hasattr(self.scaler, 'feature_names_in_'):
            # Esto asegura que el orden sea idéntico al del entrenamiento
            # También elimina columnas extra si las hubiera
            X_eng = X_eng[self.scaler.feature_names_in_]
        # ----------------------------------------
        
        # 2. Scale
        X_scaled = self.scaler.transform(X_eng)
        X_scaled = pd.DataFrame(X_scaled, columns=X_eng.columns, index=X_eng.index)
        
        # 3. Select
        if self.selected_features_names:
            return X_scaled[self.selected_features_names]
        return X_scaled

    def _engineer_features(self, X):
        # Helper to create features (stateless)
        
        # Interactions
        for i, feat1 in enumerate(self.important_features):
            for feat2 in self.important_features[i+1:]:
                if feat1 in X.columns and feat2 in X.columns:
                    X[f'{feat1}_x_{feat2}'] = X[feat1] * X[feat2]
                    X[f'{feat1}_div_{feat2}'] = X[feat1] / (X[feat2] + 1e-8)
        
        # Polynomials & Logs
        for feat in self.important_features[:3]:
            if feat in X.columns:
                X[f'{feat}_squared'] = X[feat] ** 2
                X[f'{feat}_log'] = np.log1p(np.abs(X[feat]))
                
        return X

def train_and_evaluate_model_optimized(X_train: pd.DataFrame, y_train: pd.Series, 
                                     X_test: pd.DataFrame, y_test: pd.Series,
                                     training_metadata: dict = None):
    """
    Entrena un modelo RandomForest con arquitectura avanzada y búsqueda aleatoria de hiperparámetros.
    Optimizado para maximizar F1-Score (Macro).
    """
    print("\n🚀 Paso 3: Iniciando entrenamiento avanzado (RandomizedSearch + SelectFromModel)...")
    
    # 1. DEFINICIÓN DE ESTRATEGIA DE BALANCEO
    # Probamos diferentes estrategias en el grid search si es posible, 
    # pero SMOTETomek es generalmente robusto. Lo mantenemos fijo para no explotar la complejidad,
    # pero permitiremos que el RF maneje pesos de clase también.
    sampling_technique = SMOTETomek(
        random_state=42,
        smote=SMOTE(k_neighbors=5, random_state=42),
        tomek=TomekLinks(n_jobs=-1)
    )

    # 2. DEFINICIÓN DEL PIPELINE AVANZADO
    # Usamos un RF ligero para selección de características inicial
    selector_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)

    pipeline = Pipeline([
        ('features', FeatureEngineer()),           
        ('selection', SelectFromModel(estimator=selector_model)), # Threshold se optimiza en el grid
        ('balance', sampling_technique),           
        ('rf', RandomForestClassifier(             
            random_state=42, 
            n_jobs=-1,
            oob_score=False, # False para evitar overhead en grid search
            bootstrap=True
        ))
    ])

    # 3. GRID DE HIPERPARÁMETROS AMPLIADO (RandomizedSearch)
    param_distributions = {
        # Feature Selection
        'selection__threshold': ['median', 'mean', '1.25*mean', '0.75*median'],
        
        # Random Forest Structure
        'rf__n_estimators': [200, 400, 600, 800, 1000],
        'rf__max_depth': [10, 20, 30, 40, None],
        'rf__min_samples_split': [2, 5, 10, 15],
        'rf__min_samples_leaf': [1, 2, 4, 8],
        'rf__max_features': ['sqrt', 'log2', None],
        
        # Regularization & Imbalance
        'rf__bootstrap': [True, False],
        'rf__class_weight': ['balanced', 'balanced_subsample', None],
        'rf__criterion': ['gini', 'entropy']
    }

    # 4. CONFIGURACIÓN DE VALIDACIÓN
    stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scorer = make_scorer(f1_score, average='macro')

    # Usamos RandomizedSearchCV para explorar eficientemente el espacio grande
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=50, # 50 combinaciones aleatorias
        cv=stratified_cv,
        scoring=scorer,
        n_jobs=-1,
        verbose=2,
        return_train_score=True,
        random_state=42
    )
    
    print(f"   ⚡ Ejecutando RandomizedSearchCV (n_iter=50)...")
    random_search.fit(X_train, y_train)
    
    # --- RECUPERACIÓN DE OBJETOS PARA EL RETURN ---
    
    best_pipeline = random_search.best_estimator_
    search_results = pd.DataFrame(random_search.cv_results_)
    
    # EXTRAER SCALER Y SELECTOR
    scaler = best_pipeline.named_steps['features'].scaler
    selector = best_pipeline.named_steps['selection']
    
    print(f"\n✅ MEJOR CONFIGURACIÓN ENCONTRADA: {random_search.best_params_}")
    print(f"   Mejor F1-Macro (CV): {random_search.best_score_:.4f}")

    # 5. EVALUACIÓN DETALLADA
    print(f"\n🔍 Evaluando en set de prueba...")
    y_pred = best_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{'='*60}")
    print(f"ACCURACY FINAL: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"{'='*60}")
    
    class_labels = ['Otras Tierras (0)', 'Bosque (1)', 'Deforestado (2)']
    print(classification_report(y_test, y_pred, target_names=class_labels, digits=4))

    # 6. EXTRACCIÓN DE IMPORTANCIA DE CARACTERÍSTICAS
    print(f"\n🔝 TOP CARACTERÍSITCAS SELECCIONADAS:")
    feature_importance = pd.DataFrame()
    
    try:
        # Paso A: Obtener nombres del FeatureEngineer
        feat_eng_step = best_pipeline.named_steps['features']
        if hasattr(feat_eng_step, 'feature_names_in_') and feat_eng_step.feature_names_in_ is not None:
             base_features = np.array(feat_eng_step.feature_names_in_)
        else:
             # Fallback: regenerar nombres usando una muestra pequeña
             dummy = feat_eng_step.transform(X_train.iloc[:5])
             base_features = np.array(dummy.columns)

        # Paso B: Filtrar nombres usando el Selector
        selected_mask = selector.get_support()
        final_features = base_features[selected_mask]
        
        # Paso C: Obtener importancias del RF final
        rf_step = best_pipeline.named_steps['rf']
        importances = rf_step.feature_importances_
        
        feature_importance = pd.DataFrame({
            'feature': final_features,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print(feature_importance.head(15))
        
    except Exception as e:
        print(f"⚠️ No se pudo extraer importancia detallada: {e}")
        feature_importance = None

    # Guardar resultados de búsqueda
    search_results.to_csv('random_search_optimized.csv', index=False)
    
    # RETURN EXACTO PARA COMPATIBILIDAD
    return best_pipeline, search_results, feature_importance, scaler, selector

def save_optimized_artifacts(classifier: Pipeline, 
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
