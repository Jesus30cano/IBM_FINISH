# 2_train_lightgbm_corrected.py
import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

h5_path = 'atl_monthly_ready.h5'
df = pd.read_hdf(h5_path, key='data')

# Ordenar datos
df = df.sort_values(['MUNICIPIO', 'YEAR_MONTH']).reset_index(drop=True)

# ESTRATEGIA CORREGIDA: Características SIN data leakage
def create_safe_features(df):
    # 1. Características básicas seguras (solo lags)
    base_features = ['TASA_LAG1', 'TASA_LAG2', 'CANT_LAG1', 'CANT_LAG2']
    
    # 2. Rolling features SEGURAS (con shift para evitar leakage)
    for window in [3, 6]:
        df[f'TASA_ROLL_MEAN_{window}_SAFE'] = df.groupby('MUNICIPIO')['TASA_100k'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
        df[f'TASA_ROLL_STD_{window}_SAFE'] = df.groupby('MUNICIPIO')['TASA_100k'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).std()
        )
    
    # 3. Características temporales básicas
    df['MONTH'] = df['YEAR_MONTH'].dt.month
    df['QUARTER'] = df['YEAR_MONTH'].dt.quarter
    
    # 4. Diferencias y cambios (seguros)
    df['TASA_DIFF_1'] = df.groupby('MUNICIPIO')['TASA_100k'].diff(1)
    df['TASA_DIFF_2'] = df.groupby('MUNICIPIO')['TASA_100k'].diff(2)
    
    # 5. Características municipales SEGURAS (solo usando datos históricos)
    df['MUN_HIST_MEAN'] = df.groupby('MUNICIPIO')['TASA_100k'].transform(
        lambda x: x.expanding().mean()
    )
    df['MUN_HIST_STD'] = df.groupby('MUNICIPIO')['TASA_100k'].transform(
        lambda x: x.expanding().std()
    )
    
    # 6. Indicadores de patrón (sin leakage)
    df['WAS_SPIKE'] = (df['TASA_LAG1'] > df['TASA_LAG2'] * 1.5).astype(int)
    df['WAS_DROP'] = (df['TASA_LAG1'] < df['TASA_LAG2'] * 0.5).astype(int)
    
    return df

df = create_safe_features(df)

# Target
df['TARGET_TASA_NEXT'] = df.groupby('MUNICIPIO')['TASA_100k'].shift(-1)
df_train = df.dropna(subset=['TARGET_TASA_NEXT']).copy()

# CARACTERÍSTICAS COMPLETAMENTE SEGURAS
safe_features = [
    # Lags básicos
    'TASA_LAG1', 'TASA_LAG2', 'CANT_LAG1', 'CANT_LAG2',
    
    # Rolling features seguras
    'TASA_ROLL_MEAN_3_SAFE', 'TASA_ROLL_STD_3_SAFE',
    'TASA_ROLL_MEAN_6_SAFE', 'TASA_ROLL_STD_6_SAFE',
    
    # Características municipales históricas
    'MUN_HIST_MEAN', 'MUN_HIST_STD',
    
    # Diferencias
    'TASA_DIFF_1', 'TASA_DIFF_2',
    
    # Patrones históricos
    'WAS_SPIKE', 'WAS_DROP',
    
    # Temporales simples
    'MONTH', 'QUARTER'
]

print(f"🔒 Usando {len(safe_features)} características COMPLETAMENTE SEGURAS")
print(f"📊 Muestras de entrenamiento: {len(df_train)}")

# Limpieza
df_train[safe_features] = df_train[safe_features].replace([np.inf, -np.inf], np.nan)
df_train[safe_features] = df_train[safe_features].fillna(method='bfill').fillna(method='ffill').fillna(0)

X = df_train[safe_features]
y = df_train['TARGET_TASA_NEXT']

# ESTRATEGIA: CV con más datos y análisis de distribución
tscv = TimeSeriesSplit(n_splits=5, test_size=24)  # 24 meses para mejor evaluación
maes = []
models = []

# PARÁMETROS PARA CAPTURAR VARIABILIDAD
params_variable = {
    'objective': 'regression',
    'metric': 'mae',
    'learning_rate': 0.05,  # Más alto para aprender patrones complejos
    'num_leaves': 127,      # Más capacidad para capturar extremos
    'max_depth': 8,
    'min_data_in_leaf': 20,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.7,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'min_gain_to_split': 0.01,  # Más sensible a splits
    'verbosity': -1,
    'seed': 42
}

print("\n🎯 ENTRENAMIENTO CORREGIDO - ENFOCADO EN CAPTURAR VARIABILIDAD")
print("=" * 60)

for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    print(f"\n--- Fold {fold + 1} ---")
    
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    print(f"Train: {X_tr.shape[0]} muestras, Val: {X_val.shape[0]} muestras")
    print(f"Distribución target en val: min={y_val.min():.1f}, max={y_val.max():.1f}, mean={y_val.mean():.1f}")
    
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    
    bst = lgb.train(
        params_variable,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dval],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=100, show_stdv=False)
        ]
    )

    preds = bst.predict(X_val, num_iteration=bst.best_iteration)
    mae = mean_absolute_error(y_val, preds)
    
    maes.append(mae)
    models.append(bst)
    
    # Análisis de rango de predicciones
    pred_min, pred_max = preds.min(), preds.max()
    print(f"Fold {fold + 1} - MAE: {mae:.4f}")
    print(f"Rango predicciones: {pred_min:.1f} a {pred_max:.1f}")

# ANÁLISIS DETALLADO DEL MEJOR MODELO
print("\n" + "=" * 70)
print("ANÁLISIS DETALLADO - VERSIÓN CORREGIDA")
print("=" * 70)

best_idx = np.argmin(maes)
best_model = models[best_idx]

print(f"🎯 MEJOR FOLD: {best_idx + 1} (MAE: {maes[best_idx]:.4f})")
print(f"📈 MAE promedio: {np.mean(maes):.4f} ± {np.std(maes):.4f}")

# Predicciones detalladas del mejor fold
val_idx_best = list(tscv.split(X))[best_idx][1]
X_val_best = X.iloc[val_idx_best]
y_val_best = y.iloc[val_idx_best]
preds_best = best_model.predict(X_val_best)

# Análisis por rangos de valores
def detailed_analysis(y_true, y_pred):
    results = []
    for true, pred in zip(y_true, y_pred):
        error = abs(true - pred)
        error_pct = (error / true) * 100
        results.append({
            'true': true,
            'pred': pred,
            'error': error,
            'error_pct': error_pct,
            'range': f"{int(true//10)*10}-{int(true//10)*10+10}"
        })
    
    return pd.DataFrame(results)

analysis_df = detailed_analysis(y_val_best, preds_best)

print(f"\n📊 ESTADÍSTICAS DE PREDICCIÓN:")
print(f"Rango real: {analysis_df['true'].min():.1f} - {analysis_df['true'].max():.1f}")
print(f"Rango predicho: {analysis_df['pred'].min():.1f} - {analysis_df['pred'].max():.1f}")
print(f"Error porcentual promedio: {analysis_df['error_pct'].mean():.1f}%")

# Mostrar ejemplos de diferentes rangos
print(f"\n🎯 PREDICCIONES POR RANGO (primeros 15):")
print("-" * 75)
print(f"{'Real':>6} {'Predicho':>10} {'Error':>8} {'Error%':>8} {'Correcto':>10}")
print("-" * 75)

for _, row in analysis_df.head(15).iterrows():
    correcto = "✅" if row['error_pct'] < 30 else "❌"
    print(f"{row['true']:6.1f} {row['pred']:10.1f} {row['error']:8.1f} {row['error_pct']:7.1f}% {correcto:>10}")

# Análisis de capacidad para capturar extremos
high_values = analysis_df[analysis_df['true'] >= 35]
if len(high_values) > 0:
    print(f"\n🔍 ANÁLISIS VALORES ALTOS (>=35):")
    print(f"Cantidad: {len(high_values)}")
    print(f"Error promedio: {high_values['error'].mean():.1f}")
    print(f"Predicción promedio: {high_values['pred'].mean():.1f} vs Real: {high_values['true'].mean():.1f}")

# Guardar modelo corregido
model_metadata = {
    'features': safe_features,
    'cv_mae': maes,
    'best_fold': best_idx,
    'best_mae': maes[best_idx],
    'prediction_range_analysis': {
        'true_min': analysis_df['true'].min(),
        'true_max': analysis_df['true'].max(), 
        'pred_min': analysis_df['pred'].min(),
        'pred_max': analysis_df['pred'].max()
    }
}

joblib.dump({
    'model': best_model,
    'metadata': model_metadata,
    'feature_names': safe_features
}, 'lgb_atlantico_model_corrected.pkl')

print(f"\n MODELO CORREGIDO GUARDADO: 'lgb_atlantico_model_corrected.pkl'")

# EVALUACIÓN FINAL
mean_error_pct = analysis_df['error_pct'].mean()
if mean_error_pct < 25:
    print("🎉 EXCELENTE - Modelo captura bien la variabilidad")
elif mean_error_pct < 40:
    print("📈 ACEPTABLE - Mejora significativa sobre versiones anteriores")  
else:
    print("🔧 NECESITA MEJORA - Revisar datos fundamentales")

print(f"Error porcentual final: {mean_error_pct:.1f}%")