"""
Pipeline de detección de fraude con tarjetas de crédito.
Implementa Grid Search con validación cruzada 5-fold y ajuste de umbral.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import (confusion_matrix, precision_score, recall_score, 
                             f1_score, accuracy_score, make_scorer)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("PIPELINE DE DETECCIÓN DE FRAUDE CON TARJETAS DE CRÉDITO")
print("=" * 80)

DATA_PATH = Path("datos/creditcard.csv")
OUT_DIR = Path("datos/proyecto2_resultados")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Carga y muestreo
print("\n[1/7] Cargando datos...")
df = pd.read_csv(DATA_PATH)
y_full = df["Class"].astype(int).values
X_full = df.drop(columns=["Class"]).values
cols = df.drop(columns=["Class"]).columns.tolist()

print(f"Dataset completo: {X_full.shape[0]} transacciones, {X_full.shape[1]} variables")
print(f"Fraudes: {y_full.sum()} ({100*y_full.mean():.3f}%)")

# Muestreo estratificado
idx_pos = np.where(y_full == 1)[0]
idx_neg = np.where(y_full == 0)[0]
n_pos = idx_pos.shape[0]
target_N = 12000
n_neg = min(target_N - n_pos, idx_neg.shape[0])
rng = np.random.RandomState(RANDOM_STATE)
idx_neg_sample = rng.choice(idx_neg, size=n_neg, replace=False)
idx_sample = np.concatenate([idx_pos, idx_neg_sample])
rng.shuffle(idx_sample)

X = X_full[idx_sample]
y = y_full[idx_sample]

print(f"Muestra de trabajo: {X.shape[0]} transacciones ({y.sum()} fraudes)")

# Partición y escalado
print("\n[2/7] Partición train/test (70/30) y escalado...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
)

print(f"Train: {X_train.shape[0]} muestras ({y_train.sum()} fraudes)")
print(f"Test: {X_test.shape[0]} muestras ({y_test.sum()} fraudes)")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Selección de características
print("\n[3/7] Selección de características (Mutual Information, K=10)...")
K = 10
selector = SelectKBest(mutual_info_classif, k=K)
selector.fit(X_train_scaled, y_train)
X_train_sel = selector.transform(X_train_scaled)
X_test_sel = selector.transform(X_test_scaled)
selected_cols = [cols[i] for i in selector.get_support(indices=True)]

print(f"Características seleccionadas: {', '.join(selected_cols)}")

def best_threshold(scores, y_true, fn_cost=10.0, fp_cost=1.0):
    """Encuentra el umbral que minimiza: fn_cost * FN + fp_cost * FP"""
    thresholds = np.quantile(scores, np.linspace(0, 1, 400))
    best_t, best_cost = None, float("inf")
    for t in thresholds:
        y_hat = (scores >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_hat).ravel()
        cost = fn_cost * fn + fp_cost * fp
        if cost < best_cost:
            best_cost = cost
            best_t = t
    return float(best_t)

def metrics_dict(y_true, y_pred):
    """Calcula métricas de clasificación."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "TP": int(tp),
        "FN": int(fn),
        "FP": int(fp),
        "TN": int(tn),
        "Recall_Sensibilidad": recall_score(y_true, y_pred, zero_division=0),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "F1_Score": f1_score(y_true, y_pred, zero_division=0),
        "Especificidad": (tn/(tn+fp)) if (tn+fp) > 0 else 0.0,
        "Exactitud_Accuracy": accuracy_score(y_true, y_pred)
    }

recall_scorer = make_scorer(recall_score, zero_division=0)

print("\n[4/7] Definiendo espacios de búsqueda...")

param_grids = {
    'k-NN': {
        'n_neighbors': [5, 10, 15, 20],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]
    },
    'Árbol de Decisión': {
        'max_depth': [5, 8, 10, 15],
        'min_samples_leaf': [1, 2, 5],
        'class_weight': ['balanced', None]
    },
    'SVM': {
        'C': [0.1, 0.5, 1.0, 2.0],
        'class_weight': ['balanced', None]
    },
    'Random Forest': {
        'n_estimators': [50, 80, 100],
        'max_depth': [8, 10, 15],
        'min_samples_leaf': [1, 2],
        'class_weight': ['balanced', 'balanced_subsample']
    }
}

print("\n[5/7] Ejecutando Grid Search con validación cruzada 5-fold...")

models_config = {
    'k-NN': KNeighborsClassifier(),
    'Árbol de Decisión': DecisionTreeClassifier(random_state=RANDOM_STATE),
    'SVM': LinearSVC(dual=False, random_state=RANDOM_STATE, max_iter=2000),
    'Random Forest': RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
}

results_grid = []
best_models = {}

for name, model in models_config.items():
    print(f"Entrenando {name}...")
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grids[name],
        cv=5,
        scoring=recall_scorer,
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train_sel, y_train)
    
    best_models[name] = grid_search.best_estimator_
    
    results_grid.append({
        'Modelo': name,
        'Mejores_Hiperparametros': str(grid_search.best_params_),
        'Mejor_Recall_CV': round(grid_search.best_score_, 4),
        'Num_Combinaciones_Probadas': len(grid_search.cv_results_['params'])
    })
    
    print(f"  Mejor Recall (CV): {grid_search.best_score_:.4f}")
    print(f"  Mejores hiperparámetros: {grid_search.best_params_}")
    print(f"  Combinaciones probadas: {len(grid_search.cv_results_['params'])}\n")

grid_df = pd.DataFrame(results_grid)
grid_df.to_csv(OUT_DIR / "grid_search_resultados.csv", index=False)

print("\n[6/7] Ajustando umbrales de decisión...")

X_train_val, X_val, y_train_val, y_val = train_test_split(
    X_train_sel, y_train, test_size=0.20, stratify=y_train, random_state=RANDOM_STATE
)

results_test = []

for name, model in best_models.items():
    model.fit(X_train_sel, y_train)
    
    if hasattr(model, 'predict_proba'):
        val_scores = model.predict_proba(X_val)[:, 1]
        test_scores = model.predict_proba(X_test_sel)[:, 1]
    else:
        val_scores = model.decision_function(X_val)
        test_scores = model.decision_function(X_test_sel)
    
    threshold = best_threshold(val_scores, y_val, fn_cost=10.0, fp_cost=1.0)
    y_pred = (test_scores >= threshold).astype(int)
    
    metrics = metrics_dict(y_test, y_pred)
    metrics['Modelo'] = name
    metrics['Umbral_Ajustado'] = round(threshold, 6)
    
    results_test.append(metrics)
    
    print(f"{name}:")
    print(f"  Umbral: {threshold:.6f}")
    print(f"  Recall: {metrics['Recall_Sensibilidad']:.4f} | F1: {metrics['F1_Score']:.4f}")
    print(f"  FN: {metrics['FN']} | FP: {metrics['FP']}\n")

print("\n[7/7] Guardando resultados...")

test_df = pd.DataFrame(results_test)

cols_order = ['Modelo', 'Umbral_Ajustado', 'TP', 'FN', 'FP', 'TN', 
              'Recall_Sensibilidad', 'Precision', 'F1_Score', 
              'Especificidad', 'Exactitud_Accuracy']
test_df = test_df[cols_order]

test_df.to_csv(OUT_DIR / "test_resultados_completo.csv", index=False)
test_df.to_latex(OUT_DIR / "test_resultados_completo.tex", index=False, float_format="%.4f", escape=False)
grid_df.to_latex(OUT_DIR / "grid_search_resultados.tex", index=False, escape=False)

with open(OUT_DIR / "features_seleccionadas.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(selected_cols))

print("\n" + "=" * 80)
print("RESULTADOS FINALES")
print("=" * 80)

print("\n--- Grid Search (Validación Cruzada 5-fold) ---")
print(grid_df.to_string(index=False))

print("\n--- Evaluación en Test ---")
print(test_df.to_string(index=False))

print("\n--- Mejor Modelo por Métrica ---")
best_recall = test_df.loc[test_df['Recall_Sensibilidad'].idxmax()]
best_f1 = test_df.loc[test_df['F1_Score'].idxmax()]
min_fn = test_df.loc[test_df['FN'].idxmin()]

print(f"Mejor Recall: {best_recall['Modelo']} ({best_recall['Recall_Sensibilidad']:.4f})")
print(f"Mejor F1: {best_f1['Modelo']} ({best_f1['F1_Score']:.4f})")
print(f"Menos FN: {min_fn['Modelo']} ({min_fn['FN']} fraudes no detectados)")

print("\n" + "=" * 80)
print(f"Archivos guardados en: {OUT_DIR.absolute()}")
print("=" * 80)
