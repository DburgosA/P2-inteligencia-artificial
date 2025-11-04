import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

# --- Configuración y paths ---
DATA_PATH = Path("datos/creditcard.csv")
OUT_DIR = Path("datos/proyecto2_resultados")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# --- Cargar datos ---
df = pd.read_csv(DATA_PATH)
y_full = df["Class"].astype(int).values
X_full = df.drop(columns=["Class"]).values
cols = df.drop(columns=["Class"]).columns.tolist()

# --- Muestreo estratificado rápido (incluye todos los fraudes) ---
idx_pos = np.where(y_full == 1)[0]
idx_neg = np.where(y_full == 0)[0]
n_pos = idx_pos.shape[0]  # ~492
target_N = 12000          # tamaño total aprox. para ejecución ágil
n_neg = min(target_N - n_pos, idx_neg.shape[0])
rng = np.random.RandomState(RANDOM_STATE)
idx_neg_sample = rng.choice(idx_neg, size=n_neg, replace=False)
idx_sample = np.concatenate([idx_pos, idx_neg_sample])
rng.shuffle(idx_sample)

X = X_full[idx_sample]
y = y_full[idx_sample]

# --- Partición 70/30 y validación (20% de train) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.20, stratify=y_train, random_state=RANDOM_STATE
)

# --- Escalado ---
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

# --- Selección de características (mutual information) ---
K = 10
selector = SelectKBest(mutual_info_classif, k=K)
selector.fit(X_tr_s, y_tr)
X_tr_sel = selector.transform(X_tr_s)
X_val_sel = selector.transform(X_val_s)
X_test_sel = selector.transform(X_test_s)
selected_cols = [cols[i] for i in selector.get_support(indices=True)]

def best_threshold(scores: np.ndarray, y_true: np.ndarray, fn_cost: float = 10.0, fp_cost: float = 1.0) -> float:
    """Selecciona el umbral que minimiza costo = fn_cost*FN + fp_cost*FP en validación."""
    thr = np.quantile(scores, np.linspace(0, 1, 400))
    best_t, best_c = None, float("inf")
    for t in thr:
        y_hat = (scores >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_hat).ravel()
        c = fn_cost * fn + fp_cost * fp
        if c < best_c:
            best_c = c
            best_t = t
    return float(best_t)

def metrics_dict(y_true: np.ndarray, y_hat: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat).ravel()
    return {
        "TP": int(tp), "FN": int(fn), "FP": int(fp), "TN": int(tn),
        "Recall": recall_score(y_true, y_hat, zero_division=0),
        "Precisión": precision_score(y_true, y_hat, zero_division=0),
        "F1": f1_score(y_true, y_hat, zero_division=0),
        "Especificidad": (tn/(tn+fp)) if (tn+fp) > 0 else 0.0,
        "Exactitud": accuracy_score(y_true, y_hat)
    }

rows_tuning, rows_test = [], []

# --- k-NN ---
knn = KNeighborsClassifier(n_neighbors=15, weights="distance", p=2)
knn.fit(X_tr_sel, y_tr)
thr_knn = best_threshold(knn.predict_proba(X_val_sel)[:, 1], y_val)
y_hat_knn = (knn.predict_proba(X_test_sel)[:, 1] >= thr_knn).astype(int)
rows_tuning.append({"Modelo": "k-NN",
                    "Hiperparámetros": str({"n_neighbors": 15, "weights": "distance", "p": 2}),
                    "Umbral (10:1)": round(thr_knn, 6)})
rows_test.append({"Modelo": "k-NN", **metrics_dict(y_test, y_hat_knn)})

# --- Árbol de Decisión ---
tree = DecisionTreeClassifier(max_depth=8, min_samples_leaf=2, class_weight="balanced", random_state=RANDOM_STATE)
tree.fit(X_tr_sel, y_tr)
thr_tree = best_threshold(tree.predict_proba(X_val_sel)[:, 1], y_val)
y_hat_tree = (tree.predict_proba(X_test_sel)[:, 1] >= thr_tree).astype(int)
rows_tuning.append({"Modelo": "Árbol de Decisión",
                    "Hiperparámetros": str({"max_depth": 8, "min_samples_leaf": 2, "class_weight": "balanced"}),
                    "Umbral (10:1)": round(thr_tree, 6)})
rows_test.append({"Modelo": "Árbol de Decisión", **metrics_dict(y_test, y_hat_tree)})

# --- SVM (LinearSVC) ---
svm = LinearSVC(C=1.0, class_weight="balanced", dual=False, random_state=RANDOM_STATE)
svm.fit(X_tr_sel, y_tr)
thr_svm = best_threshold(svm.decision_function(X_val_sel), y_val)
y_hat_svm = (svm.decision_function(X_test_sel) >= thr_svm).astype(int)
rows_tuning.append({"Modelo": "SVM (LinearSVC)",
                    "Hiperparámetros": str({"C": 1.0, "class_weight": "balanced"}),
                    "Umbral (10:1)": round(thr_svm, 6)})
rows_test.append({"Modelo": "SVM (LinearSVC)", **metrics_dict(y_test, y_hat_svm)})

# --- Random Forest ---
rf = RandomForestClassifier(n_estimators=80, max_depth=10, min_samples_leaf=1,
                            class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(X_tr_sel, y_tr)
thr_rf = best_threshold(rf.predict_proba(X_val_sel)[:, 1], y_val)
y_hat_rf = (rf.predict_proba(X_test_sel)[:, 1] >= thr_rf).astype(int)
rows_tuning.append({"Modelo": "Random Forest",
                    "Hiperparámetros": str({"n_estimators": 80, "max_depth": 10, "min_samples_leaf": 1,
                                            "class_weight": "balanced_subsample"}),
                    "Umbral (10:1)": round(thr_rf, 6)})
rows_test.append({"Modelo": "Random Forest", **metrics_dict(y_test, y_hat_rf)})

# --- Guardar resultados ---
cv_df = pd.DataFrame(rows_tuning)
test_df = pd.DataFrame(rows_test)

(OUT_DIR / "cv_resumen.csv").write_text(cv_df.to_csv(index=False), encoding="utf-8")
(OUT_DIR / "test_resumen.csv").write_text(test_df.to_csv(index=False), encoding="utf-8")
(OUT_DIR / "cv_resumen.tex").write_text(cv_df.to_latex(index=False, escape=False), encoding="utf-8")
(OUT_DIR / "test_resumen.tex").write_text(test_df.to_latex(index=False, float_format="%.4f", escape=False), encoding="utf-8")
(OUT_DIR / "features_seleccionadas.txt").write_text("\n".join(selected_cols), encoding="utf-8")

# --- Imprimir resumen en consola ---
print("Características seleccionadas (K=10):", selected_cols)
print("\nResumen tuning:")
print(cv_df.to_string(index=False))
print("\nResultados en test:")
print(test_df.to_string(index=False))
