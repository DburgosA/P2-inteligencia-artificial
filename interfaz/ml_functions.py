"""
Funciones de Machine Learning para el Pipeline de Deteccion de Fraude
Contiene funciones reutilizables para entrenamiento y evaluacion
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier


def best_threshold(scores, y_true, fn_cost=10.0, fp_cost=1.0):
    """
    Encuentra el umbral optimo que minimiza el costo total:
    Costo = fn_cost * FN + fp_cost * FP
    
    Parametros:
    -----------
    scores : array-like
        Scores de prediccion del modelo
    y_true : array-like
        Etiquetas verdaderas
    fn_cost : float
        Costo de falsos negativos (fraudes no detectados)
    fp_cost : float
        Costo de falsos positivos (alertas falsas)
    
    Retorna:
    --------
    float : Umbral optimo
    """
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
    """
    Calcula metricas relevantes para deteccion de fraude.
    
    Parametros:
    -----------
    y_true : array-like
        Etiquetas verdaderas
    y_pred : array-like
        Predicciones del modelo
    
    Retorna:
    --------
    dict : Diccionario con todas las metricas
    """
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


def get_param_grids():
    """
    Retorna los espacios de busqueda para Grid Search.
    
    Retorna:
    --------
    dict : Diccionario con param_grids para cada modelo
    """
    return {
        'k-NN': {
            'n_neighbors': [5, 10, 15, 20],
            'weights': ['uniform', 'distance'],
            'p': [1, 2]
        },
        'Arbol de Decision': {
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


def get_models_config(random_state=42):
    """
    Retorna la configuracion de modelos a entrenar.
    
    Parametros:
    -----------
    random_state : int
        Semilla para reproducibilidad
    
    Retorna:
    --------
    dict : Diccionario con instancias de modelos
    """
    return {
        'k-NN': KNeighborsClassifier(),
        'Arbol de Decision': DecisionTreeClassifier(random_state=random_state),
        'SVM': LinearSVC(dual=False, random_state=random_state, max_iter=2000),
        'Random Forest': RandomForestClassifier(random_state=random_state, n_jobs=-1)
    }


def train_with_gridsearch(X_train, y_train, model_name, model, param_grid, cv=5, random_state=42):
    """
    Entrena un modelo con Grid Search y validacion cruzada.
    
    Parametros:
    -----------
    X_train : array-like
        Datos de entrenamiento
    y_train : array-like
        Etiquetas de entrenamiento
    model_name : str
        Nombre del modelo
    model : estimator
        Instancia del modelo de sklearn
    param_grid : dict
        Espacio de busqueda de hiperparametros
    cv : int
        Numero de folds para validacion cruzada
    random_state : int
        Semilla para reproducibilidad
    
    Retorna:
    --------
    tuple : (mejor_modelo, mejores_params, mejor_score, num_combinaciones)
    """
    recall_scorer = make_scorer(recall_score, zero_division=0)
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring=recall_scorer,
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    return (
        grid_search.best_estimator_,
        grid_search.best_params_,
        grid_search.best_score_,
        len(grid_search.cv_results_['params'])
    )


def evaluate_model(model, X_val, y_val, X_test, y_test, fn_cost=10.0, fp_cost=1.0):
    """
    Evalua un modelo en el conjunto de test con ajuste de umbral.
    
    Parametros:
    -----------
    model : estimator
        Modelo entrenado
    X_val : array-like
        Datos de validacion para ajustar umbral
    y_val : array-like
        Etiquetas de validacion
    X_test : array-like
        Datos de test
    y_test : array-like
        Etiquetas de test
    fn_cost : float
        Costo de falsos negativos
    fp_cost : float
        Costo de falsos positivos
    
    Retorna:
    --------
    tuple : (metricas_dict, umbral_optimo)
    """
    # Obtener scores de validacion
    if hasattr(model, 'predict_proba'):
        val_scores = model.predict_proba(X_val)[:, 1]
        test_scores = model.predict_proba(X_test)[:, 1]
    else:  # LinearSVC usa decision_function
        val_scores = model.decision_function(X_val)
        test_scores = model.decision_function(X_test)
    
    # Encontrar umbral optimo
    threshold = best_threshold(val_scores, y_val, fn_cost=fn_cost, fp_cost=fp_cost)
    
    # Prediccion en test con umbral ajustado
    y_pred = (test_scores >= threshold).astype(int)
    
    # Calcular metricas
    metrics = metrics_dict(y_test, y_pred)
    metrics['Umbral_Ajustado'] = round(threshold, 6)
    
    return metrics, threshold
