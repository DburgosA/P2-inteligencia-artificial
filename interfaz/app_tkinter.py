"""
Interfaz Grafica con Tkinter para Pipeline de Deteccion de Fraude
Version optimizada - mas rapida que Streamlit
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import numpy as np
import threading
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif

from ml_functions import (
    get_param_grids, 
    get_models_config, 
    train_with_gridsearch, 
    evaluate_model
)


class FraudDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pipeline de Deteccion de Fraude - Tarjetas de Credito")
        self.root.geometry("1200x800")
        
        # Variables de datos
        self.df = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.selected_features = []
        self.suggested_features = []
        self.best_models = {}
        self.results_grid = []
        self.results_test = []
        
        # Variables de configuracion
        self.train_size = tk.IntVar(value=70)
        self.k_features = tk.IntVar(value=10)
        self.random_state = tk.IntVar(value=42)
        
        # Crear interfaz
        self.create_widgets()
        
    def create_widgets(self):
        # Frame principal con notebook (pestañas)
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Pestañas
        self.tab1 = ttk.Frame(notebook)
        self.tab2 = ttk.Frame(notebook)
        self.tab3 = ttk.Frame(notebook)
        
        notebook.add(self.tab1, text="Paso 1: Carga y Configuracion")
        notebook.add(self.tab2, text="Paso 2: Grid Search")
        notebook.add(self.tab3, text="Paso 3: Evaluacion")
        
        self.create_tab1()
        self.create_tab2()
        self.create_tab3()
        
    def create_tab1(self):
        # PASO 1: Carga y configuracion
        main_frame = ttk.Frame(self.tab1, padding="10")
        main_frame.pack(fill='both', expand=True)
        
        # Titulo
        ttk.Label(main_frame, text="PASO 1: CARGA Y CONFIGURACION DE DATOS", 
                 font=('Arial', 14, 'bold')).pack(pady=10)
        
        # Seccion: Cargar archivo
        file_frame = ttk.LabelFrame(main_frame, text="Cargar Archivo CSV", padding="10")
        file_frame.pack(fill='x', pady=5)
        
        ttk.Button(file_frame, text="Seleccionar Archivo CSV", 
                  command=self.load_file).pack(side='left', padx=5)
        
        self.file_label = ttk.Label(file_frame, text="No se ha cargado ningun archivo")
        self.file_label.pack(side='left', padx=5)
        
        # Seccion: Configuracion
        config_frame = ttk.LabelFrame(main_frame, text="Configuracion", padding="10")
        config_frame.pack(fill='x', pady=5)
        
        # Train/Test split
        ttk.Label(config_frame, text="Porcentaje Train:").grid(row=0, column=0, sticky='w', pady=5)
        train_scale = ttk.Scale(config_frame, from_=50, to=90, variable=self.train_size, 
                               orient='horizontal', length=200, command=self.update_split_label)
        train_scale.grid(row=0, column=1, padx=5)
        self.split_label = ttk.Label(config_frame, text="Train: 70% | Test: 30%")
        self.split_label.grid(row=0, column=2, padx=5)
        
        # K features
        ttk.Label(config_frame, text="Numero de caracteristicas (K):").grid(row=1, column=0, sticky='w', pady=5)
        ttk.Spinbox(config_frame, from_=5, to=28, textvariable=self.k_features, width=10).grid(row=1, column=1, sticky='w', padx=5)
        
        # Random state
        ttk.Label(config_frame, text="Semilla aleatoria:").grid(row=2, column=0, sticky='w', pady=5)
        ttk.Entry(config_frame, textvariable=self.random_state, width=10).grid(row=2, column=1, sticky='w', padx=5)
        
        # Seccion: Sugerencias
        suggest_frame = ttk.LabelFrame(main_frame, text="Sugerencias de Caracteristicas", padding="10")
        suggest_frame.pack(fill='both', expand=True, pady=5)
        
        ttk.Button(suggest_frame, text="Calcular Sugerencias (Mutual Information)", 
                  command=self.calculate_suggestions).pack(pady=5)
        
        self.suggestions_text = scrolledtext.ScrolledText(suggest_frame, height=5, wrap=tk.WORD)
        self.suggestions_text.pack(fill='both', expand=True, pady=5)
        
        # Seccion: Seleccion manual
        select_frame = ttk.LabelFrame(main_frame, text="Seleccion Manual de Caracteristicas", padding="10")
        select_frame.pack(fill='both', expand=True, pady=5)
        
        # Frame para checkboxes
        self.checks_frame = ttk.Frame(select_frame)
        self.checks_frame.pack(fill='both', expand=True)
        
        # Botones de accion
        action_frame = ttk.Frame(main_frame)
        action_frame.pack(fill='x', pady=10)
        
        ttk.Button(action_frame, text="Preprocesar Datos", 
                  command=self.preprocess_data).pack(side='left', padx=5)
        
        self.status_label = ttk.Label(action_frame, text="", foreground="blue")
        self.status_label.pack(side='left', padx=10)
        
    def create_tab2(self):
        # PASO 2: Grid Search
        main_frame = ttk.Frame(self.tab2, padding="10")
        main_frame.pack(fill='both', expand=True)
        
        # Titulo
        ttk.Label(main_frame, text="PASO 2: ENTRENAMIENTO CON GRID SEARCH", 
                 font=('Arial', 14, 'bold')).pack(pady=10)
        
        # Boton de entrenamiento
        ttk.Button(main_frame, text="Entrenar Modelos (Grid Search)", 
                  command=self.train_models).pack(pady=10)
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='determinate', length=600)
        self.progress.pack(pady=5)
        
        self.progress_label = ttk.Label(main_frame, text="")
        self.progress_label.pack(pady=5)
        
        # Resultados
        results_frame = ttk.LabelFrame(main_frame, text="Resultados del Grid Search", padding="10")
        results_frame.pack(fill='both', expand=True, pady=5)
        
        self.results_grid_text = scrolledtext.ScrolledText(results_frame, height=20, wrap=tk.WORD)
        self.results_grid_text.pack(fill='both', expand=True)
        
    def create_tab3(self):
        # PASO 3: Evaluacion
        main_frame = ttk.Frame(self.tab3, padding="10")
        main_frame.pack(fill='both', expand=True)
        
        # Titulo
        ttk.Label(main_frame, text="PASO 3: EVALUACION EN CONJUNTO DE TEST", 
                 font=('Arial', 14, 'bold')).pack(pady=10)
        
        # Boton de evaluacion
        ttk.Button(main_frame, text="Evaluar en Test", 
                  command=self.evaluate_models).pack(pady=10)
        
        # Resultados
        results_frame = ttk.LabelFrame(main_frame, text="Resultados de Evaluacion", padding="10")
        results_frame.pack(fill='both', expand=True, pady=5)
        
        self.results_test_text = scrolledtext.ScrolledText(results_frame, height=25, wrap=tk.WORD)
        self.results_test_text.pack(fill='both', expand=True)
        
        # Botones de exportacion
        export_frame = ttk.Frame(main_frame)
        export_frame.pack(fill='x', pady=10)
        
        ttk.Button(export_frame, text="Exportar Resultados Grid Search (CSV)", 
                  command=self.export_grid_results).pack(side='left', padx=5)
        
        ttk.Button(export_frame, text="Exportar Resultados Test (CSV)", 
                  command=self.export_test_results).pack(side='left', padx=5)
        
    def update_split_label(self, event=None):
        train = int(self.train_size.get())
        test = 100 - train
        self.split_label.config(text=f"Train: {train}% | Test: {test}%")
        
    def load_file(self):
        filename = filedialog.askopenfilename(
            title="Seleccionar archivo CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                self.df = pd.read_csv(filename)
                self.file_label.config(text=f"Archivo cargado: {Path(filename).name}")
                
                # Validar columnas
                if 'Class' not in self.df.columns:
                    messagebox.showerror("Error", "El archivo debe contener la columna 'Class'")
                    return
                
                v_cols = [col for col in self.df.columns if col.startswith('V')]
                if len(v_cols) < self.k_features.get():
                    messagebox.showerror("Error", 
                        f"El dataset solo tiene {len(v_cols)} variables PCA, pero se solicitaron {self.k_features.get()}")
                    return
                
                messagebox.showinfo("Exito", 
                    f"Archivo cargado: {len(self.df)} transacciones\n"
                    f"Fraudes: {self.df['Class'].sum()} ({100*self.df['Class'].mean():.3f}%)")
                
            except Exception as e:
                messagebox.showerror("Error", f"Error al cargar archivo: {str(e)}")
                
    def calculate_suggestions(self):
        if self.df is None:
            messagebox.showwarning("Advertencia", "Primero debes cargar un archivo CSV")
            return
        
        try:
            # Preparar datos
            y_temp = self.df['Class'].astype(int).values
            X_temp = self.df.drop(columns=['Class']).values
            cols_temp = self.df.drop(columns=['Class']).columns.tolist()
            
            # Escalar
            scaler_temp = StandardScaler()
            X_temp_scaled = scaler_temp.fit_transform(X_temp)
            
            # Calcular mutual information
            mi_scores = mutual_info_classif(X_temp_scaled, y_temp, 
                                           random_state=self.random_state.get())
            
            # Filtrar solo variables V
            v_indices = [i for i, col in enumerate(cols_temp) if col.startswith('V')]
            v_names = [cols_temp[i] for i in v_indices]
            v_scores = [mi_scores[i] for i in v_indices]
            
            # Ordenar por score
            sorted_features = sorted(zip(v_names, v_scores), key=lambda x: x[1], reverse=True)
            
            # Guardar sugerencias
            self.suggested_features = [f[0] for f in sorted_features[:self.k_features.get()]]
            
            # Mostrar resultados
            result_text = f"CARACTERISTICAS SUGERIDAS (Top {self.k_features.get()}):\n"
            result_text += ", ".join(self.suggested_features) + "\n\n"
            result_text += "RANKING COMPLETO:\n"
            result_text += f"{'Rank':<6}{'Caracteristica':<15}{'MI Score':<15}\n"
            result_text += "-" * 40 + "\n"
            
            for i, (feat, score) in enumerate(sorted_features, 1):
                result_text += f"{i:<6}{feat:<15}{score:<15.6f}\n"
            
            self.suggestions_text.delete('1.0', tk.END)
            self.suggestions_text.insert('1.0', result_text)
            
            # Crear checkboxes
            self.create_feature_checkboxes(v_names)
            
            messagebox.showinfo("Exito", "Sugerencias calculadas correctamente")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al calcular sugerencias: {str(e)}")
            
    def create_feature_checkboxes(self, all_features):
        # Limpiar frame anterior
        for widget in self.checks_frame.winfo_children():
            widget.destroy()
        
        # Crear variables y checkboxes
        self.feature_vars = {}
        
        # Titulo
        ttk.Label(self.checks_frame, text=f"Selecciona exactamente {self.k_features.get()} caracteristicas:", 
                 font=('Arial', 10, 'bold')).grid(row=0, column=0, columnspan=4, pady=5, sticky='w')
        
        # Crear checkboxes en grid (4 columnas)
        for i, feature in enumerate(all_features):
            var = tk.BooleanVar(value=(feature in self.suggested_features))
            self.feature_vars[feature] = var
            
            row = (i // 4) + 1
            col = i % 4
            
            cb = ttk.Checkbutton(self.checks_frame, text=feature, variable=var)
            cb.grid(row=row, column=col, padx=10, pady=2, sticky='w')
        
        # Contador de seleccionados
        count_frame = ttk.Frame(self.checks_frame)
        count_frame.grid(row=100, column=0, columnspan=4, pady=10)
        
        ttk.Button(count_frame, text="Verificar Seleccion", 
                  command=self.verify_selection).pack()
        
    def verify_selection(self):
        selected = [feat for feat, var in self.feature_vars.items() if var.get()]
        k = self.k_features.get()
        
        if len(selected) == k:
            self.selected_features = selected
            messagebox.showinfo("Exito", 
                f"Has seleccionado {k} caracteristicas correctamente:\n" + 
                ", ".join(selected))
        elif len(selected) < k:
            messagebox.showwarning("Advertencia", 
                f"Debes seleccionar {k} caracteristicas. Actualmente: {len(selected)}")
        else:
            messagebox.showwarning("Advertencia", 
                f"Has seleccionado {len(selected)} caracteristicas, pero solo necesitas {k}")
        
    def preprocess_data(self):
        if self.df is None:
            messagebox.showwarning("Advertencia", "Primero debes cargar un archivo CSV")
            return
        
        if not self.selected_features:
            messagebox.showwarning("Advertencia", 
                "Primero debes calcular sugerencias y seleccionar caracteristicas")
            return
        
        if len(self.selected_features) != self.k_features.get():
            messagebox.showwarning("Advertencia", 
                f"Debes seleccionar exactamente {self.k_features.get()} caracteristicas")
            return
        
        try:
            # Separar X e y
            y = self.df['Class'].astype(int).values
            X = self.df[self.selected_features].values
            
            # Split train/test
            test_size = (100 - self.train_size.get()) / 100
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=test_size, 
                stratify=y, 
                random_state=self.random_state.get()
            )
            
            # Escalado
            scaler = StandardScaler()
            X_train_sel = scaler.fit_transform(X_train)
            X_test_sel = scaler.transform(X_test)
            
            # Crear conjunto de validacion
            X_train_final, X_val, y_train_final, y_val = train_test_split(
                X_train_sel, y_train,
                test_size=0.20,
                stratify=y_train,
                random_state=self.random_state.get()
            )
            
            # Guardar
            self.X_train = X_train_final
            self.X_val = X_val
            self.X_test = X_test_sel
            self.y_train = y_train_final
            self.y_val = y_val
            self.y_test = y_test
            
            self.status_label.config(
                text=f"Datos preprocesados | Train: {len(y_train_final)} | Val: {len(y_val)} | Test: {len(y_test)}",
                foreground="green"
            )
            
            messagebox.showinfo("Exito", "Datos preprocesados exitosamente")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al preprocesar datos: {str(e)}")
            
    def train_models(self):
        if self.X_train is None:
            messagebox.showwarning("Advertencia", "Primero debes preprocesar los datos")
            return
        
        # Ejecutar en hilo separado para no bloquear la interfaz
        thread = threading.Thread(target=self.train_models_thread)
        thread.start()
        
    def train_models_thread(self):
        try:
            # Limpiar resultados anteriores
            self.results_grid_text.delete('1.0', tk.END)
            self.results_grid = []
            self.best_models = {}
            
            # Obtener configuracion
            param_grids = get_param_grids()
            models_config = get_models_config(self.random_state.get())
            
            self.progress['value'] = 0
            self.progress['maximum'] = len(models_config)
            
            result_text = "RESULTADOS DEL GRID SEARCH\n"
            result_text += "=" * 80 + "\n\n"
            
            # Entrenar cada modelo
            for idx, (name, model) in enumerate(models_config.items()):
                self.progress_label.config(text=f"Entrenando {name}...")
                
                # Grid Search
                best_model, best_params, best_score, num_combinations = train_with_gridsearch(
                    self.X_train, self.y_train, name, model, 
                    param_grids[name], cv=5, random_state=self.random_state.get()
                )
                
                # Guardar
                self.best_models[name] = best_model
                self.results_grid.append({
                    'Modelo': name,
                    'Mejor_Recall_CV': round(best_score, 4),
                    'Mejores_Hiperparametros': str(best_params),
                    'Num_Combinaciones_Probadas': num_combinations
                })
                
                # Mostrar resultados
                result_text += f"{name}\n"
                result_text += f"  Mejor Recall (CV): {best_score:.4f}\n"
                result_text += f"  Mejores hiperparametros: {best_params}\n"
                result_text += f"  Combinaciones probadas: {num_combinations}\n\n"
                
                self.results_grid_text.delete('1.0', tk.END)
                self.results_grid_text.insert('1.0', result_text)
                
                # Actualizar progreso
                self.progress['value'] = idx + 1
                self.root.update_idletasks()
            
            self.progress_label.config(text="Entrenamiento completado")
            messagebox.showinfo("Exito", "Todos los modelos han sido entrenados exitosamente")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error durante el entrenamiento: {str(e)}")
            
    def evaluate_models(self):
        if not self.best_models:
            messagebox.showwarning("Advertencia", "Primero debes entrenar los modelos")
            return
        
        # Ejecutar en hilo separado
        thread = threading.Thread(target=self.evaluate_models_thread)
        thread.start()
        
    def evaluate_models_thread(self):
        try:
            self.results_test_text.delete('1.0', tk.END)
            self.results_test = []
            
            result_text = "RESULTADOS EN CONJUNTO DE TEST\n"
            result_text += "=" * 80 + "\n\n"
            
            # Evaluar cada modelo
            for name, model in self.best_models.items():
                # Evaluar
                metrics, threshold = evaluate_model(
                    model, self.X_val, self.y_val, 
                    self.X_test, self.y_test, 
                    fn_cost=10.0, fp_cost=1.0
                )
                
                metrics['Modelo'] = name
                self.results_test.append(metrics)
                
                # Mostrar resultados
                result_text += f"{name}\n"
                result_text += f"  Umbral: {threshold:.6f}\n"
                result_text += f"  Recall: {metrics['Recall_Sensibilidad']:.4f} | "
                result_text += f"F1: {metrics['F1_Score']:.4f}\n"
                result_text += f"  FN: {metrics['FN']} | FP: {metrics['FP']} | "
                result_text += f"TP: {metrics['TP']} | TN: {metrics['TN']}\n"
                result_text += f"  Precision: {metrics['Precision']:.4f} | "
                result_text += f"Especificidad: {metrics['Especificidad']:.4f} | "
                result_text += f"Accuracy: {metrics['Exactitud_Accuracy']:.4f}\n\n"
            
            # Mejor modelo
            df_test = pd.DataFrame(self.results_test)
            best_idx = df_test['Recall_Sensibilidad'].idxmax()
            best_model = df_test.loc[best_idx, 'Modelo']
            best_recall = df_test.loc[best_idx, 'Recall_Sensibilidad']
            
            result_text += "=" * 80 + "\n"
            result_text += f"MEJOR MODELO: {best_model} (Recall: {best_recall:.4f})\n"
            
            self.results_test_text.delete('1.0', tk.END)
            self.results_test_text.insert('1.0', result_text)
            
            messagebox.showinfo("Exito", "Evaluacion completada exitosamente")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error durante la evaluacion: {str(e)}")
            
    def export_grid_results(self):
        if not self.results_grid:
            messagebox.showwarning("Advertencia", "No hay resultados para exportar")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filename:
            df = pd.DataFrame(self.results_grid)
            df.to_csv(filename, index=False)
            messagebox.showinfo("Exito", f"Resultados exportados a: {filename}")
            
    def export_test_results(self):
        if not self.results_test:
            messagebox.showwarning("Advertencia", "No hay resultados para exportar")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filename:
            df = pd.DataFrame(self.results_test)
            cols_order = ['Modelo', 'Umbral_Ajustado', 'TP', 'FN', 'FP', 'TN', 
                         'Recall_Sensibilidad', 'Precision', 'F1_Score', 
                         'Especificidad', 'Exactitud_Accuracy']
            df = df[cols_order]
            df.to_csv(filename, index=False)
            messagebox.showinfo("Exito", f"Resultados exportados a: {filename}")


if __name__ == "__main__":
    root = tk.Tk()
    app = FraudDetectionApp(root)
    root.mainloop()
