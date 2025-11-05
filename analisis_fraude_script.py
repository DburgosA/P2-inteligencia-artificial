"""
Análisis y Visualización del Dataset de Detección de Fraude con Tarjetas de Crédito

Dataset: Credit Card Fraud Detection Dataset (Kaggle)
- Registros: 284,807
- Fraudes: 492 (0.172%)
- Atributos: Time, Amount, V1-V28 (PCA), Class

Este script genera un análisis completo del dataset y guarda todos los resultados
en la carpeta 'analisis_csv/'
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ========================================
# CONFIGURACIÓN INICIAL
# ========================================

# Crear carpeta de salida
OUTPUT_DIR = Path("analisis_csv")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configuración de visualización
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 70)
print("ANÁLISIS DE DETECCIÓN DE FRAUDE - TARJETAS DE CRÉDITO")
print("=" * 70)
print(f"\nCarpeta de salida: {OUTPUT_DIR.absolute()}\n")

# ========================================
# 1. CARGAR DATASET
# ========================================
print("Cargando dataset...")
df = pd.read_csv('datos/creditcard.csv')
print("Dataset cargado exitosamente!")
print(f"Forma del dataset: {df.shape}")

# Guardar información básica
with open(OUTPUT_DIR / "01_info_general.txt", "w", encoding="utf-8") as f:
    f.write("=" * 60 + "\n")
    f.write("INFORMACIÓN GENERAL DEL DATASET\n")
    f.write("=" * 60 + "\n")
    f.write(f"Forma: {df.shape}\n")
    f.write(f"Columnas: {list(df.columns)}\n\n")
    df.info(buf=f)

# ========================================
# 2. ESTADÍSTICAS DESCRIPTIVAS
# ========================================
print("\nGenerando estadísticas descriptivas...")
df.describe().to_csv(OUTPUT_DIR / "02_estadisticas_descriptivas.csv")

# Guardar primeras filas
df.head(20).to_csv(OUTPUT_DIR / "03_primeras_20_filas.csv", index=False)

# ========================================
# 3. ANÁLISIS DE LA VARIABLE OBJETIVO (CLASS)
# ========================================
print("\nAnalizando variable objetivo (Class)...")

class_counts = df['Class'].value_counts()
class_percentages = df['Class'].value_counts(normalize=True) * 100

analisis_class = f"""
{'=' * 60}
ANÁLISIS DE LA VARIABLE OBJETIVO (Class)
{'=' * 60}

Transacciones legítimas (0): {class_counts[0]:,} ({class_percentages[0]:.3f}%)
Transacciones fraudulentas (1): {class_counts[1]:,} ({class_percentages[1]:.3f}%)

Ratio de desbalance: {class_counts[0]/class_counts[1]:.2f}:1

Dataset altamente desbalanceado: Solo el {class_percentages[1]:.3f}% son fraudes
"""

with open(OUTPUT_DIR / "04_analisis_clases.txt", "w", encoding="utf-8") as f:
    f.write(analisis_class)

print(analisis_class)

# ========================================
# 4. VISUALIZACIÓN DE DISTRIBUCIÓN DE CLASES
# ========================================
print("\nGenerando gráfico 1: Distribución de clases...")

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Gráfico de barras
colors = ['#2ecc71', '#e74c3c']
axes[0].bar(['Legítima (0)', 'Fraude (1)'], class_counts.values, color=colors, alpha=0.7, edgecolor='black')
axes[0].set_ylabel('Número de Transacciones', fontsize=12, fontweight='bold')
axes[0].set_title('Distribución de Clases - Conteo', fontsize=14, fontweight='bold')
axes[0].set_yscale('log')
for i, v in enumerate(class_counts.values):
    axes[0].text(i, v, f'{v:,}', ha='center', va='bottom', fontweight='bold')

# Gráfico de pastel
axes[1].pie(class_counts.values, labels=['Legítima (0)', 'Fraude (1)'], 
            autopct='%1.3f%%', colors=colors, startangle=90, explode=(0, 0.1))
axes[1].set_title('Distribución de Clases - Porcentaje', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_distribucion_clases.png", dpi=300, bbox_inches='tight')
plt.close()

# ========================================
# 5. ANÁLISIS DE LA VARIABLE TIME
# ========================================
print("\nAnalizando variable Time...")

# Convertir Time a horas
df['Time_hours'] = df['Time'] / 3600

analisis_time = f"""
{'=' * 60}
ANÁLISIS DE LA VARIABLE TIME
{'=' * 60}

Tiempo mínimo: {df['Time'].min()} segundos (0 horas)
Tiempo máximo: {df['Time'].max()} segundos ({df['Time_hours'].max():.2f} horas)
Duración total: {df['Time_hours'].max():.2f} horas ({df['Time_hours'].max()/24:.2f} días)
"""

with open(OUTPUT_DIR / "06_analisis_time.txt", "w", encoding="utf-8") as f:
    f.write(analisis_time)

print(analisis_time)

# Visualización de Time
print("Generando gráfico 2: Análisis de Time...")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Distribución general
axes[0, 0].hist(df['Time_hours'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Tiempo (horas)', fontsize=11, fontweight='bold')
axes[0, 0].set_ylabel('Número de Transacciones', fontsize=11, fontweight='bold')
axes[0, 0].set_title('Distribución de Transacciones en el Tiempo', fontsize=12, fontweight='bold')
axes[0, 0].grid(alpha=0.3)

# Distribución por clase
legitimate = df[df['Class'] == 0]['Time_hours']
fraud = df[df['Class'] == 1]['Time_hours']

axes[0, 1].hist([legitimate, fraud], bins=50, label=['Legítima', 'Fraude'], 
                color=['#2ecc71', '#e74c3c'], alpha=0.6, edgecolor='black')
axes[0, 1].set_xlabel('Tiempo (horas)', fontsize=11, fontweight='bold')
axes[0, 1].set_ylabel('Número de Transacciones', fontsize=11, fontweight='bold')
axes[0, 1].set_title('Distribución Temporal por Clase', fontsize=12, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Boxplot
df.boxplot(column='Time_hours', by='Class', ax=axes[1, 0], patch_artist=True)
axes[1, 0].set_xlabel('Clase (0=Legítima, 1=Fraude)', fontsize=11, fontweight='bold')
axes[1, 0].set_ylabel('Tiempo (horas)', fontsize=11, fontweight='bold')
axes[1, 0].set_title('Distribución de Time por Clase', fontsize=12, fontweight='bold')
plt.sca(axes[1, 0])
plt.xticks([1, 2], ['Legítima (0)', 'Fraude (1)'])

# Densidad de fraudes
axes[1, 1].hist(fraud, bins=48, color='#e74c3c', alpha=0.7, edgecolor='black')
axes[1, 1].set_xlabel('Tiempo (horas)', fontsize=11, fontweight='bold')
axes[1, 1].set_ylabel('Número de Fraudes', fontsize=11, fontweight='bold')
axes[1, 1].set_title('Distribución Temporal de Fraudes', fontsize=12, fontweight='bold')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "07_analisis_time.png", dpi=300, bbox_inches='tight')
plt.close()

# ========================================
# 6. ANÁLISIS DE LA VARIABLE AMOUNT
# ========================================
print("\nAnalizando variable Amount...")

analisis_amount = f"""
{'=' * 60}
ANÁLISIS DE LA VARIABLE AMOUNT
{'=' * 60}

Estadísticas globales:
Monto mínimo: ${df['Amount'].min():.2f}
Monto máximo: ${df['Amount'].max():.2f}
Monto promedio: ${df['Amount'].mean():.2f}
Monto mediano: ${df['Amount'].median():.2f}
Desviación estándar: ${df['Amount'].std():.2f}

Transacciones Legítimas:
Monto promedio: ${df[df['Class']==0]['Amount'].mean():.2f}
Monto mediano: ${df[df['Class']==0]['Amount'].median():.2f}

Transacciones Fraudulentas:
Monto promedio: ${df[df['Class']==1]['Amount'].mean():.2f}
Monto mediano: ${df[df['Class']==1]['Amount'].median():.2f}
"""

with open(OUTPUT_DIR / "08_analisis_amount.txt", "w", encoding="utf-8") as f:
    f.write(analisis_amount)

print(analisis_amount)

# Visualización de Amount
print("Generando gráfico 3: Análisis de Amount...")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Histograma general
axes[0, 0].hist(df['Amount'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Monto ($)', fontsize=11, fontweight='bold')
axes[0, 0].set_ylabel('Frecuencia', fontsize=11, fontweight='bold')
axes[0, 0].set_title('Distribución del Monto de Transacciones', fontsize=12, fontweight='bold')
axes[0, 0].set_yscale('log')
axes[0, 0].grid(alpha=0.3)

# Comparación por clase
legitimate_amount = df[df['Class'] == 0]['Amount']
fraud_amount = df[df['Class'] == 1]['Amount']

axes[0, 1].hist([legitimate_amount, fraud_amount], bins=50, 
                label=['Legítima', 'Fraude'], 
                color=['#2ecc71', '#e74c3c'], alpha=0.6, edgecolor='black')
axes[0, 1].set_xlabel('Monto ($)', fontsize=11, fontweight='bold')
axes[0, 1].set_ylabel('Frecuencia', fontsize=11, fontweight='bold')
axes[0, 1].set_title('Distribución de Montos por Clase', fontsize=12, fontweight='bold')
axes[0, 1].set_yscale('log')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Boxplot
df.boxplot(column='Amount', by='Class', ax=axes[1, 0], patch_artist=True)
axes[1, 0].set_xlabel('Clase (0=Legítima, 1=Fraude)', fontsize=11, fontweight='bold')
axes[1, 0].set_ylabel('Monto ($)', fontsize=11, fontweight='bold')
axes[1, 0].set_title('Distribución de Montos por Clase (Boxplot)', fontsize=12, fontweight='bold')
axes[1, 0].set_yscale('log')
plt.sca(axes[1, 0])
plt.xticks([1, 2], ['Legítima (0)', 'Fraude (1)'])

# Violin plot
parts = axes[1, 1].violinplot([legitimate_amount[legitimate_amount > 0], 
                                fraud_amount[fraud_amount > 0]], 
                               positions=[0, 1], showmeans=True, showmedians=True)
axes[1, 1].set_xlabel('Clase', fontsize=11, fontweight='bold')
axes[1, 1].set_ylabel('Monto ($)', fontsize=11, fontweight='bold')
axes[1, 1].set_title('Distribución de Montos por Clase (Violin Plot)', fontsize=12, fontweight='bold')
axes[1, 1].set_xticks([0, 1])
axes[1, 1].set_xticklabels(['Legítima (0)', 'Fraude (1)'])
axes[1, 1].set_yscale('log')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "09_analisis_amount.png", dpi=300, bbox_inches='tight')
plt.close()

# ========================================
# 7. ANÁLISIS DE VARIABLES PCA (V1-V28)
# ========================================
print("\nAnalizando variables PCA (V1-V28)...")

v_columns = [col for col in df.columns if col.startswith('V')]
print(f"Variables PCA encontradas: {len(v_columns)}")

# Matriz de correlación
print("Generando gráfico 4: Matriz de correlación PCA...")

plt.figure(figsize=(16, 14))
correlation_matrix = df[v_columns].corr()
sns.heatmap(correlation_matrix, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Matriz de Correlación de Variables PCA (V1-V28)', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "10_correlacion_pca.png", dpi=300, bbox_inches='tight')
plt.close()

# Guardar matriz de correlación
correlation_matrix.to_csv(OUTPUT_DIR / "11_matriz_correlacion_pca.csv")

# Distribución de variables PCA
print("Generando gráfico 5: Distribución de variables PCA (V1-V9)...")

fig, axes = plt.subplots(3, 3, figsize=(18, 14))
axes = axes.ravel()

for i, col in enumerate(v_columns[:9]):
    legitimate_v = df[df['Class'] == 0][col]
    fraud_v = df[df['Class'] == 1][col]
    
    axes[i].hist([legitimate_v, fraud_v], bins=50, 
                 label=['Legítima', 'Fraude'],
                 color=['#2ecc71', '#e74c3c'], alpha=0.6, edgecolor='black')
    axes[i].set_xlabel(col, fontsize=10, fontweight='bold')
    axes[i].set_ylabel('Frecuencia', fontsize=10, fontweight='bold')
    axes[i].set_title(f'Distribución de {col}', fontsize=11, fontweight='bold')
    axes[i].legend(fontsize=8)
    axes[i].grid(alpha=0.3)

plt.suptitle('Distribución de Variables PCA (V1-V9) por Clase', 
             fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "12_distribucion_pca_v1_v9.png", dpi=300, bbox_inches='tight')
plt.close()

# ========================================
# 8. CORRELACIÓN CON VARIABLE OBJETIVO
# ========================================
print("\nAnalizando correlación con variable objetivo...")

correlations = df.corr()['Class'].drop('Class').sort_values(ascending=False)

analisis_correlacion = f"""
{'=' * 60}
CORRELACIÓN CON LA VARIABLE OBJETIVO (Class)
{'=' * 60}

Top 10 variables con mayor correlación positiva:
{correlations.head(10)}

Top 10 variables con mayor correlación negativa:
{correlations.tail(10)}
"""

with open(OUTPUT_DIR / "13_correlacion_con_class.txt", "w", encoding="utf-8") as f:
    f.write(analisis_correlacion)

print(analisis_correlacion)

# Guardar correlaciones completas
correlations.to_csv(OUTPUT_DIR / "14_correlaciones_completas.csv", header=['Correlación_con_Class'])

# Visualizar correlaciones
print("Generando gráfico 6: Correlaciones con Class...")

plt.figure(figsize=(12, 8))
correlations.plot(kind='barh', color=['#e74c3c' if x < 0 else '#2ecc71' for x in correlations])
plt.xlabel('Correlación con Class', fontsize=12, fontweight='bold')
plt.ylabel('Variables', fontsize=12, fontweight='bold')
plt.title('Correlación de Variables con la Variable Objetivo (Class)', 
          fontsize=14, fontweight='bold')
plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
plt.grid(alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "15_grafico_correlaciones.png", dpi=300, bbox_inches='tight')
plt.close()

# ========================================
# 9. VARIABLES MÁS RELEVANTES PARA FRAUDE
# ========================================
print("\nAnalizando variables más relevantes para fraude...")

top_positive = correlations.head(3).index.tolist()
top_negative = correlations.tail(3).index.tolist()
top_features = top_positive + top_negative

analisis_relevantes = f"""
Variables más relevantes para detectar fraude:

Correlación positiva: {', '.join(top_positive)}
Correlación negativa: {', '.join(top_negative)}
"""

with open(OUTPUT_DIR / "16_variables_relevantes.txt", "w", encoding="utf-8") as f:
    f.write(analisis_relevantes)

print(analisis_relevantes)

# Visualizar variables más importantes
print("Generando gráfico 7: Variables más correlacionadas...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()

for i, feature in enumerate(top_features):
    legitimate_f = df[df['Class'] == 0][feature]
    fraud_f = df[df['Class'] == 1][feature]
    
    axes[i].hist([legitimate_f, fraud_f], bins=50,
                 label=['Legítima', 'Fraude'],
                 color=['#2ecc71', '#e74c3c'], alpha=0.6, edgecolor='black')
    axes[i].set_xlabel(feature, fontsize=11, fontweight='bold')
    axes[i].set_ylabel('Frecuencia', fontsize=11, fontweight='bold')
    axes[i].set_title(f'{feature} (Corr: {correlations[feature]:.3f})', 
                      fontsize=12, fontweight='bold')
    axes[i].legend()
    axes[i].grid(alpha=0.3)
    axes[i].set_yscale('log')

plt.suptitle('Variables más Correlacionadas con Fraude', 
             fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "17_variables_mas_correlacionadas.png", dpi=300, bbox_inches='tight')
plt.close()

# ========================================
# 10. SCATTER PLOTS DE VARIABLES CLAVE
# ========================================
print("\nGenerando gráfico 8: Scatter plots de variables clave...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Tomar muestra
sample_size = 5000
df_sample = df.sample(n=min(sample_size, len(df)), random_state=42)

# Plot 1: V17 vs V14
for class_val, color, label in [(0, '#2ecc71', 'Legítima'), (1, '#e74c3c', 'Fraude')]:
    mask = df_sample['Class'] == class_val
    axes[0].scatter(df_sample[mask]['V17'], df_sample[mask]['V14'],
                   c=color, alpha=0.5, s=30, label=label, edgecolors='black', linewidth=0.5)

axes[0].set_xlabel('V17', fontsize=12, fontweight='bold')
axes[0].set_ylabel('V14', fontsize=12, fontweight='bold')
axes[0].set_title('V17 vs V14', fontsize=13, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Plot 2: V12 vs V10
for class_val, color, label in [(0, '#2ecc71', 'Legítima'), (1, '#e74c3c', 'Fraude')]:
    mask = df_sample['Class'] == class_val
    axes[1].scatter(df_sample[mask]['V12'], df_sample[mask]['V10'],
                   c=color, alpha=0.5, s=30, label=label, edgecolors='black', linewidth=0.5)

axes[1].set_xlabel('V12', fontsize=12, fontweight='bold')
axes[1].set_ylabel('V10', fontsize=12, fontweight='bold')
axes[1].set_title('V12 vs V10', fontsize=13, fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.suptitle('Relación entre Variables PCA más Relevantes', 
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "18_scatter_plots_variables_clave.png", dpi=300, bbox_inches='tight')
plt.close()

# ========================================
# 11. RESUMEN ESTADÍSTICO FINAL
# ========================================
print("\nGenerando resumen estadístico final...")

resumen_final = f"""
{'=' * 70}
RESUMEN ESTADÍSTICO DEL DATASET
{'=' * 70}

 INFORMACIÓN GENERAL:
   Total de transacciones: {len(df):,}
   Total de atributos: {len(df.columns)}
   Transacciones legítimas: {class_counts[0]:,} ({class_percentages[0]:.3f}%)
   Transacciones fraudulentas: {class_counts[1]:,} ({class_percentages[1]:.3f}%)
   Ratio de desbalance: {class_counts[0]/class_counts[1]:.2f}:1

⏱ TIEMPO:
   Duración del dataset: {df['Time_hours'].max():.2f} horas ({df['Time_hours'].max()/24:.2f} días)

 MONTOS:
   Rango de montos: ${df['Amount'].min():.2f} - ${df['Amount'].max():.2f}
   Monto promedio (legítimas): ${df[df['Class']==0]['Amount'].mean():.2f}
   Monto promedio (fraudes): ${df[df['Class']==1]['Amount'].mean():.2f}

 VARIABLES PCA:
   Número de componentes principales: {len(v_columns)}
   Variable más correlacionada (+): {correlations.idxmax()} ({correlations.max():.3f})
   Variable más correlacionada (-): {correlations.idxmin()} ({correlations.min():.3f})

 CALIDAD DE DATOS:
   Valores faltantes: {df.isnull().sum().sum()}
   Duplicados: {df.duplicated().sum()}

{'=' * 70}
"""

with open(OUTPUT_DIR / "19_resumen_final.txt", "w", encoding="utf-8") as f:
    f.write(resumen_final)

print(resumen_final)

# ========================================
# 12. EXPORTAR RESUMEN NUMÉRICO
# ========================================
print("\nExportando resumen numérico...")

resumen_datos = {
    'Total_Transacciones': len(df),
    'Transacciones_Legitimas': class_counts[0],
    'Transacciones_Fraudulentas': class_counts[1],
    'Porcentaje_Fraude': class_percentages[1],
    'Duracion_Horas': df['Time_hours'].max(),
    'Duracion_Dias': df['Time_hours'].max() / 24,
    'Monto_Promedio_Legitimas': df[df['Class']==0]['Amount'].mean(),
    'Monto_Promedio_Fraudes': df[df['Class']==1]['Amount'].mean(),
    'Monto_Mediana_Legitimas': df[df['Class']==0]['Amount'].median(),
    'Monto_Mediana_Fraudes': df[df['Class']==1]['Amount'].median(),
    'Ratio_Desbalance': class_counts[0]/class_counts[1],
    'Variable_Mas_Correlacionada_Positiva': correlations.idxmax(),
    'Correlacion_Maxima': correlations.max(),
    'Variable_Mas_Correlacionada_Negativa': correlations.idxmin(),
    'Correlacion_Minima': correlations.min()
}

resumen_df = pd.DataFrame([resumen_datos])
resumen_df.to_csv(OUTPUT_DIR / '20_resumen_numerico.csv', index=False)

print("\n" + "=" * 70)
print(" ANÁLISIS COMPLETADO EXITOSAMENTE")
print("=" * 70)
print(f"\nTodos los archivos han sido guardados en: {OUTPUT_DIR.absolute()}")
print(f"\nArchivos generados:")
print(f"  - 20 archivos (TXT, CSV, PNG)")
print(f"  - 8 gráficos en alta resolución (300 DPI)")
print(f"  - Matrices de correlación y estadísticas completas")
print("\n" + "=" * 70)
