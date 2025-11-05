"""
Script de inicio para la aplicacion Tkinter
Ejecutar este archivo para abrir la interfaz grafica
"""

import sys
from pathlib import Path

# Agregar directorio actual al path
sys.path.insert(0, str(Path(__file__).parent))

# Importar y ejecutar la aplicacion
from app_tkinter import FraudDetectionApp
import tkinter as tk

if __name__ == "__main__":
    print("Iniciando interfaz grafica con Tkinter...")
    print("Esta version es mas rapida que Streamlit")
    print("-" * 50)
    
    root = tk.Tk()
    app = FraudDetectionApp(root)
    root.mainloop()
