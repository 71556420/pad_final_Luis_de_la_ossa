# Proyecto Integrador - Análisis del MCU Box Office Dataset

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import gdown

# URL de descarga directa de Google Drive
file_id = "1jxPMcs9oe2hlxd8GSwXQQqDwWLYIrJZN"
download_url = f"https://drive.google.com/uc?id={file_id}"

dataset_dir = "./mcu_dataset"
dataset_path = os.path.join(dataset_dir, "marvel_cinematic_universe_box_office.csv")

# Crear el directorio si no existe
os.makedirs(dataset_dir, exist_ok=True)

# Descargar el archivo si no existe
if not os.path.exists(dataset_path):
    print("Descargando dataset desde Google Drive...")
    try:
        gdown.download(download_url, dataset_path, quiet=False)
        print("Descarga completa.")
    except Exception as e:
        print(f"ERROR: Ocurrió un problema al descargar el archivo. {e}")

# Cargar dataset
df = None
if os.path.exists(dataset_path):
    try:
        df = pd.read_csv(dataset_path, encoding='utf-8')
        df.columns = df.columns.str.strip()  # Eliminar espacios en nombres de columnas
        print("Dataset cargado con éxito.")
    except Exception as e:
        print(f"ERROR: No se pudo cargar el dataset. {e}")
else:
    print("ERROR: No se encontró el archivo CSV. Verifica la descarga.")

if df is not None:
    # 2. Exploración de datos
    print("Vista previa del dataset:")
    print(df.head(10))
    print(f"\nEl dataset tiene {df.shape[0]} filas y {df.shape[1]} columnas.")
    print("\nColumnas disponibles exactamente como están:")
    for col in df.columns:
        print(f"- '{col}'")

    print("\nInformación general:")
    print(df.info())
    print("\nEstadísticas descriptivas:")
    print(df.describe())

    # 3. Limpieza de datos
    print("\nValores nulos por columna:")
    print(df.isnull().sum())

    # Eliminar filas con valores nulos
    df.dropna(inplace=True)

    # 4. Conversión de tipos de datos
    # Convertir 'release_date' a tipo datetime
    if 'release_date' in df.columns:
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        df.dropna(subset=['release_date'], inplace=True)

    # Convertir columnas de ingresos a numéricas
    income_columns = ['production_budget', 'opening_weekend', 'domestic_box_office', 'worldwide_box_office']
    for col in income_columns:
        if col in df.columns:
            df[col] = df[col].replace('[\$,]', '', regex=True).astype(float)

    ## 5. Visualización de datos

# Identificar columna de ingresos
known_possible_columns = [
    'worldwide_box_office'
]
matching_col = None
for candidate in known_possible_columns:
    for col in df.columns:
        if candidate.lower() == col.lower():
            matching_col = col
            break
    if matching_col:
        break

if matching_col:
    print(f"Se utilizará la columna '{matching_col}' para los gráficos.")

    # Convertir columna de ingresos a numérica si es necesario
    df[matching_col] = pd.to_numeric(df[matching_col], errors='coerce')

    # Eliminar filas con valores nulos en 'worldwide_box_office' o 'movie_title'
    df = df.dropna(subset=[matching_col, 'movie_title'])

    # Gráfico de distribución de ingresos
    plt.figure(figsize=(10, 6))
    sns.histplot(df[matching_col], bins=20, kde=True)
    plt.title("Distribución de ingresos en taquilla del MCU")
    plt.xlabel("Ingresos en dólares")
    plt.ylabel("Frecuencia")
    plt.show()

    # Gráfico de tendencia de ingresos por fecha de estreno
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='release_date', y=matching_col, data=df, marker='o')
    plt.xticks(rotation=45)
    plt.title("Tendencia de ingresos en taquilla por fecha de estreno")
    plt.xlabel("Fecha de estreno")
    plt.ylabel("Ingresos en dólares")
    plt.show()

    # Gráfico de las 10 películas con mayores ingresos en taquilla
    top_10 = df.sort_values(by=matching_col, ascending=False).head(10)
    plt.figure(figsize=(12, 6))
    sns.barplot(x='worldwide_box_office', y='movie_title', data=top_10, hue='movie_title', palette='viridis', legend=False)
    plt.title("Top 10 películas con mayores ingresos en taquilla")
    plt.xlabel("Ingresos en dólares")
    plt.ylabel("Película")
    plt.show()
else:
    print("ERROR: No se encontró una columna válida para los ingresos en taquilla.")

    # 6. Matriz de correlación
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Matriz de correlación entre variables")
    plt.show()

    # 7. Exportación del dataset limpio
    df.to_csv("MCU_Box_Office_Clean.csv", index=False, encoding='utf-8')
    print("Proceso completado. Dataset limpio exportado como 'MCU_Box_Office_Clean.csv'")
