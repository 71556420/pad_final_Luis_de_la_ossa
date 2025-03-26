import pandas as pd
import os

def crear_dataframe_frutas():
    """
    # 1 Crea un DataFrame de Pandas con la información de frutas.

    Returns:
        pandas.DataFrame: DataFrame con la información de frutas.
    """
    datos = {'Granadilla': [20], 'Tomates': [50]}
    df_frutas = pd.DataFrame(datos)
    df_frutas.to_csv('static/frutas.csv', index=False)  # Guarda el DataFrame en un archivo CSV dentro de la carpeta 'static'
    return df_frutas

if __name__ == "__main__":
    df_frutas = crear_dataframe_frutas()
    print(df_frutas)

def crear_dataframe_ventas_frutas():
    """
    # 2 Crea un DataFrame de Pandas con la información de ventas de frutas.

    Returns:
        pandas.DataFrame: DataFrame con la información de ventas de frutas.
    """
    datos = {
        'Granadilla': [20, 49],
        'Tomates': [50, 100]
    }
    indices = ['ventas 2021', 'ventas 2022']
    df_ventas_frutas = pd.DataFrame(datos, index=indices)
    df_ventas_frutas.to_csv('static/ventas_frutas.csv')  # Guarda el DataFrame en un archivo CSV dentro de la carpeta 'static'
    return df_ventas_frutas

if __name__ == "__main__":
    df_ventas_frutas = crear_dataframe_ventas_frutas()
    print(df_ventas_frutas)

def crear_serie_utensilios():
    """
    # 3 Crea una Serie de Pandas con la información de utensilios de cocina.

    Returns:
        pandas.Series: Serie con la información de utensilios de cocina.
    """
    datos = {
        'Cuchara': '3 unidades',
        'Tenedor': '2 unidades',
        'Cuchillo': '4 unidades',
        'Plato': '5 unidades'
    }
    serie_utensilios = pd.Series(datos, name='Cocina')
    serie_utensilios.to_csv('static/utensilios.csv', header=True)  # Guarda la Serie en un archivo CSV dentro de la carpeta 'static'
    return serie_utensilios

if __name__ == "__main__":
    serie_utensilios = crear_serie_utensilios()
    print(serie_utensilios)

# 4 Crea una función que cargue el dataset 'wine review' en un DataFrame y lo devuelva.

def cargar_dataset():
    """Carga el dataset 'wine review' en un DataFrame y lo devuelve."""
    file_path = os.path.join(os.path.dirname(__file__), '../static/winemag-data-130k-v2.csv')
    if os.path.exists(file_path):
        review = pd.read_csv(file_path, index_col=0)
        return review
    else:
        raise FileNotFoundError("El archivo del dataset no se encuentra en la carpeta 'static'. Asegúrate de descargarlo desde Kaggle.")

if __name__ == "__main__":
    df_review = cargar_dataset()
    print(df_review.head())

# 5 Visualiza las primeras 5 filas del DataFrame 'df_review'.

def visualizar_primeras_filas():
    """Carga el dataset 'wine review' y muestra las primeras 5 filas."""
    try:
        df_review = cargar_dataset()
        df_review.head().to_csv('static/wine_review_head.csv')  # Guarda las primeras 5 filas en un archivo CSV dentro de la carpeta 'static'
        print(df_review.head())
    except FileNotFoundError as e:
        print(e)

if __name__ == "__main__":
    visualizar_primeras_filas()

# 6 Utiliza el método .info() para averiguar cuántas entradas hay. ¿Cuántas encontraste?

def mostrar_info_dataset():
    """Carga el dataset 'wine review' y muestra información sobre el DataFrame."""
    try:
        df_review = cargar_dataset()
        df_review.info()  # Imprime la información del DataFrame en la terminal
    except FileNotFoundError as e:
        print(e)

if __name__ == "__main__":
    mostrar_info_dataset()

# Entradas visualizadas en el método .info(): 129971

def obtener_numero_entradas():
    """Carga el dataset 'wine review' y devuelve el número de entradas."""
    try:
        df_review = cargar_dataset()
        return len(df_review)
    except FileNotFoundError as e:
        return f"Error: {e}"

if __name__ == "__main__":
    numero_entradas = obtener_numero_entradas()
    print(f"Número de entradas en el DataFrame: {numero_entradas}")

# 7 Utiliza el método .describe() para obtener información estadística sobre el DataFrame. ¿Cuál es el precio promedio de un vino?

def obtener_precio_promedio():
    """Carga el dataset 'wine review' y devuelve el precio promedio."""
    try:
        df_review = cargar_dataset()
        precio_promedio = df_review['price'].mean()
        return precio_promedio
    except FileNotFoundError as e:
        return f"Error: {e}"

if __name__ == "__main__":
    precio_promedio = obtener_precio_promedio()
    print(f"Precio promedio de los vinos: {precio_promedio}")

# 8 Utiliza el método .describe() para obtener información estadística sobre el DataFrame. ¿Cuál es el precio más alto pagado?

def obtener_precio_maximo():
    """Carga el dataset 'wine review' y devuelve el precio máximo."""
    try:
        df_review = cargar_dataset()
        precio_maximo = df_review['price'].max()
        return precio_maximo
    except FileNotFoundError as e:
        return f"Error: {e}"

if __name__ == "__main__":
    precio_maximo = obtener_precio_maximo()
    print(f"Precio máximo pagado por un vino: {precio_maximo}")

# 9 Crea DataFrame con vinos de California.

def obtener_vinos_california():
    """Carga el dataset 'wine review' y devuelve un DataFrame con vinos de California."""
    try:
        df_review = cargar_dataset()
        vinos_california = df_review[df_review['province'] == 'California']
        vinos_california.to_csv('static/vinos_california.csv')  # Guarda el DataFrame en un archivo CSV dentro de la carpeta 'static'
        return vinos_california
    except FileNotFoundError as e:
        return f"Error: {e}"

if __name__ == "__main__":
    vinos_california = obtener_vinos_california()
    if isinstance(vinos_california, pd.DataFrame):
        print(vinos_california.head())  # Muestra las primeras filas del DataFrame
    else:
        print(vinos_california)  # Muestra el mensaje de error si ocurre

# 10 Utiliza idxmax() para encontrar el índice del primer vino más costoso y devolver la información de ese vino.

def obtener_vino_precio_maximo():
    """Carga el dataset 'wine review' y devuelve la información del vino con el precio máximo."""
    try:
        df_review = cargar_dataset()
        indice_precio_maximo = df_review['price'].idxmax()
        vino_precio_maximo = df_review.loc[indice_precio_maximo]
        vino_precio_maximo.to_csv('static/vino_mas_caro.csv', header=True)  # Guarda la Serie en un archivo CSV dentro de la carpeta 'static'
        return vino_precio_maximo
    except FileNotFoundError as e:
        return f"Error: {e}"

if __name__ == "__main__":
    vino_mas_caro = obtener_vino_precio_maximo()
    if isinstance(vino_mas_caro, pd.Series):
        print("El vino más caro es:")
        print(vino_mas_caro)
    else:
        print(vino_mas_caro)  # Muestra el mensaje de error si ocurre

# 11 ¿Cuáles son los tipos de uvas más comunes en California?

def obtener_tipos_uva_comunes_california():
    """Carga el dataset 'wine review' y devuelve los tipos de uva más comunes en California."""
    try:
        df_review = cargar_dataset()
        vinos_california = df_review[df_review['province'] == 'California']
        tipos_uva_comunes = vinos_california['variety'].value_counts()
        tipos_uva_comunes.to_csv('static/tipos_uva_comunes_california.csv', header=True)
        return tipos_uva_comunes
    except FileNotFoundError as e:
        return f"Error: {e}"

if __name__ == "__main__":
    tipos_uva_california = obtener_tipos_uva_comunes_california()
    if isinstance(tipos_uva_california, pd.Series):
        print("Tipos de uva más comunes en California:")
        print(tipos_uva_california)
    else:
        print(tipos_uva_california)

# 12 ¿Cuáles son los 10 tipos de uva más comunes en California?

def obtener_top_10_tipos_uva_california():
    """Carga el dataset 'wine review' y devuelve los 10 tipos de uva más comunes en California."""
    try:
        df_review = cargar_dataset()
        vinos_california = df_review[df_review['province'] == 'California']
        top_10_tipos_uva = vinos_california['variety'].value_counts().head(10)
        top_10_tipos_uva.to_csv('static/top_10_tipos_uva_california.csv', header=True)
        return top_10_tipos_uva
    except FileNotFoundError as e:
        return f"Error: {e}"

if __name__ == "__main__":
    top_10_uva_california = obtener_top_10_tipos_uva_california()
    if isinstance(top_10_uva_california, pd.Series):
        print("Los 10 tipos de uva más comunes en California:")
        print(top_10_uva_california)
    else:
        print(top_10_uva_california)