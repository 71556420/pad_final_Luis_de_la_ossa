import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

class NumpyOperations:
    def generate_array(self, start, end):
        return np.arange(start, end + 1)

    def sum_ones_array(self, size):
        return np.ones((size, size)).sum()

    def elementwise_product(self, size, max_val):
        arr1 = np.random.randint(1, max_val + 1, size)
        arr2 = np.random.randint(1, max_val + 1, size)
        return arr1 * arr2

    def matrix_inverse(self, size):
        matrix = np.random.rand(size, size)
        return np.linalg.inv(matrix)

    def find_min_max(self, size):
        arr = np.random.rand(size)
        return arr.min(), arr.argmin(), arr.max(), arr.argmax()

class DataHandling:
    def save_array_to_csv(self, array, filename):
        if not os.path.exists("src/pad20251/static/csv"):
            os.makedirs("src/pad20251/static/csv")
        df = pd.DataFrame(array)
        df.to_csv(f"src/pad20251/static/csv/{filename}.csv", index=False)

numpy_ops = NumpyOperations()
data_handler = DataHandling()

# 1. Genera un array de NumPy con valores desde 10 hasta 29.
array_1 = numpy_ops.generate_array(10, 29)
data_handler.save_array_to_csv(array_1, "array_1")

# 2. Calcula la suma de todos los elementos en un array de NumPy de tamaño 10x10, lleno de unos.
array_2 = np.ones((10, 10))
data_handler.save_array_to_csv(array_2, "array_2")
sum_2 = numpy_ops.sum_ones_array(10)
print(f"Suma de unos: {sum_2}")

# 3. Dados dos arrays de tamaño 5, llenos de números aleatorios desde 1 hasta 10, realiza un producto elemento a elemento.
product_3 = numpy_ops.elementwise_product(5, 10)
data_handler.save_array_to_csv(product_3, "product_3")

# 4. Crea una matriz de 4x4, donde cada elemento es igual a i+j (con i y j siendo el índice de fila y columna, respectivamente) y calcula su inversa.
inverse_4 = numpy_ops.matrix_inverse(4)
data_handler.save_array_to_csv(inverse_4, "inverse_4")

# 5. Encuentra los valores máximo y mínimo en un array de 100 elementos aleatorios y muestra sus índices.
array_5 = np.random.rand(100)
data_handler.save_array_to_csv(array_5, "array_5")
min_val_5, min_index_5, max_val_5, max_index_5 = numpy_ops.find_min_max(100)
print(f"Valor mínimo: {min_val_5}, Índice mínimo: {min_index_5}")
print(f"Valor máximo: {max_val_5}, Índice máximo: {max_index_5}")

class NumpyOperations:
    # ... (funciones de los puntos 1-5) ...

    def broadcasting_sum(self, size1, size2):
        arr1 = np.random.rand(size1, 1)
        arr2 = np.random.rand(1, size2)
        return arr1 + arr2

    def extract_submatrix(self, matrix, start_row, start_col, size):
        return matrix[start_row:start_row + size, start_col:start_col + size]

    def modify_array(self, size, start, end, value):
        arr = np.zeros(size)
        arr[start:end + 1] = value
        return arr

    def reverse_rows(self, matrix):
        return matrix[::-1]

    def select_greater_than(self, arr, threshold):
        return arr[arr > threshold]
class DataHandling:
    def save_array_to_csv(self, array, filename):
        if not os.path.exists("src/pad20251/static/csv"):
            os.makedirs("src/pad20251/static/csv")
        df = pd.DataFrame(array)
        df.to_csv(f"src/pad20251/static/csv/{filename}.csv", index=False)

numpy_ops = NumpyOperations()
data_handler = DataHandling()

# ejecución de los puntos 1-5

# 6. Crea un array de tamaño 3x1 y uno de 1x3, y súmalos utilizando broadcasting para obtener un array de 3x3.
sum_6 = numpy_ops.broadcasting_sum(3, 3)
data_handler.save_array_to_csv(sum_6, "sum_6")

# 7. De una matriz 5x5, extrae una submatriz 2x2 que comience en la segunda fila y columna.
matrix_7 = np.random.rand(5, 5)
submatrix_7 = numpy_ops.extract_submatrix(matrix_7, 1, 1, 2)
data_handler.save_array_to_csv(submatrix_7, "submatrix_7")

# 8. Crea un array de ceros de tamaño 10 y usa indexado para cambiar el valor de los elementos en el rango de índices 3 a 6 a 5.
array_8 = numpy_ops.modify_array(10, 3, 6, 5)
data_handler.save_array_to_csv(array_8, "array_8")

# 9. Dada una matriz de 3x3, invierte el orden de sus filas.
matrix_9 = np.random.rand(3, 3)
reversed_matrix_9 = numpy_ops.reverse_rows(matrix_9)
data_handler.save_array_to_csv(reversed_matrix_9, "reversed_matrix_9")

# 10. Dado un array de números aleatorios de tamaño 10, selecciona y muestra solo aquellos que sean mayores a 0.5.
array_10 = np.random.rand(10)
selected_10 = numpy_ops.select_greater_than(array_10, 0.5)
data_handler.save_array_to_csv(selected_10, "selected_10")

# 11. Genera dos arrays de tamaño 100 con números aleatorios y crea un gráfico de dispersión.
        
class Plotting:
    def scatter_plot(self, x, y, filename):
        img_path = "src/pad20251/static/img"
        if not os.path.exists(img_path):
            os.makedirs(img_path)
        plt.figure()
        plt.scatter(x, y)
        plt.savefig(os.path.join(img_path, f"{filename}.png"))
        plt.close()

# Instancia de la clase Plotting
plotting = Plotting()

# 11. Genera dos arrays de tamaño 100 con números aleatorios y crea un gráfico de dispersión.
x_11 = np.random.rand(100)
y_11 = np.random.rand(100)
plotting.scatter_plot(x_11, y_11, "scatter_11")

# 12. Genera un gráfico de dispersión las variables 𝑥 y 𝑦 = 𝑠𝑖𝑛(𝑥)+ ruido Gaussiano.
class Plotting:
    def scatter_plot(self, x, y, filename):
        img_path = "src/pad20251/static/img"
        if not os.path.exists(img_path):
            os.makedirs(img_path)
        plt.figure()
        plt.scatter(x, y)
        plt.savefig(os.path.join(img_path, f"{filename}.png"))
        plt.close()

# 13. Utiliza la función np.meshgrid para crear una cuadrícula y luego aplica la función z = np.cos(x) + np.sin(y) para generar y mostrar un gráfico de contorno.
        plotting.contour_plot("contour_13")

    def scatter_sin_plot(self, filename):
        img_path = "src/pad20251/static/img"
        if not os.path.exists(img_path):
            os.makedirs(img_path)
        x = np.linspace(-2 * np.pi, 2 * np.pi, 100)
        y = np.sin(x) + np.random.normal(0, 0.3, 100)
        plt.figure()
        plt.scatter(x, y, label="$sin(x)$ + ruido")
        plt.plot(x, np.sin(x), color='red', label="$sin(x)$")
        plt.xlabel('Eje X')
        plt.ylabel('Eje Y')
        plt.title('Gráfico de Dispersión')
        plt.legend()
        plt.savefig(os.path.join(img_path, f"{filename}.png"))
        plt.close()

# 14. Crea un gráfico de dispersión con 1000 puntos aleatorios y utiliza la densidad de estos puntos para ajustar el color de cada punto.
        plotting.density_scatter_plot("density_scatter_14")

    def contour_plot(self, filename):
        img_path = "src/pad20251/static/img"
        if not os.path.exists(img_path):
            os.makedirs(img_path)
        x = np.linspace(-5, 5, 100)
        y = np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.cos(X) + np.sin(Y)
        plt.figure()
        plt.contour(X, Y, Z)
        plt.savefig(os.path.join(img_path, f"{filename}.png"))
        plt.close()

# 15. A partir de la misma función del ejercicio 12, genera un gráfico de contorno lleno.
        plotting.filled_contour_plot("filled_contour_15")

    def density_scatter_plot(self, filename):
        img_path = "src/pad20251/static/img"
        if not os.path.exists(img_path):
            os.makedirs(img_path)
        x = np.random.rand(1000)
        y = np.random.rand(1000)
        plt.figure()
        plt.scatter(x, y, c=np.arctan2(y, x), cmap='viridis')
        plt.savefig(os.path.join(img_path, f"{filename}.png"))
        plt.close()

# 16. Añade etiquetas para el eje X (‘Eje X’), eje Y (‘Eje Y’) y un título (‘Gráfico de Dispersión’) a tu gráfico de dispersión del ejercicio 12 y crea leyendas para cada gráfico usando código LaTex
# (Ya se incluyeron las etiquetas, títulos y leyendas en la función scatter_sin_plot)

    def filled_contour_plot(self, filename):
        img_path = "src/pad20251/static/img"
        if not os.path.exists(img_path):
            os.makedirs(img_path)
        x = np.linspace(-2 * np.pi, 2 * np.pi, 100)
        y = np.linspace(-2 * np.pi, 2 * np.pi, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(X) + np.cos(Y)
        plt.figure()
        plt.contourf(X, Y, Z)  # Usar plt.contourf en lugar de plt.tricontourf
        plt.savefig(os.path.join(img_path, f"{filename}.png"))
        plt.close()

# Instancias de las clases
numpy_ops = NumpyOperations()
data_handler = DataHandling()
plotting = Plotting()

# ejecución de los puntos 17-21

class Plotting:
    # ... (funciones de los puntos 11-16) ...

# 17. Crea un histograma a partir de un array de 1000 números aleatorios generados con una distribución normal.
    def histogram(self, data, bins, filename, mean_line=False, mean=0):
        img_path = "src/pad20251/static/img"
        if not os.path.exists(img_path):
            os.makedirs(img_path)
        plt.figure()
        plt.hist(data, bins=bins, alpha=0.7)
        if mean_line:
            plt.axvline(mean, color='red', linestyle='dashed', linewidth=2)
        plt.savefig(os.path.join(img_path, f"{filename}.png"))
        plt.close()

    
# Instancias de las clases
numpy_ops = NumpyOperations()
data_handler = DataHandling()
plotting = Plotting()

class Plotting:
    
# 17. Crea un histograma a partir de un array de 1000 números aleatorios generados con una distribución normal.
    def histogram(self, data, bins, filename, mean_line=False, mean=0):
        img_path = "src/pad20251/static/img"
        if not os.path.exists(img_path):
            os.makedirs(img_path)
        plt.figure()
        plt.hist(data, bins=bins, alpha=0.7)
        if mean_line:
            plt.axvline(mean, color='red', linestyle='dashed', linewidth=2)
        plt.savefig(os.path.join(img_path, f"{filename}.png"))
        plt.close()

# 18. Genera dos sets de datos con distribuciones normales diferentes y muéstralos en el mismo histograma.
    def overlapping_histograms(self, data1, data2, bins, filename):
        img_path = "src/pad20251/static/img"
        if not os.path.exists(img_path):
            os.makedirs(img_path)
        plt.figure()
        plt.hist(data1, bins=bins, alpha=0.5, label='Data 1')
        plt.hist(data2, bins=bins, alpha=0.5, label='Data 2')
        plt.legend()
        plt.savefig(os.path.join(img_path, f"{filename}.png"))
        plt.close()

# Instancias de las clases
numpy_ops = NumpyOperations()
data_handler = DataHandling()
plotting = Plotting()

# 18. Genera dos sets de datos con distribuciones normales diferentes y muéstralos en el mismo histograma.
data_18_1 = np.random.normal(0, 1, 1000)
data_18_2 = np.random.normal(2, 1.5, 1000)
plotting.overlapping_histograms(data_18_1, data_18_2, 30, "overlapping_histograms_18")

# 19. Experimenta con diferentes valores de bins (por ejemplo, 10, 30, 50) en un histograma y observa cómo cambia la representación.
data_19 = np.random.normal(0, 1, 1000)
plotting.histogram(data_19, 10, "histogram_19_10_bins")
plotting.histogram(data_19, 30, "histogram_19_30_bins")
plotting.histogram(data_19, 50, "histogram_19_50_bins")

# 20. Añade una línea vertical que indique la media de los datos en el histograma.
data_20 = np.random.normal(0, 1, 1000)
mean_20 = np.mean(data_20)
plotting.histogram(data_20, 30, "histogram_20_mean", mean_line=True, mean=mean_20)

# 21. Crea histogramas superpuestos para los dos sets de datos del ejercicio 17, usando colores y transparencias diferentes para distinguirlos.
data_21_1 = np.random.normal(0, 1, 1000)
data_21_2 = np.random.normal(2, 1.5, 1000)
plotting.overlapping_histograms(data_21_1, data_21_2, 30, "overlapping_histograms_21")