import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Genera un array de NumPy con valores desde 10 hasta 29.
array_10_29 = np.arange(10, 30)

# 2. Calcula la suma de todos los elementos en un array de NumPy de tamaño 10x10, lleno de unos.
array_ones_10x10 = np.ones((10, 10))
sum_ones_10x10 = np.sum(array_ones_10x10)

# 3. Dados dos arrays de tamaño 5, llenos de números aleatorios desde 1 hasta 10, realiza un producto elemento a elemento.
array_1 = np.random.randint(1, 11, 5)
array_2 = np.random.randint(1, 11, 5)
product_elements = array_1 * array_2

# 4. Crea una matriz de 4x4, donde cada elemento es igual a i+j (con i y j siendo el índice de fila y columna, respectivamente)
# Intentamos crear una matriz que no sea singular cambiando su construcción a una diagonal no singular.
matrix_4x4 = np.fromfunction(lambda i, j: i + j, (4, 4))
matrix_4x4 += np.eye(4) * 5  # Le agregamos una matriz identidad multiplicada por 5 para hacerla no singular
try:
    matrix_inverse = np.linalg.inv(matrix_4x4)
except np.linalg.LinAlgError:
    matrix_inverse = "La matriz es singular y no tiene inversa."

# 5. Encuentra los valores máximo y mínimo en un array de 100 elementos aleatorios y muestra sus índices.
array_100 = np.random.randint(1, 101, 100)
max_value = np.max(array_100)
min_value = np.min(array_100)
max_index = np.argmax(array_100)
min_index = np.argmin(array_100)

# Guardar los resultados en un archivo Excel
with pd.ExcelWriter('resultados_actividad.xlsx') as writer:
    # 1. Array de 10 a 29
    df_10_29 = pd.DataFrame(array_10_29, columns=["Valores de 10 a 29"])
    df_10_29.to_excel(writer, sheet_name="Punto 1", index=False)
    
    # 2. Suma de los elementos del array de unos 10x10
    df_sum_ones = pd.DataFrame([sum_ones_10x10], columns=["Suma de los elementos del array 10x10"])
    df_sum_ones.to_excel(writer, sheet_name="Punto 2", index=False)
    
    # 3. Producto elemento a elemento
    df_product = pd.DataFrame({"Array 1": array_1, "Array 2": array_2, "Producto elemento a elemento": product_elements})
    df_product.to_excel(writer, sheet_name="Punto 3", index=False)
    
    # 4. Matriz 4x4 e inversa
    df_matrix_4x4 = pd.DataFrame(matrix_4x4)
    df_matrix_inverse = pd.DataFrame(matrix_inverse)
    df_matrix_4x4.to_excel(writer, sheet_name="Punto 4 - Matriz 4x4", index=False)
    df_matrix_inverse.to_excel(writer, sheet_name="Punto 4 - Inversa", index=False)
    
    # 5. Máximo y mínimo
    df_max_min = pd.DataFrame({
        "Valor máximo": [max_value], "Índice máximo": [max_index],
        "Valor mínimo": [min_value], "Índice mínimo": [min_index]
    })
    df_max_min.to_excel(writer, sheet_name="Punto 5", index=False)

print("Archivo 'resultados_actividad.xlsx' generado con éxito.")

# 6. Broadcasting de arrays
array_3x1 = np.array([[1], [2], [3]])  # 3x1
array_1x3 = np.array([4, 5, 6])  # 1x3
broadcasted_sum = array_3x1 + array_1x3  # Resultado de la suma usando broadcasting

# 7. Submatriz 2x2 de una matriz 5x5
matrix_5x5 = np.random.randint(1, 10, (5, 5))  # Generamos una matriz 5x5 con valores aleatorios
submatrix_2x2 = matrix_5x5[1:3, 1:3]  # Extraemos la submatriz 2x2 que comienza en la segunda fila y columna

# 8. Indexado de un array de ceros
zeros_array = np.zeros(10)
zeros_array[3:7] = 5  # Cambiamos los valores del índice 3 al 6 a 5

# 9. Inversión del orden de las filas en una matriz 3x3
matrix_3x3 = np.random.randint(1, 10, (3, 3))  # Generamos una matriz 3x3 con valores aleatorios
matrix_3x3_reversed = matrix_3x3[::-1]  # Invertimos el orden de las filas

# 10. Filtrar números mayores a 0.5 en un array aleatorio
random_array = np.random.rand(10)  # Generamos un array de números aleatorios
filtered_array = random_array[random_array > 0.5]  # Seleccionamos los valores mayores a 0.5

# Guardar los resultados en un nuevo archivo Excel
filename = 'resultados_actividad_6_a_10.xlsx'

# Crear un DataFrame para cada resultado y escribirlo en una hoja separada del archivo Excel
with pd.ExcelWriter(filename, engine='openpyxl') as writer:
    pd.DataFrame(broadcasted_sum).to_excel(writer, sheet_name='Broadcasting', index=False, header=False)
    pd.DataFrame(submatrix_2x2).to_excel(writer, sheet_name='Submatriz 2x2', index=False, header=False)
    pd.DataFrame(zeros_array).to_excel(writer, sheet_name='Array Ceros', index=False, header=False)
    pd.DataFrame(matrix_3x3_reversed).to_excel(writer, sheet_name='Matriz Invertida', index=False, header=False)
    pd.DataFrame(filtered_array).to_excel(writer, sheet_name='Filtrado > 0.5', index=False, header=False)

print("Los resultados se han guardado en 'resultados_actividad_6_a_10.xlsx'.")

# 11. Generar dos arrays de tamaño 100 con números aleatorios y crear un gráfico de dispersión
x_rand = np.random.rand(100)
y_rand = np.random.rand(100)

# Gráfico de dispersión
plt.figure(figsize=(6, 6))
plt.scatter(x_rand, y_rand)
plt.title('Gráfico de Dispersión (Arrays Aleatorios)')
plt.xlabel('X')
plt.ylabel('Y')
scatter_filename = 'scatter_random.png'
plt.savefig(scatter_filename)
plt.close()

# 12. Generar un gráfico de dispersión para x entre -2pi y 2pi, y = sin(x) + ruido gaussiano
x = np.linspace(-2*np.pi, 2*np.pi, 100)
y = np.sin(x) + np.random.normal(0, 0.1, size=x.shape)

# Graficar
plt.figure(figsize=(6, 6))
plt.scatter(x, y, label=r'$y = \sin(x) + \text{Ruido Gaussiano}$')
plt.plot(x, np.sin(x), label=r'$y = \sin(x)$', color='red')
plt.title('Gráfico de Dispersión con Ruido Gaussiano')
plt.xlabel(r'$Eje X$')
plt.ylabel(r'$Eje Y$')
plt.legend(loc='best', fontsize=10, frameon=False, labels=[r'$\sin(x) + \text{Ruido}$', r'$\sin(x)$'])
scatter_sin_filename = 'scatter_sin_noise.png'
plt.savefig(scatter_sin_filename)
plt.close()

# 13. Crear una cuadrícula con np.meshgrid y un gráfico de contorno para z = cos(x) + sin(y)
x_vals = np.linspace(-5, 5, 100)
y_vals = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z = np.cos(X) + np.sin(Y)

# Gráfico de contorno
plt.figure(figsize=(6, 6))
plt.contour(X, Y, Z, cmap='viridis')
plt.title(r'Gráfico de Contorno para $z = \cos(x) + \sin(y)$')
contour_filename = 'contour_plot.png'
plt.savefig(contour_filename)
plt.close()

# 14. Crear un gráfico de dispersión con 1000 puntos aleatorios y usar la densidad para el color
x_points = np.random.rand(1000)
y_points = np.random.rand(1000)

plt.figure(figsize=(6, 6))
plt.hexbin(x_points, y_points, gridsize=30, cmap='inferno')
plt.colorbar(label='Densidad')
plt.title('Gráfico de Dispersión con Densidad de Puntos')
density_filename = 'scatter_density.png'
plt.savefig(density_filename)
plt.close()

# 15. Generar un gráfico de contorno lleno basado en el gráfico de dispersión de la función sin(x) con ruido
plt.figure(figsize=(6, 6))
plt.contourf(X, Y, Z, cmap='coolwarm')
plt.title(r'Gráfico de Contorno Lleno para $z = \cos(x) + \sin(y)$')
filled_contour_filename = 'filled_contour_plot.png'
plt.savefig(filled_contour_filename)
plt.close()

# 16. Gráfico de dispersión con etiquetas LaTeX para los ejes y el título
plt.figure(figsize=(6, 6))
plt.scatter(x, y, label=r'$y = \sin(x) + \text{Ruido Gaussiano}$')
plt.plot(x, np.sin(x), label=r'$y = \sin(x)$', color='red')
plt.title(r'$Gráfico\ de\ Dispersión$')
plt.xlabel(r'$Eje\ X$')
plt.ylabel(r'$Eje\ Y$')
plt.legend(loc='best', fontsize=10, frameon=False, labels=[r'$\sin(x) + \text{Ruido}$', r'$\sin(x)$'])
labeled_scatter_filename = 'scatter_labeled.png'
plt.savefig(labeled_scatter_filename)
plt.close()

# Guardar los resultados en un nuevo archivo Excel
filename = 'resultados_actividad_11_a_16.xlsx'

# Guardar los resultados en un nuevo archivo Excel
filename = 'resultados_actividad_11_a_16.xlsx'

with pd.ExcelWriter(filename, engine='openpyxl') as writer:
    # Guardar los arrays generados en el archivo Excel
    pd.DataFrame({'x_random': x_rand, 'y_random': y_rand}).to_excel(writer, sheet_name='Dispersión Aleatoria', index=False)
    pd.DataFrame({'x': x, 'y': y}).to_excel(writer, sheet_name='Dispersión con Ruido', index=False)

    # Corregido: Crear un DataFrame con todos los puntos de la cuadrícula
    data = {'x': X.flatten(), 'y': Y.flatten(), 'z': Z.flatten()}
    df_contour = pd.DataFrame(data)
    df_contour.to_excel(writer, sheet_name='Contorno Z', index=False)

    pd.DataFrame({'x': x_points, 'y': y_points}).to_excel(writer, sheet_name='Dispersión Densidad', index=False)

print("Los resultados y los gráficos se han guardado en los archivos correspondientes.")

# 17. Crea un histograma a partir de un array de 1000 números aleatorios generados con una distribución normal.
data1 = np.random.normal(0, 1, 1000)  # Distribución normal, media 0 y desviación estándar 1

# Crear histograma y guardar como archivo PNG
plt.figure(figsize=(8, 6))
plt.hist(data1, bins=30, alpha=0.7, color='blue', edgecolor='black')
plt.title('Histograma de Distribución Normal')
plt.xlabel('Valores')
plt.ylabel('Frecuencia')
plt.savefig('histograma_normal.png')  # Guardar como archivo PNG
plt.show()

# 18. Genera dos sets de datos con distribuciones normales diferentes y muéstralos en el mismo histograma.
data2 = np.random.normal(2, 1.5, 1000)  # Otra distribución normal con media 2 y desviación estándar 1.5

# Crear histograma con ambos sets de datos y guardar como PDF
plt.figure(figsize=(8, 6))
plt.hist(data1, bins=30, alpha=0.5, color='blue', edgecolor='black', label='Distribución Normal 1')
plt.hist(data2, bins=30, alpha=0.5, color='red', edgecolor='black', label='Distribución Normal 2')
plt.title('Histograma de dos Distribuciones Normales')
plt.xlabel('Valores')
plt.ylabel('Frecuencia')
plt.legend()
plt.savefig('histograma_dos_distribuciones.pdf')  # Guardar como archivo PDF
plt.show()

# 19. Experimenta con diferentes valores de bins (por ejemplo, 10, 30, 50) en un histograma y observa cómo cambia la representación.
plt.figure(figsize=(8, 6))
plt.hist(data1, bins=10, alpha=0.5, color='green', edgecolor='black', label='10 Bins')
plt.hist(data1, bins=30, alpha=0.5, color='blue', edgecolor='black', label='30 Bins')
plt.hist(data1, bins=50, alpha=0.5, color='orange', edgecolor='black', label='50 Bins')
plt.title('Histograma con Diferentes Valores de Bins')
plt.xlabel('Valores')
plt.ylabel('Frecuencia')
plt.legend()
plt.savefig('histograma_bins.png')  # Guardar como archivo PNG
plt.show()

# 20. Añade una línea vertical que indique la media de los datos en el histograma.
mean_data1 = np.mean(data1)
plt.figure(figsize=(8, 6))
plt.hist(data1, bins=30, alpha=0.7, color='purple', edgecolor='black')
plt.axvline(mean_data1, color='red', linestyle='dashed', linewidth=2, label=f'Media: {mean_data1:.2f}')
plt.title('Histograma con Línea de Media')
plt.xlabel('Valores')
plt.ylabel('Frecuencia')
plt.legend()
plt.savefig('histograma_media.pdf')  # Guardar como archivo PDF
plt.show()

# 21. Crea histogramas superpuestos para los dos sets de datos del ejercicio 17, usando colores y transparencias diferentes para distinguirlos.
plt.figure(figsize=(8, 6))
plt.hist(data1, bins=30, alpha=0.5, color='blue', edgecolor='black', label='Distribución Normal 1')
plt.hist(data2, bins=30, alpha=0.5, color='red', edgecolor='black', label='Distribución Normal 2')
plt.title('Histogramas Superpuestos')
plt.xlabel('Valores')
plt.ylabel('Frecuencia')
plt.legend()
plt.savefig('histogramas_superpuestos.png')  # Guardar como archivo PNG
plt.show()
