import numpy as np
import os
import matplotlib.pyplot as plt 

# 1 generar un array de NumPy con valores desde 10 hasta 29
array_10_29 = np.arange(10, 30)
print("array de 10 a 29:", array_10_29)

# 2 crear un array de 10x10 lleno de unos y calcular su suma
array_ones = np.ones((10, 10))
sum_ones = np.sum(array_ones)
print("suma de todos los elementos en un array 10x10 de unos:", sum_ones)

# 3 producto elemento a elemento de dos arrays aleatorios de tamaño 5
array_random_1 = np.random.randint(1, 11, 5)
array_random_2 = np.random.randint(1, 11, 5)
product_arrays = array_random_1 * array_random_2
print("array aleatorio 1:", array_random_1)
print("array aleatorio 2:", array_random_2)
print("Producto elemento a elemento:", product_arrays)

# 4 crear una matriz 4x4 invertible y calcular su inversa
matrix_4x4 = np.array([[4, 7, 2, 3], [3, 6, 1, 2], [2, 5, 3, 1], [1, 4, 2, 6]])
matrix_inverse = np.linalg.inv(matrix_4x4)
print("matriz 4x4 invertible:", matrix_4x4)
print("inversa de la matriz:", matrix_inverse)

# 5 encontrar valores máximo y mínimo en un array de 100 elementos aleatorios
array_100 = np.random.randint(1, 101, 100)
max_value = np.max(array_100)
min_value = np.min(array_100)
max_index = np.argmax(array_100)
min_index = np.argmin(array_100)
print("array de 100 elementos aleatorios:", array_100)
print("valor máximo:", max_value, "en índice:", max_index)
print("valor mínimo:", min_value, "en índice:", min_index)

# Segunda parte de la actividad: broadcasting e indexado de arrays

# 6 crear un array de tamaño 3x1 y otro de 1x3, sumarlos con broadcasting para obtener un 3x3
array_3x1 = np.array([[1], [2], [3]])
array_1x3 = np.array([[4, 5, 6]])
array_broadcasted = array_3x1 + array_1x3
print("array 3x3 resultante de broadcasting:", array_broadcasted)

# 7 extraer una submatriz 2x2 desde la segunda fila y columna de una matriz 5x5
matrix_5x5 = np.random.randint(1, 10, (5, 5))
submatrix_2x2 = matrix_5x5[1:3, 1:3]
print("matriz 5x5:", matrix_5x5)
print("submatriz 2x2:", submatrix_2x2)

# 8 crear un array de ceros de tamaño 10 y cambiar valores de índices 3 a 6 a 5
array_zeros = np.zeros(10)
array_zeros[3:7] = 5
print("array con valores cambiados en rango 3-6:", array_zeros)

# 9 invertir el orden de las filas en una matriz 3x3
matrix_3x3 = np.random.randint(1, 10, (3, 3))
inverted_matrix = matrix_3x3[::-1]
print("matriz original:", matrix_3x3)
print("matriz con filas invertidas:", inverted_matrix)

# 10 seleccionar y mostrar valores mayores a 0.5 en un array de tamaño 10
array_random_10 = np.random.rand(10)
values_greater_than_05 = array_random_10[array_random_10 > 0.5]
print("array aleatorio de tamaño 10:", array_random_10)
print("valores mayores a 0.5:", values_greater_than_05)

# guardar el archivo en la ruta correspondiente
def guardar_archivo():
    ruta = "src/pad20251/static/actividad_2.py"
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with open(ruta, "w") as file:
        file.write("""import numpy as np\n\n""")
    print(f"archivo guardado en: {ruta}")

guardar_archivo()

# tercera parte de la actividad: gráficos con matplotlib

# 11 gráfico de dispersión de dos arrays de tamaño 100 con números aleatorios
array1 = np.random.rand(100)
array2 = np.random.rand(100)
plt.scatter(array1, array2)
plt.title('gráfico de dispersión de arrays aleatorios')
plt.xlabel('array 1')
plt.ylabel('array 2')
plt.show()

# 12 gráfico de dispersión de y = sin(x) + ruido Gaussiano
x = np.linspace(-2 * np.pi, 2 * np.pi, 100)
y = np.sin(x) + np.random.normal(0, 0.1, x.shape)
plt.scatter(x, y, label='y = sin(x) + ruido Gaussiano')
plt.plot(x, np.sin(x), color='r', label='y = sin(x)')
plt.title('gráfico de dispersión y = sin(x) + ruido Gaussiano')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

# 13 crear cuadricula y gráfico de contorno usando np.meshgrid
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = np.cos(X) + np.sin(Y)
plt.contour(X, Y, Z)
plt.title('gráfico de Contorno z = cos(x) + sin(y)')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# 14 gráfico de dispersión con 1000 puntos aleatorios y densidad ajustada
x = np.random.rand(1000)
y = np.random.rand(1000)
plt.scatter(x, y, c=np.sqrt(x**2 + y**2), cmap='viridis')
plt.colorbar(label='densidad')
plt.title('gráfico de dispersión con densidad ajustada')
plt.xlabel('eje X')
plt.ylabel('eje Y')
plt.show()

# 15 gráfico de contorno lleno
plt.contourf(X, Y, Z, cmap='viridis')
plt.title('gráfico de contorno lleno z = cos(x) + sin(y)')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Generar los arrays
x = np.linspace(-2 * np.pi, 2 * np.pi, 100)
y = np.sin(x) + np.random.normal(0, 0.1, x.shape)

# 16 añadir etiquetas y crear leyendas en el gráfico de dispersión usando código LaTex
plt.scatter(x, y, label=r'$y = \sin(x) + \text{ruido Gaussiano}$')
plt.plot(x, np.sin(x), color='r', label=r'$y = \sin(x)$')
plt.title(r'Gráfico de Dispersión')
plt.xlabel(r'Eje X')
plt.ylabel(r'Eje Y')
plt.legend()
plt.show()

# cuarta parte de la actividad: histogramas
# 16 generar un array de 1000 números aleatorios con distribución normal
data = np.random.normal(0, 1, 1000)

# crear el histograma
plt.hist(data, bins=30, edgecolor='black')
plt.title('Histograma de 1000 Números Aleatorios con Distribución Normal')
plt.xlabel('Valor')
plt.ylabel('Frecuencia')
plt.show()

# 17 generar dos sets de datos con distribuciones normales diferentes
data1 = np.random.normal(0, 1, 1000)
data2 = np.random.normal(5, 1.5, 1000)

# crear el histograma
plt.hist(data1, bins=30, alpha=0.5, label='data1', edgecolor='black')
plt.hist(data2, bins=30, alpha=0.5, label='data2', edgecolor='black')
plt.title('Histogramas de dos sets de datos con distribuciones normales diferentes')
plt.xlabel('Valor')
plt.ylabel('Frecuencia')
plt.legend()
plt.show()

# 18 crear histogramas con diferentes valores de bins
bins_values = [10, 30, 50]
data = np.random.normal(0, 1, 1000)

for bins in bins_values:
    plt.hist(data, bins=bins, edgecolor='black')
    plt.title(f'Histograma con {bins} bins')
    plt.xlabel('Valor')
    plt.ylabel('Frecuencia')
    plt.show()

# 19 añadir una línea vertical para la media 
data = np.random.normal(0, 1, 1000)
mean_value = np.mean(data)

plt.hist(data, bins=30, edgecolor='black')
plt.axvline(mean_value, color='r', linestyle='dashed', linewidth=1, label=f'Media: {mean_value:.2f}')
plt.title('Histograma con Línea de la Media')
plt.xlabel('Valor')
plt.ylabel('Frecuencia')
plt.legend()
plt.show()

# Generar dos sets de datos con distribuciones normales diferentes
data1 = np.random.normal(0, 1, 1000)
data2 = np.random.normal(5, 1.5, 1000)

# 20 crear histogramas superpuestos usando colores diferentes
plt.hist(data1, bins=30, alpha=0.5, label='data1', edgecolor='black')
plt.hist(data2, bins=30, alpha=0.5, label='data2', edgecolor='black', color='orange')
plt.title('Histogramas Superpuestos de dos sets de datos con distribuciones normales diferentes')
plt.xlabel('Valor')
plt.ylabel('Frecuencia')
plt.legend()
plt.show()
