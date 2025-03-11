import numpy as np
import os

# 1. Generar un array de NumPy con valores desde 10 hasta 29
array_10_29 = np.arange(10, 30)
print("Array de 10 a 29:", array_10_29)

# 2. Crear un array de 10x10 lleno de unos y calcular su suma
array_ones = np.ones((10, 10))
sum_ones = np.sum(array_ones)
print("Suma de todos los elementos en un array 10x10 de unos:", sum_ones)

# 3. Producto elemento a elemento de dos arrays aleatorios de tamaño 5
array_random_1 = np.random.randint(1, 11, 5)
array_random_2 = np.random.randint(1, 11, 5)
product_arrays = array_random_1 * array_random_2
print("Array aleatorio 1:", array_random_1)
print("Array aleatorio 2:", array_random_2)
print("Producto elemento a elemento:", product_arrays)

# 4. Crear una matriz 4x4 invertible y calcular su inversa
matrix_4x4 = np.array([[4, 7, 2, 3], [3, 6, 1, 2], [2, 5, 3, 1], [1, 4, 2, 6]])
matrix_inverse = np.linalg.inv(matrix_4x4)
print("Matriz 4x4 invertible:", matrix_4x4)
print("Inversa de la matriz:", matrix_inverse)

# 5. Encontrar valores máximo y mínimo en un array de 100 elementos aleatorios
array_100 = np.random.randint(1, 101, 100)
max_value = np.max(array_100)
min_value = np.min(array_100)
max_index = np.argmax(array_100)
min_index = np.argmin(array_100)
print("Array de 100 elementos aleatorios:", array_100)
print("Valor máximo:", max_value, "en índice:", max_index)
print("Valor mínimo:", min_value, "en índice:", min_index)

# Segunda parte de la actividad: Broadcasting e indexado de Arrays

# 6. Crear un array de tamaño 3x1 y otro de 1x3, sumarlos con broadcasting para obtener un 3x3
array_3x1 = np.array([[1], [2], [3]])
array_1x3 = np.array([[4, 5, 6]])
array_broadcasted = array_3x1 + array_1x3
print("Array 3x3 resultante de broadcasting:", array_broadcasted)

# 7. Extraer una submatriz 2x2 desde la segunda fila y columna de una matriz 5x5
matrix_5x5 = np.random.randint(1, 10, (5, 5))
submatrix_2x2 = matrix_5x5[1:3, 1:3]
print("Matriz 5x5:", matrix_5x5)
print("Submatriz 2x2:", submatrix_2x2)

# 8. Crear un array de ceros de tamaño 10 y cambiar valores de índices 3 a 6 a 5
array_zeros = np.zeros(10)
array_zeros[3:7] = 5
print("Array con valores cambiados en rango 3-6:", array_zeros)

# 9. Invertir el orden de las filas en una matriz 3x3
matrix_3x3 = np.random.randint(1, 10, (3, 3))
inverted_matrix = matrix_3x3[::-1]
print("Matriz original:", matrix_3x3)
print("Matriz con filas invertidas:", inverted_matrix)

# 10. Seleccionar y mostrar valores mayores a 0.5 en un array de tamaño 10
array_random_10 = np.random.rand(10)
values_greater_than_05 = array_random_10[array_random_10 > 0.5]
print("Array aleatorio de tamaño 10:", array_random_10)
print("Valores mayores a 0.5:", values_greater_than_05)

# Guardar el archivo en la ruta correspondiente
def guardar_archivo():
    ruta = "src/pad20251/static/actividad_2.py"
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with open(ruta, "w") as file:
        file.write("""import numpy as np\n\n""")
    print(f"Archivo guardado en: {ruta}")

guardar_archivo()
