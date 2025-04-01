import pandas as pd
import os

# Datos de los equipos participantes en la Copa Libertadores 2025
equipos = {
    'Grupo A': ['Botafogo', 'Estudiantes de La Plata', 'Universidad de Chile', 'Carabobo'],
    'Grupo B': ['River Plate', 'Independiente del Valle', 'Universitario', 'Barcelona de Guayaquil'],
    'Grupo C': ['Flamengo', 'LDU Quito', 'Deportivo Táchira', 'Central Córdoba'],
    'Grupo D': ['São Paulo', 'Libertad', 'Talleres', 'Alianza Lima'],
    'Grupo E': ['Racing Club', 'Colo-Colo', 'Fortaleza EC', 'Atlético Bucaramanga'],
    'Grupo F': ['Nacional', 'Internacional', 'Atlético Nacional', 'Bahia'],
    'Grupo G': ['Palmeiras', 'Bolívar', 'Sporting Cristal', 'Cerro Porteño'],
    'Grupo H': ['Peñarol', 'Olimpia', 'Velez Sarsfield', 'San Antonio Bulo Bulo']
}

# Diccionario de países para cada equipo
paises = {
    'Botafogo': 'Brasil',
    'Estudiantes de La Plata': 'Argentina',
    'Universidad de Chile': 'Chile',
    'Carabobo': 'Venezuela',
    'River Plate': 'Argentina',
    'Independiente del Valle': 'Ecuador',
    'Universitario': 'Perú',
    'Barcelona de Guayaquil': 'Ecuador',
    'Flamengo': 'Brasil',
    'LDU Quito': 'Ecuador',
    'Deportivo Táchira': 'Venezuela',
    'Central Córdoba': 'Argentina',
    'São Paulo': 'Brasil',
    'Libertad': 'Paraguay',
    'Talleres': 'Argentina',
    'Alianza Lima': 'Perú',
    'Racing Club': 'Argentina',
    'Colo-Colo': 'Chile',
    'Fortaleza EC': 'Brasil',
    'Atlético Bucaramanga': 'Colombia',
    'Nacional': 'Uruguay',
    'Internacional': 'Brasil',
    'Atlético Nacional': 'Colombia',
    'Bahia': 'Brasil',
    'Palmeiras': 'Brasil',
    'Bolívar': 'Bolivia',
    'Sporting Cristal': 'Perú',
    'Cerro Porteño': 'Paraguay',
    'Peñarol': 'Uruguay',
    'Olimpia': 'Paraguay',
    'Velez Sarsfield': 'Argentina',
    'San Antonio Bulo Bulo': 'Bolivia'
}

# Crear conjuntos de equipos por país
equipos_por_pais = {}
for equipo, pais in paises.items():
    if pais not in equipos_por_pais:
        equipos_por_pais[pais] = set()
    equipos_por_pais[pais].add(equipo)

# Equipos campeones de la Copa Libertadores y número de títulos
campeones = {
    'Argentina': {'Independiente': 7, 'Boca Juniors': 6, 'River Plate': 4, 'Estudiantes de La Plata': 4, 'Racing Club': 1, 'Argentinos Juniors': 1, 'Vélez Sarsfield': 1, 'San Lorenzo': 1},
    'Brasil': {'São Paulo': 3, 'Santos': 3, 'Palmeiras': 3, 'Flamengo': 3, 'Grêmio': 3, 'Internacional': 2, 'Cruzeiro': 2, 'Vasco da Gama': 1, 'Corinthians': 1, 'Atlético Mineiro': 1},
    'Uruguay': {'Peñarol': 5, 'Nacional': 3},
    'Paraguay': {'Olimpia': 3},
    'Colombia': {'Atlético Nacional': 2, 'Once Caldas': 1},
    'Chile': {'Colo-Colo': 1},
    'Ecuador': {'LDU Quito': 1},
    'Bolivia': {},
    'Perú': {},
    'Venezuela': {}
}

# Crear conjuntos de equipos campeones por país
equipos_campeones_por_pais = {}
for pais, equipos_campeones in campeones.items():
    equipos_campeones_por_pais[pais] = set(equipos_campeones.keys())

# Crear el directorio 'static' si no existe
if not os.path.exists('static'):
    os.makedirs('static')

# Operaciones de conjuntos y generación de CSVs
def realizar_operaciones_y_guardar(nombre_operacion, conjunto1, conjunto2=None):
    if conjunto2 is None:
        resultado = conjunto1
    elif nombre_operacion == 'interseccion':
        resultado = conjunto1.intersection(conjunto2)
    elif nombre_operacion == 'union':
        resultado = conjunto1.union(conjunto2)
    elif nombre_operacion == 'diferencia':
        resultado = conjunto1.difference(conjunto2)
    elif nombre_operacion == 'diferencia_simetrica':
        resultado = conjunto1.symmetric_difference(conjunto2)
    else:
        print(f"Operación desconocida: {nombre_operacion}")
        return

    print(f"\n{nombre_operacion.capitalize()}: {resultado}")

    # Guardar en CSV
    df_resultado = pd.DataFrame(list(resultado), columns=['Equipo'])
    nombre_archivo_csv = f'static/{nombre_operacion}.csv'
    df_resultado.to_csv(nombre_archivo_csv, index=False)
    print(f"Guardado en: {nombre_archivo_csv}")

# Ejemplo de operaciones
# 1. # Equipos argentinos que participan en la Libertadores 2025 y son campeones
realizar_operaciones_y_guardar(
    'interseccion',
    equipos_por_pais['Argentina'],
    equipos_campeones_por_pais.get('Argentina', set())  # Usar .get() para evitar KeyError
)

# 2. # Equipos brasileños que participan en la Libertadores 2025 y son campeones
realizar_operaciones_y_guardar(
    'interseccion',
    equipos_por_pais['Brasil'],
    equipos_campeones_por_pais.get('Brasil', set())
)

# 3. # Todos los equipos argentinos o brasileños en la Libertadores 2025
realizar_operaciones_y_guardar(
    'union',
    equipos_por_pais['Argentina'],
    equipos_por_pais['Brasil']
)

# 4. # Equipos de la Libertadores 2025 que no son argentinos
todos_los_equipos = set().union(*equipos_por_pais.values())
realizar_operaciones_y_guardar(
    'diferencia',
    todos_los_equipos,
    equipos_por_pais['Argentina']
)

# 5. # Equipos que son campeones de Argentina o Brasil pero no ambos
realizar_operaciones_y_guardar(
    'diferencia_simetrica',
    equipos_campeones_por_pais.get('Argentina', set()),
    equipos_campeones_por_pais.get('Brasil', set())
)


# 6. # Equipos campeones de la Copa Libertadores en los últimos 10 años

# Equipos campeones de la Copa Libertadores en los últimos 10 años
campeones_recientes = {
    2023: 'Fluminense',
    2022: 'Flamengo',
    2021: 'Palmeiras',
    2020: 'Palmeiras',
    2019: 'Flamengo',
    2018: 'River Plate',
    2017: 'Grêmio',
    2016: 'Atlético Nacional',
    2015: 'River Plate'
}

# Diccionario de países para los equipos campeones
paises_campeones = {
    'Fluminense': 'Brasil',
    'Flamengo': 'Brasil',
    'Palmeiras': 'Brasil',
    'River Plate': 'Argentina',
    'Grêmio': 'Brasil',
    'Atlético Nacional': 'Colombia'
}

# Crear conjuntos de equipos campeones por país
equipos_campeones_por_pais = {}
for equipo, pais in paises_campeones.items():
    if pais not in equipos_campeones_por_pais:
        equipos_campeones_por_pais[pais] = set()
    equipos_campeones_por_pais[pais].add(equipo)

# Operación de conjuntos: Campeones que no son ni brasileños ni argentinos
campeones_no_bra_arg = set()
for pais, equipos in equipos_campeones_por_pais.items():
    if pais not in ['Argentina', 'Brasil']:
        campeones_no_bra_arg.update(equipos)

print("Equipos campeones de la Libertadores en los últimos 10 años que no son ni brasileños ni argentinos:", campeones_no_bra_arg)

# Crear DataFrame y guardar en CSV
if not os.path.exists('static'):
    os.makedirs('static')

df_campeones_no_bra_arg = pd.DataFrame(list(campeones_no_bra_arg), columns=['Equipo'])
df_campeones_no_bra_arg.to_csv('static/campeones_no_bra_arg.csv', index=False)