import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import os

# Cargar el catálogo de datos desde cat_z00.npy
file_path = "/Users/hakeem/Desktop/cat_z00.npy"

if not os.path.exists(file_path):
    raise FileNotFoundError(f"Error: No se encontró el archivo en {file_path}")

data = np.load(file_path, allow_pickle=True).item()

# Extraer las propiedades necesarias del catálogo
if "halo_pos" in data and "pos" in data and "cross_sub2halo" in data and "mst" in data:
    halos_posiciones = np.array(data["halo_pos"])  # Posiciones de los halos
    galaxias_posiciones = np.array(data["pos"])  # Posiciones de las galaxias
    cross_sub2halo = np.array(data["cross_sub2halo"])  # Asociación galaxia-halo
    masa_estelar = np.array(data["mst"])  # Masa estelar de las galaxias

    # Reducir el tamaño de los datos para pruebas rápidas
    galaxias_posiciones = galaxias_posiciones[:10000]
    cross_sub2halo = cross_sub2halo[:10000]
    halos_posiciones = halos_posiciones[:10000]

    print("Datos cargados correctamente. Formas (reducidas):")
    print("Halos posiciones:", halos_posiciones.shape)
    print("Galaxias posiciones:", galaxias_posiciones.shape)
    print("Cross sub2halo:", cross_sub2halo.shape)
    print("Masa estelar:", masa_estelar.shape)
else:
    raise ValueError("El archivo no contiene las claves esperadas. Verifica su estructura.")
# Definir los bins para la función de correlación
bins = np.linspace(0, 10, 10)
caja_tamaño = 50.0  # Tamaño de la caja en Mpc/h

# Aplicar condiciones periódicas
def aplicar_condiciones_periodicas(posiciones, caja_tamaño):
    return posiciones % caja_tamaño

# Función optimizada para calcular la función de correlación de dos puntos usando cKDTree
def calcular_correlacion_2p(posiciones, bins):
    tree = cKDTree(posiciones)
    counts = np.zeros(len(bins) - 1)
    
    for i, r in enumerate(bins[:-1]):
        vecinos = tree.query_ball_point(posiciones, r)  # Encuentra vecinos dentro de un radio `r`
        counts[i] = sum(len(v) - 1 for v in vecinos)  # Cuenta vecinos, excluyendo la galaxia misma
    
    return counts / np.sum(counts)

# Función para aplicar shuffling moviendo galaxias a nuevos halos respetando la posición relativa
def aplicar_shuffling_galaxias(galaxias_posiciones, halos_posiciones, cross_sub2halo):
    galaxias_posiciones_nuevas = np.zeros_like(galaxias_posiciones)
    
    # Filtrar los IDs de halos para que estén dentro del rango permitido
    ids_halos = np.unique(cross_sub2halo)
    ids_halos_validos = ids_halos[ids_halos < len(halos_posiciones)]  # Solo IDs válidos

    # Permutar los halos válidos
    halos_posiciones_permutadas = halos_posiciones[ids_halos_validos].copy()
    np.random.shuffle(halos_posiciones_permutadas)

    for i, halo_idx in enumerate(ids_halos_validos):
        indices_galaxias = np.where(cross_sub2halo == halo_idx)[0]
        galaxias_posiciones_nuevas[indices_galaxias] = (
            galaxias_posiciones[indices_galaxias]
            - halos_posiciones[halo_idx]
            + halos_posiciones_permutadas[i]
        )

    return galaxias_posiciones_nuevas

# Calcular la función de correlación antes del shuffling
print("Calculando función de correlación antes del shuffling...")
correlacion_original = calcular_correlacion_2p(galaxias_posiciones, bins)
print("Correlación Original calculada:", correlacion_original)

# Aplicar shuffling
print("Aplicando shuffling moviendo galaxias a nuevos halos...")
galaxias_posiciones_shuffled = aplicar_shuffling_galaxias(galaxias_posiciones, halos_posiciones, cross_sub2halo)

# Calcular la función de correlación después del shuffling
print("Calculando función de correlación después del shuffling...")
correlacion_shuffled = calcular_correlacion_2p(galaxias_posiciones_shuffled, bins)
print("Correlación Shuffled calculada:", correlacion_shuffled)

# Calcular el sesgo relativo evitando divisiones por cero
bias_relativo = np.sqrt(np.divide(correlacion_original, correlacion_shuffled, out=np.ones_like(correlacion_original), where=correlacion_shuffled!=0))

# Graficar los resultados
plt.figure(figsize=(10, 6))

plt.plot(bins[:-1], correlacion_original, label="Original", color="blue", linestyle="--", linewidth=2, alpha=0.7)
plt.plot(bins[:-1], correlacion_shuffled, label="Shuffled", color="orange", linewidth=2, alpha=0.7)
plt.plot(bins[:-1], bias_relativo, label="Bias Relativo", color="green", linewidth=2)

plt.ylim(0, max(np.max(correlacion_original), np.max(correlacion_shuffled), 1))
plt.xlabel("Separación (Mpc/h)")
plt.ylabel("Correlación")
plt.legend()
plt.title("Galaxy Assembly Bias con cat_z00.npy (Optimizado con cKDTree)")
plt.show()
