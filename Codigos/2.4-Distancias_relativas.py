# Importar librerias
import numpy as np # Se usa extensivamente para manejar arrays, hacer logaritmos, máscaras
import matplotlib.pyplot as plt # se usa para crear gráficos.
from matplotlib.ticker import LogLocator # se usa explícitamente para configurar los ejes logarítmicos menores.

# Ruta al archivo del catálogo
archivo = '/Users/hakeem/Desktop/Modular/cat_z00.npy'

# Cargar los datos: contiene un diccionario con propiedades de galaxias y halos.
dat = np.load(archivo, allow_pickle=True).item()

# Parámetros de la caja
boxsize = 205  # Tamaño de la caja simulada Mpc/h.
nbins = 20  # Número de bins
nthreads = 4  # Número de hilos
max_dist = (boxsize * np.sqrt(3)) / 2  # máxima distancia entre dos puntos dentro de una caja cúbica periódica → diagonal del cubo dividida entre 2.
print("Máxima distancia permitida:", max_dist)

# 🔹 Extraer datos relevantes
halo_masas = 10**np.array(dat['halo_mass'])
halo_pos = np.array(dat['halo_pos'])
galaxy_pos = np.array(dat['pos'])
halo_ids = np.array(dat['cross_sub2halo'])

# 🔹 Filtrar IDs inválidos: Elimina galaxias con halos inexistentes o fuera del rango del arreglo.
valid_mask = (halo_ids >= 0) & (halo_ids < len(halo_pos))
halo_ids = halo_ids[valid_mask]
galaxy_pos = galaxy_pos[valid_mask]

# 🔹 Obtener la posición del centro del halo para cada galaxia
galaxy_halo_pos = halo_pos[halo_ids]

# 🔹 Aplicar condiciones periódicas correctamente
delta_pos = galaxy_pos - galaxy_halo_pos
delta_pos = delta_pos - boxsize * np.round(delta_pos / boxsize)
distancias_relativas = np.sqrt(np.sum(delta_pos**2, axis=1))

# 🔹 Comprobar si hay puntos que excedan la distancia máxima permitida
num_puntos_fuera = np.sum(distancias_relativas > max_dist)
print("Número de puntos por encima del límite:", num_puntos_fuera)

# 🔹 Definir los bines de masa de los halos correctamente
mass_bins = np.logspace(np.log10(np.min(halo_masas)), np.log10(np.max(halo_masas)), num=4)
bin_indices = np.digitize(halo_masas[halo_ids], bins=mass_bins) - 1
bin_indices = np.clip(bin_indices, 0, 2)

# 🔹 Scatter plot corregido:
# Eje X: masa del halo.
# Eje Y: distancia de la galaxia al centro de su halo.
# Color: indica el bin de masa.
# Se añade una línea horizontal que marca el límite máximo teórico de distancia.
plt.figure(figsize=(8,6))
scatter = plt.scatter(halo_masas[halo_ids], distancias_relativas, c=bin_indices, cmap="coolwarm", alpha=0.5, s=1)
plt.axhline(max_dist, color='green', linestyle='--', label=f'Máxima distancia permitida ({max_dist:.2f})')

plt.xscale("log")
plt.yscale("log")

plt.xlabel("Masa del halo [$M_{\odot}/h$]")
plt.ylabel("Distancia relativa [Mpc/h]")
plt.title("Distancia de galaxias al centro del halo vs Masa del halo")

# 🔹 Barra de colores
cbar = plt.colorbar(scatter)
cbar.set_label("Bin de masa del halo")

# 🔹 Ajustar subdivisiones menores
locator = LogLocator(base=10.0, subs='auto', numticks=10)
plt.gca().xaxis.set_minor_locator(locator)
plt.gca().yaxis.set_minor_locator(locator)
plt.grid(True, which="both", linestyle="--", alpha=0.5)
# Se muestra el título y se guarda la imagen.
plt.legend()
plt.savefig("/Users/hakeem/Desktop/Scatter_Distancias_Galaxia_Halo.png", dpi=300, bbox_inches='tight')
plt.show()
