import numpy as np # Necesario para manipulación de arrays, operaciones matemáticas.
import matplotlib.pyplot as plt # Necesario para la creación de gráficos 2D.
from Corrfunc.theory.xi import xi  # Sirve en el cálculo eficiente de la función de correlación en cajas periódicas.

# Cargar catálogo de simulación: contiene un diccionario con propiedades de galaxias y halos.
dat = np.load('/Users/hakeem/Desktop/Modular/cat_z00.npy', allow_pickle=True).item()

# Definición de parámetros básicos.
boxsize = 205 # Tamaño de la caja periÓdica en Mpc/h
nthreads = 4.  # Número de hilos para Corrfunc

# Extraer datos: Extrae la información necesaria del catálogo y aplica condiciones periódicas
halo_masas = np.array(dat['halo_mass'])  # Masas de halos
halo_pos = np.array(dat['halo_pos']) % boxsize  # Posiciones de halos 
galaxy_pos = np.array(dat['pos']) % boxsize     # Posiciones de galaxias
halo_ids = np.array(dat['cross_sub2halo'])   # Índices de halo para cada galaxia
galaxy_type = np.array(dat['type'])         # 0 = central, 1 = satélite

# Definición de bins en masa de halo.
log_mass = np.log10(halo_masas)  # Convierte a escala logarítmica si fuera necesario.
bin_width = 0.1                 # Crea bins de 0.1 dex de ancho.
mass_bins = np.arange(np.min(log_mass), np.max(log_mass) + bin_width, bin_width)
bin_indices = np.digitize(log_mass, mass_bins)  # Asigna a cada halo el índice de bin correspondiente.

# Construir estructura de halos con galaxias: Agrupa galaxias por halo.
# Identifica la galaxia central (tipo 0) y las satélites (tipo 1).
# Calcula las posiciones relativas de los satélites respecto a la central, ajustando por periodicidad.
# Almacena todo en una lista de diccionarios por halo.
halos = []
for i in range(len(halo_pos)):
    mask = (halo_ids == i)
    gal_pos = galaxy_pos[mask]
    gal_type = galaxy_type[mask]

    if np.sum(gal_type == 0) == 0:
        continue

    central = gal_pos[gal_type == 0][0]
    satellites = gal_pos[gal_type == 1]

    rel_sat = satellites - central
    rel_sat -= boxsize * np.round(rel_sat / boxsize)

    halos.append({
        "id": i,
        "bin": bin_indices[i],
        "halo_pos": halo_pos[i],
        "central": central,
        "rel_satellites": rel_sat
    })

shuffled_galaxies = []
original_galaxies = []
# Aplicar el Shuffling por bin
# Dentro de cada bin de masa de halo.
# Se permutan las galaxias centrales y satélites entre halos del mismo bin.
# Se preserva la estructura interna (posiciones relativas).
for bin_num in np.unique(bin_indices):
    halos_bin = [h for h in halos if h["bin"] == bin_num]
    if len(halos_bin) < 2:
        continue

    indices = np.arange(len(halos_bin))
    np.random.shuffle(indices)

    for orig, idx in zip(halos_bin, indices):
        shuf = halos_bin[idx]

        new_central = orig['halo_pos'] + (shuf['central'] - shuf['halo_pos'])
        new_central %= boxsize
        new_sats = new_central + shuf['rel_satellites']
        new_sats %= boxsize
# Se generan dos nuevas listas de posiciones.
        shuffled_galaxies.append(new_central) # galaxias tras el intercambio.
        shuffled_galaxies.extend(new_sats)

        original_galaxies.append(orig['central']) # galaxias sin modificar.
        original_galaxies.extend(orig['rel_satellites'] + orig['central'])
# Conversión a arrays y separación de coordenadas
original_galaxies = np.array(original_galaxies) 
shuffled_galaxies = np.array(shuffled_galaxies)  

# Cálculo de funciones de correlación
bin_edges = np.logspace(np.log10(0.1), np.log10(50), 20)
x_o, y_o, z_o = original_galaxies.T
x_s, y_s, z_s = shuffled_galaxies.T

xi_o = xi(boxsize, nthreads, bin_edges, x_o, y_o, z_o)['xi']  # Calcula función de correlación para el catálogo original.
xi_s = xi(boxsize, nthreads, bin_edges, x_s, y_s, z_s)['xi']  # Calcula función de correlación para el catálogo modificado.
ratio = (xi_s + 1e-10) / (xi_o + 1e-10)  # calcula el cociente. Añade una pequeña constante para evitar divisiones por cero.
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

# Plot función de correlación: Se genera un gráfico log-log de la función de corelación antes y después del shuffling.
plt.figure(figsize=(8, 6))
plt.loglog(bin_centers, xi_o, marker='o', label='Antes del shuffling')
plt.loglog(bin_centers, xi_s, marker='s', label='Después del shuffling')
plt.xlabel('r [Mpc/h]')
plt.ylabel(r'$\xi(r)$')
plt.title('Función de correlación')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("/Users/hakeem/Desktop/correlacion_catalogo_completo.png", dpi=300)
