# Importar librerias 
import numpy as np  # operaciones con arrays
import matplotlib.pyplot as plt # para graficar
from Corrfunc.theory.xi import xi # calcula la función de correlación de dos puntos ξ(r).

# Carga el catálogo de simulación, almacenado como diccionario en un archivo .npy.
dat = np.load('/Users/hakeem/Desktop/Modular/cat_z00.npy', allow_pickle=True).item()

# Parámetros del tamaño de la caja de simulación y núcleos de procesamiento para Corrfunc.
boxsize = 205
nthreads = 4

# Extraer propiedades del catálogo: 
halo_masas = np.array(dat['halo_mass'])  # masas de halos.
halo_pos = np.array(dat['halo_pos']) % boxsize  # posiciones de halos.
galaxy_pos = np.array(dat['pos']) % boxsize  # posiciones de galaxias.
halo_ids = np.array(dat['cross_sub2halo'])  # ID del halo al que pertenece cada galaxia.
galaxy_type = np.array(dat['type'])  # tipo de galaxia (0 = central, 1 = satélite).

# Crear bins de masa de halo: Bins log10(halo_mass) de 0.2 dex
log_mass = np.log10(halo_masas)  # Calcula el logaritmo de las masas de halo.
bin_width = 0.1  # Define bins de 0.1 dex para agrupar halos de masas similares.
mass_bins = np.arange(np.min(log_mass), np.max(log_mass) + bin_width, bin_width)
bin_indices = np.digitize(log_mass, mass_bins) # indica en qué bin cae cada halo.

# Construir estructura de halos con sus galaxias
# Para cada halo:
# Se recuperan las galaxias que contiene (por ID).
# Se separa central y satélites.
# Satélites se guardan como posiciones relativas al central (con condiciones de contorno periódicas).
# Se almacena todo en un diccionario por halo.
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

# Shuffling por bin
shuffled_galaxies = []
original_galaxies = []

# Shuffling por bin de masa: Dentro de cada bin de masa: Se mezclan las configuraciones de galaxias entre halos.
# Se mantiene la estructura interna (satélites relativos), pero se asignan a otros halos.
# Así se rompe la correlación entre propiedades de galaxias y ambiente, sin alterar distribución de masa de halo.
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

        shuffled_galaxies.append(new_central)
        shuffled_galaxies.extend(new_sats)

        original_galaxies.append(orig['central'])
        original_galaxies.extend(orig['rel_satellites'] + orig['central'])

original_galaxies = np.array(original_galaxies)
shuffled_galaxies = np.array(shuffled_galaxies)

# Calcular la función de correlación
bin_edges = np.logspace(np.log10(0.1), np.log10(50), 20)
x_o, y_o, z_o = original_galaxies.T
x_s, y_s, z_s = shuffled_galaxies.T
# Se calcula ξ(r) para el catálogo original y el shuffled.
xi_o = xi(boxsize, nthreads, bin_edges, x_o, y_o, z_o)['xi']
xi_s = xi(boxsize, nthreads, bin_edges, x_s, y_s, z_s)['xi']
# Luego se calcula el cociente (ratio), que muestra cuánto cambia la señal de clustering tras eliminar el assembly bias.
ratio = (xi_s + 1e-10) / (xi_o + 1e-10)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

# Plot distribución de galaxias antes y después del shuffling. Muestra cómo cambia la distribución espacial (X vs Y) de las galaxias tras el shuffling.
fig, axs = plt.subplots(1, 2, figsize=(14, 6))
axs[0].scatter(original_galaxies[:, 0], original_galaxies[:, 1], s=0.5, color='blue', alpha=0.4)
axs[0].set_title("Distribución original")
axs[0].set_xlabel("X [Mpc/h]")
axs[0].set_ylabel("Y [Mpc/h]")
axs[0].grid(True, alpha=0.3)

axs[1].scatter(shuffled_galaxies[:, 0], shuffled_galaxies[:, 1], s=0.5, color='red', alpha=0.4)
axs[1].set_title("Después del shuffling")
axs[1].set_xlabel("X [Mpc/h]")
axs[1].set_ylabel("Y [Mpc/h]")
axs[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("/Users/hakeem/Desktop/distribucion_shuffling_completa.png", dpi=300)

# Plot función de correlación: Muestra la caída de la correlación en función de la escala r
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

# Plot ratio: Si el ratio es <1, indica que el clustering disminuyó tras el shuffling → hay assembly bias.
plt.figure(figsize=(8, 6))
plt.semilogx(bin_centers, ratio, marker='o', color='purple')
plt.axhline(1, linestyle='--', color='black')
plt.xlabel('r [Mpc/h]')
plt.ylabel(r'Ratio $\xi_{shuffle}/\xi_{original}$')
plt.title('Ratio de la función de correlación')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
# Se guardan los 3 gráficos: Distribución espacial original vs shuffled / ξ(r) antes y después/ Ratio ξshuffle/ξoriginal
plt.savefig("/Users/hakeem/Desktop/ratio_catalogo_completo.png", dpi=300)

