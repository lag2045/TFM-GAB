import numpy as np
import matplotlib.pyplot as plt
from Corrfunc.theory.xi import xi

# Cargar catálogo
dat = np.load('/Users/hakeem/Desktop/Modular/cat_z00.npy', allow_pickle=True).item()

# Parámetros
boxsize = 205
nthreads = 4

# Extraer datos
halo_masas = np.array(dat['halo_mass'])
halo_pos = np.array(dat['halo_pos']) % boxsize
galaxy_pos = np.array(dat['pos']) % boxsize
halo_ids = np.array(dat['cross_sub2halo'])
galaxy_type = np.array(dat['type'])

# Bins log10(halo_mass) de 0.1 dex
log_mass = np.log10(halo_masas)
bin_width = 0.1
mass_bins = np.arange(np.min(log_mass), np.max(log_mass) + bin_width, bin_width)
bin_indices = np.digitize(log_mass, mass_bins)

# Construir estructura de halos con galaxias
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

# Correlación
bin_edges = np.logspace(np.log10(0.1), np.log10(50), 20)
x_o, y_o, z_o = original_galaxies.T
x_s, y_s, z_s = shuffled_galaxies.T

xi_o = xi(boxsize, nthreads, bin_edges, x_o, y_o, z_o)['xi']
xi_s = xi(boxsize, nthreads, bin_edges, x_s, y_s, z_s)['xi']
ratio = (xi_s + 1e-10) / (xi_o + 1e-10)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

# Plot función de correlación
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
