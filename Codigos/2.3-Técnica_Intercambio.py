# Intercambio de galaxias entre los 3 halos más masivos

import numpy as np
import matplotlib.pyplot as plt

# Cargar catálogo de datos
dat = np.load('/Users/hakeem/Desktop/Modular/cat_z00.npy', allow_pickle=True).item()

# Parámetro de caja
boxsize = 205

# Extraer datos relevantes
halo_masas = np.array(dat['halo_mass'])
halo_pos = np.array(dat['halo_pos']) % boxsize
galaxy_pos = np.array(dat['pos']) % boxsize
halo_ids = np.array(dat['cross_sub2halo'])
galaxy_type = np.array(dat['type'])

# Seleccionar los 3 halos más masivos
halo_indices = np.argsort(halo_masas)[-3:][::-1]
halo_positions = halo_pos[halo_indices]

# Colores fijos
halo_colors = ['green', 'yellow', 'red']
central_colors = ['cyan', 'orange', 'pink']
satellite_colors = ['black', 'gray', 'purple']

# Almacenar galaxias antes del shuffling
halos_dict = {}
for i, halo_index in enumerate(halo_indices):
    mask_galaxies = (halo_ids == halo_index)
    g_pos = galaxy_pos[mask_galaxies]
    g_type = galaxy_type[mask_galaxies]

    central = g_pos[g_type == 0][0]  # posición galaxia central
    satellites = g_pos[g_type == 1]
    # posición relativa satélites
    rel_satellites = satellites - central
    rel_satellites -= boxsize * np.round(rel_satellites / boxsize)

    halos_dict[i] = {'halo_pos': halo_positions[i],
                     'central': central,
                     'satellites': rel_satellites,
                     'halo_color': halo_colors[i],
                     'central_color': central_colors[i],
                     'satellite_color': satellite_colors[i]}

# Shuffling aleatorio
indices_shuffle = np.array(list(halos_dict.keys()))
np.random.shuffle(indices_shuffle)

# Parámetros gráficos
fig, ax = plt.subplots(1, 2, figsize=(14, 7))
ax[0].set_title('Antes del intercambio')
ax[1].set_title('Después del intercambio')

for i in halos_dict:
    halo = halos_dict[i]
    # Antes del shuffle
    ax[0].scatter(*halo['halo_pos'][:2], c=halo['halo_color'], s=500, edgecolor='k', label=f'Halo {i+1}')
    ax[0].scatter(*halo['central'][:2], c=halo['central_color'], s=250, edgecolor='k', label=f'Central {i+1}')
    ax[0].scatter(halo['central'][0] + halo['satellites'][:, 0],
                  halo['central'][1] + halo['satellites'][:, 1],
                  c=halo['satellite_color'], s=10)

# Después del shuffle
for i, idx_shuffle in zip(halos_dict, indices_shuffle):
    original = halos_dict[i]
    shuffled = halos_dict[idx_shuffle]

    new_central = original['halo_pos'] + (shuffled['central'] - shuffled['halo_pos'])
    new_central %= boxsize

    new_satellites = new_central + shuffled['satellites']
    new_satellites %= boxsize

    ax[1].scatter(*original['halo_pos'][:2], c=original['halo_color'], s=500, edgecolor='k', label=f'Halo {i+1}')
    ax[1].scatter(*new_central[:2], c=shuffled['central_color'], s=250, edgecolor='k', label=f'Central {i+1}')
    ax[1].scatter(new_satellites[:, 0], new_satellites[:, 1], c=shuffled['satellite_color'], s=10)

# Configuración visual
for axis in ax:
    axis.set_xlim(0, boxsize)
    axis.set_ylim(0, boxsize)
    axis.grid(True, linestyle='--', alpha=0.5)
    axis.legend()

plt.suptitle("Intercambio de galaxias entre los 3 halos más masivos")
plt.savefig('/Users/hakeem/Desktop/shuffling_3_halos.png', dpi=300, bbox_inches='tight')
plt.show()
