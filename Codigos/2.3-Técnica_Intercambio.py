# Intercambio de galaxias entre los 3 halos más masivos
import numpy as np # Necesario para manejar arrays numéricos y funciones matemáticas.
import matplotlib.pyplot as plt # Necesario para graficar los resultados.

# Cargar catálogo de datos
dat = np.load('/Users/hakeem/Desktop/Modular/cat_z00.npy', allow_pickle=True).item()

# Parámetro de caja
boxsize = 205 # Tamaño de la caja simulada en Mpc/h, usado para manejar condiciones periódicas.

# Extraer datos relevantes
halo_masas = np.array(dat['halo_mass'])  # masas de halos
halo_pos = np.array(dat['halo_pos']) % boxsize   # posiciones de halos
galaxy_pos = np.array(dat['pos']) % boxsize  # posiciones de galaxias
halo_ids = np.array(dat['cross_sub2halo'])  # índice que asigna cada galaxia a un halo
galaxy_type = np.array(dat['type']) # tipo de galaxia (0 = central, 1 = satélite)

# Seleccionar los 3 halos más masivos
halo_indices = np.argsort(halo_masas)[-3:][::-1]
halo_positions = halo_pos[halo_indices]

# Colores fijos para halos, galaxias centrales y satélites
halo_colors = ['green', 'yellow', 'red']
central_colors = ['cyan', 'orange', 'pink']
satellite_colors = ['black', 'gray', 'purple']

# Almacenar galaxias antes del shuffling
halos_dict = {} # Se crea un diccionario que almacena: Posición del halo/ Posición de la galaxia central/Posiciones relativas de los satélites respecto a su central.
for i, halo_index in enumerate(halo_indices):
    mask_galaxies = (halo_ids == halo_index) # selecciona todas las galaxias dentro del halo.
    g_pos = galaxy_pos[mask_galaxies]
    g_type = galaxy_type[mask_galaxies]

    central = g_pos[g_type == 0][0]  # posición galaxia central
    satellites = g_pos[g_type == 1]  # posición de galaxia satélite 
    # Se calcula la posición relativa de los satélites
    rel_satellites = satellites - central 
    rel_satellites -= boxsize * np.round(rel_satellites / boxsize)

    halos_dict[i] = {'halo_pos': halo_positions[i],
                     'central': central,
                     'satellites': rel_satellites,
                     'halo_color': halo_colors[i],
                     'central_color': central_colors[i],
                     'satellite_color': satellite_colors[i]}

# Aplicación del Shuffling aleatorio: Se genera una permutación aleatoria de los halos para intercambiar sus galaxias entre sí.
indices_shuffle = np.array(list(halos_dict.keys()))
np.random.shuffle(indices_shuffle)

# Configuración inicial del gráfico 
fig, ax = plt.subplots(1, 2, figsize=(14, 7))
ax[0].set_title('Antes del intercambio') # Panel izquierdo: antes del intercambio
ax[1].set_title('Después del intercambio') # Panel derecho: después del intercambio
# Gráico antes del intercambio. Se visualizan las posiciones originales: Halos como círculos grandes/Centrales como círculos medianos/Satélites en posiciones relativas a la central.
for i in halos_dict:
    halo = halos_dict[i]
    # Antes del shuffle
    ax[0].scatter(*halo['halo_pos'][:2], c=halo['halo_color'], s=500, edgecolor='k', label=f'Halo {i+1}')
    ax[0].scatter(*halo['central'][:2], c=halo['central_color'], s=250, edgecolor='k', label=f'Central {i+1}')
    ax[0].scatter(halo['central'][0] + halo['satellites'][:, 0],
                  halo['central'][1] + halo['satellites'][:, 1],
                  c=halo['satellite_color'], s=10)

# Gráfico después del intercambio. Este bloque traslada las galaxias de un halo a otro:
# La nueva central se reubica manteniendo su posición relativa respecto al nuevo halo.
# Lo mismo se aplica a los satélites.
# Se asegura que todas las posiciones estén dentro de la caja simulada (aplicando módulo con boxsize).
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

# Configuración visual: Se ajustan los límites y rejillas de ambos subgráficos.
for axis in ax:
    axis.set_xlim(0, boxsize)
    axis.set_ylim(0, boxsize)
    axis.grid(True, linestyle='--', alpha=0.5)
    axis.legend()
# Se muestra el título y se guarda la imagen con alta resolución.
plt.suptitle("Intercambio de galaxias entre los 3 halos más masivos")
plt.savefig('/Users/hakeem/Desktop/shuffling_3_halos.png', dpi=300, bbox_inches='tight')
plt.show()
