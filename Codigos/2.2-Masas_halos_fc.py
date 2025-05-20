import numpy as np # Necesario para manejar arrays numéricos y funciones matemáticas.
import matplotlib.pyplot as plt  # Necesario para graficar los resultados.
from Corrfunc.theory.xi import xi # Necesario para calcular la función de correlación usando la librería Corrfunc, diseñada para conteo eficiente de pares.

# Ruta al archivo del catálogo de galaxias
archivo = '/Users/hakeem/Desktop/cat_z00.npy'

# Carga el archivo .npy que contiene un diccionario con los datos de la simulacion
# allow_pickle=True permite cargar objetos almacenados con pickle.
# .item() extrae el diccionario de datos del archivo.
dat = np.load(archivo, allow_pickle=True).item()

# Parámetros
boxsize = 205 # Tamaño de la caja simulada en Mpc/h
nbins = 20  # Número de bins para la función de correlación (Número de intervalos en los que se calculará ξ(r))
bin_edges = np.logspace(-1, np.log10(boxsize / 2.1), nbins + 1) 
# Genera bins espaciados logarítmicamente entre 0.1 Mpc/h y  boxsize/2.1
# Se usa logspace porque la función de correlación tiene un comportamiento que se entiende mejor en escala log-log.
nthreads = 4  # Número de hilos para optimizar el cálculo

# Extraer masas y posiciones de los halos
halo_masas = dat['halo_mass']  # Masas de los halos
halo_posiciones = dat['halo_pos']  # Posiciones (x, y, z) de los halos

# Ordenar los halos por masa de menor a mayor
indices_ordenados = np.argsort(halo_masas)

# Seleccionar los halos según su masa
high_mass_indices_10k = indices_ordenados[-10000:]  # 10,000 más masivos
high_mass_indices_20k = indices_ordenados[-20000:]  # 20,000 más masivos
high_mass_indices_30k = indices_ordenados[-30000:]  # 30,000 más masivos

# Obtener posiciones de cada conjunto de halos
high_mass_halos_10k = halo_posiciones[high_mass_indices_10k]
high_mass_halos_20k = halo_posiciones[high_mass_indices_20k]
high_mass_halos_30k = halo_posiciones[high_mass_indices_30k]

# Función auxiliar para calcular ξ(r)
def compute_xi(halo_positions):
    x, y, z = halo_positions.T  # Extraer coordenadas
    return xi(boxsize, nthreads, bin_edges, x, y, z)

# Calcular función de correlación para cada conjunto
results_high_10k = compute_xi(high_mass_halos_10k)
results_high_20k = compute_xi(high_mass_halos_20k)
results_high_5k = compute_xi(high_mass_halos_30k)

# Usar los centros de los bins para graficar
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

# Crear la figura
plt.figure(figsize=(7,5))

# Graficar cada función de correlación
plt.plot(bin_centers, results_high_10k['xi'], marker='s', linestyle='-', color='blue', label='10,000 halos más masivos')
plt.plot(bin_centers, results_high_20k['xi'], marker='^', linestyle='-', color='red', label='20,000 halos más masivos')
plt.plot(bin_centers, results_high_5k['xi'], marker='s', linestyle='-', color='green', label='30000 halos más masivos')

# Escalas logarítmicas
plt.xscale('log')
plt.yscale('log')

# Etiquetas y título
plt.xlabel('Distancia [Mpc/h]')
plt.ylabel(r'$\xi(r)$')
plt.title('Función de correlación de halos según su masa')

# Leyenda y rejilla
plt.legend()
plt.grid()

# Guardar y mostrar
plt.savefig("/Users/hakeem/Desktop/Funcion_correlacion_halos.png", dpi=300, bbox_inches='tight')
plt.show()
