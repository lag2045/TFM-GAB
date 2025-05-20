import numpy as np  # Necesario para manejar arrays numéricos y funciones matemáticas.
from Corrfunc.theory.xi import xi   # Necesario para calcular la función de correlación usando la librería Corrfunc, diseñada para conteo eficiente de pares.
import matplotlib.pyplot as plt  # Necesario para graficar los resultados.

# Ruta al archivo del catálogo de simulación 
archivo = '/Users/hakeem/Desktop/cat_z00.npy'

# Cargar los datos
dat = np.load(archivo, allow_pickle=True).item()

# Parámetros
boxsize = 205 # Tamaño de la caja simulada
nbins = 20  # Número de bins
bin_edges = np.logspace(-1, np.log10(boxsize / 2.1), nbins + 1)  # Ajustar rmax
nthreads = 4  # Número de hilos de CPU usados por Corrfunc

# Definir cortes y colores específicos
cortes = [9, 9.5, 10]
colores = ['blue', 'red', 'green']  # Azul, rojo, verde

# Iterar sobre los cortes de masa estelar
for corte, color in zip(cortes, colores):
    # Filtrar galaxias
    mstar = dat['mst']  # Masa estelar
    pos = dat['pos']  # Posiciones
    galaxias_filtradas = pos[mstar > corte] # Se seleccionan las galaxias con log10(M*) > corte.
    print(f"Galaxias con log10(Mstar) > {corte}: {len(galaxias_filtradas)}")

    # Extraer posiciones. Se transponen las coordenadas y se separan en los tres ejes espaciales X, Y y Z para pasarlos a Corrfunc.
    x, y, z = galaxias_filtradas.T

    # Calcular la función de correlación para las posiciones filtradas de galaxias.
    try:
        results = xi(boxsize, nthreads, bin_edges, x, y, z)
        print(f"Resultados para log10(Mstar) > {corte}:")
        print("ravg:", results['ravg']) # Distancia media de cada bin.
        print("xi:", results['xi']) # Valor de la función de correlación en ese bin.

        # Usar centros de los bins para graficar
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        plt.plot(bin_centers, results['xi'], marker='o', linestyle='-', color=color, label=f'log10(Mstar) > {corte}')
    except Exception as e:
        print(f"Error en Corrfunc para log10(Mstar) > {corte}: {e}")

# Se configuran los ejes en escala logarítmica para mejor visualización, luego se añaden etiquetas, título, leyenda y rejilla.
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Distancia [Mpc/h]')
plt.ylabel(r'$\xi(r)$')
plt.title('Función de correlación en muestras de galaxias')
plt.legend()
plt.grid()
plt.show()
