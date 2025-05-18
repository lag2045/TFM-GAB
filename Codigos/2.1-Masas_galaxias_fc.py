import numpy as np
from Corrfunc.theory.xi import xi
import matplotlib.pyplot as plt

# Ruta al archivo del catálogo
archivo = '/Users/hakeem/Desktop/cat_z00.npy'

# Cargar los datos
dat = np.load(archivo, allow_pickle=True).item()

# Parámetros
boxsize = 205 # Tamaño de la caja simulada
nbins = 20  # Número de bins
bin_edges = np.logspace(-1, np.log10(boxsize / 2.1), nbins + 1)  # Ajustar rmax
nthreads = 4  # Número de hilos

# Definir cortes y colores específicos
cortes = [9, 9.5, 10]
colores = ['blue', 'red', 'green']  # Azul, rojo, verde

# Iterar sobre los cortes de masa estelar
for corte, color in zip(cortes, colores):
    # Filtrar galaxias
    mstar = dat['mst']  # Masa estelar
    pos = dat['pos']  # Posiciones
    galaxias_filtradas = pos[mstar > corte]
    print(f"Galaxias con log10(Mstar) > {corte}: {len(galaxias_filtradas)}")

    # Extraer posiciones
    x, y, z = galaxias_filtradas.T

    # Calcular la función de correlación
    try:
        results = xi(boxsize, nthreads, bin_edges, x, y, z)
        print(f"Resultados para log10(Mstar) > {corte}:")
        print("ravg:", results['ravg'])
        print("xi:", results['xi'])

        # Usar centros de los bins para graficar
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        plt.plot(bin_centers, results['xi'], marker='o', linestyle='-', color=color, label=f'log10(Mstar) > {corte}')
    except Exception as e:
        print(f"Error en Corrfunc para log10(Mstar) > {corte}: {e}")

# Configurar gráfico
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Distancia [Mpc/h]')
plt.ylabel(r'$\xi(r)$')
plt.title('Función de correlación en muestras de galaxias')
plt.legend()
plt.grid()
plt.show()
