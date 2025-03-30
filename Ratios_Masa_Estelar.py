# Importar librerias 
import numpy as np # para análisis numérico
import matplotlib.pyplot as plt  # para generar gráficos
from Corrfunc.theory.xi import xi  # calular la funcion de correlación
import os 

# Parámetros globales de la caja de simulación: Define el tamaño de la caja de simulación, el número de hilos y los bins de distancia r para la correlación.
boxsize = 205
nthreads = 4
bin_edges = np.logspace(np.log10(0.1), np.log10(50), 20)
# Crea una carpeta donde se guardarán los resultados (npz y gráficas).
output_dir = "/Users/hakeem/Desktop/Python/shuffling_por_masa_galaxias"
os.makedirs(output_dir, exist_ok=True)

# Carga el diccionario con los datos del catálogo de simulación 
dat = np.load('/Users/hakeem/Desktop/Modular/cat_z00.npy', allow_pickle=True).item()

# Extrae: posiciones de galaxias, tipo (central/satélite), masa estelar logarítmica, IDs de halo, posiciones y masas de halos.
gal_pos_all = np.array(dat['pos']) % boxsize
gal_type_all = np.array(dat['type'])
gal_mst_all = np.array(dat['mst'])  # log10(M*)
halo_ids_all = np.array(dat['cross_sub2halo'])
halo_pos = np.array(dat['halo_pos']) % boxsize
halo_mass = np.array(dat['halo_mass'])

# Función que procesa las N_top galaxias más masivas para aplicar shuffling y calcular el ratio de correlación.
def procesar_galaxias_masivas(N_top):
# Selecciona las N_top galaxias con mayor masa estelar.
    indices_masivos = np.argsort(gal_mst_all)[::-1][:N_top]
# Extrae las propiedades de la galaxias y halos
    gal_pos = gal_pos_all[indices_masivos]
    gal_type = gal_type_all[indices_masivos]
    halo_ids = halo_ids_all[indices_masivos]
# Asocia galaxias a halos y filtra sus propiedades: Reasigna los IDs de halo a un rango continuo para facilitar el análisis.
    halos_usados = np.unique(halo_ids)
    old_to_new = {old: new for new, old in enumerate(halos_usados)}

    halo_ids = np.array([old_to_new[h] for h in halo_ids])
    halo_pos_filtrado = halo_pos[halos_usados]
    halo_mass_filtrado = halo_mass[halos_usados]
# Bin en masa de halo: Agrupa los halos usados en bins logarítmicos de masa (0.1 dex)
    log_mass = np.log10(halo_mass_filtrado)
    bin_width = 0.1
    mass_bins = np.arange(np.min(log_mass), np.max(log_mass) + bin_width, bin_width)
    bin_indices = np.digitize(log_mass, mass_bins)
 
# Construcción de estructura de halos: Para cada halo: se identifica el central y los satélites / Guarda la posición central y los satélites relativos (con condiciones periódicas).
    halos = []
    for i in range(len(halo_pos_filtrado)):
        mask = (halo_ids == i)
        gal_pos_i = gal_pos[mask]
        gal_type_i = gal_type[mask]

        if np.sum(gal_type_i == 0) == 0:
            continue

        central = gal_pos_i[gal_type_i == 0][0]
        sats = gal_pos_i[gal_type_i == 1]

        rel_sats = sats - central
        rel_sats -= boxsize * np.round(rel_sats / boxsize)

        halos.append({
            "id": i,
            "bin": bin_indices[i],
            "halo_pos": halo_pos_filtrado[i],
            "central": central,
            "rel_satellites": rel_sats
        })
# Shuffling dentro de cada bin de masa: Reordena la asignación de galaxias entre halos del mismo bin de masa.
# Conserva la estructura interna (centrales + satélites), pero en otro halo
    original_galaxies, shuffled_galaxies = [], []

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

  # Cálculo de funciones de correlación: Calcula la función de correlación ξ(r) antes y después del shuffling. El cociente muestra cuánto se ha alterado la señal de clustering.
    x_o, y_o, z_o = original_galaxies.T
    x_s, y_s, z_s = shuffled_galaxies.T

    xi_o = xi(boxsize, nthreads, bin_edges, x_o, y_o, z_o)['xi']
    xi_s = xi(boxsize, nthreads, bin_edges, x_s, y_s, z_s)['xi']
    ratio = (xi_s + 1e-10) / (xi_o + 1e-10)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
# Guardado de resultados: Guarda los resultados (bin_centers, xi, ratio) como archivo .npz para cada valor de N_top.
    tag = f"{N_top}_galaxias"
    np.savez(f"{output_dir}/xi_data_{tag}.npz", r=bin_centers, xi_original=xi_o, xi_shuffled=xi_s, ratio=ratio)

# Función para comparar ratios con cortes en masa estelar
def plot_comparacion_ratios(lista_N):
    plt.figure(figsize=(8,6))
# Compara el assembly bias (via el ratio de ξ) para distintas selecciones por masa estelar.
# Carga los datos de cada archivo .npz generado previamente.
# Extrae el rango de masa estelar correspondiente.
# Dibuja el ratio ξshuffle/ξoriginal para cada subconjunto.
    for N_top in lista_N:
        data = np.load(f"{output_dir}/xi_data_{N_top}_galaxias.npz")
        bin_centers, ratio = data['r'], data['ratio']
        corte_mst = np.sort(gal_mst_all)[::-1][N_top-1]
       
      # Definir el rango de masa estelar para el corte
        corte_min = corte_mst  # Valor mínimo de log10(M*)
        corte_max = np.max(gal_mst_all[:N_top])  # Valor máximo en la selección
        plt.semilogx(bin_centers, ratio, marker='o', 
             label=rf'log$_{{10}}(M_{{\odot}})$ = [{corte_min:.2f} - {corte_max:.2f}] ({N_top} galaxias)')

   # Define el estilo del gráfico: Añade línea de referencia en 1 (donde no hay efecto del shuffling) / Etiqueta bien el gráfico y lo guarda como imagen en el escritorio.
    plt.axhline(1, linestyle='--', color='black')
    plt.xlim(0.1, 20)
    plt.xlabel("r [Mpc/h]")
    plt.ylabel(r"Ratio $\xi_{\mathrm{shuffle}} / \xi_{\mathrm{original}}$")
    plt.title(r"Comparación de ratios para masa estelar de galaxias más masivas")
    plt.grid(True, alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.expanduser("~/Desktop/comparacion_ratios.png"), dpi=300)
    plt.show()

# Ejecución automática: Ejecuta todo el análisis para 4 subconjuntos distintos de galaxias masivas. Luego compara los resultados en un solo gráfico.
lista_N = [10000, 20000, 40000, 60000]
for N in lista_N:
    procesar_galaxias_masivas(N)

plot_comparacion_ratios(lista_N)
