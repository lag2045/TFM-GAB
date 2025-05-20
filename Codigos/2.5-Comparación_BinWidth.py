import numpy as np # Necesario para manejo de arrays, logaritmos, estadísticas.
import matplotlib.pyplot as plt  # Sirve en la creación de gráficos.
from Corrfunc.theory.xi import xi # Necesario en el cálculo eficiente de la función de correlación.
import os  # Sirve en la gestión de rutas de archivos y creación de carpetas.

# Configuración inicial 
boxsize = 205  # Tamaño de la caja simulada (Mpc/h).
nthreads = 4 # número de hilos para Corrfunc.
bin_edges = np.logspace(np.log10(0.1), np.log10(50), 20) # bordes de bins logarítmicos para r en el cálculo de la función de correlación.

# Carga el catálogo de galaxias como diccionario con las variables clave.
cat_path = '/Users/hakeem/Desktop/Modular/cat_z00.npy'
dat = np.load(cat_path, allow_pickle=True).item()
# Extracción de variables del catálogo: Se extraen y normalizan posiciones (dentro del box).
gal_pos_all = np.array(dat['pos']) % boxsize
gal_type_all = np.array(dat['type']) # 0 = central, 1 = satélite.
gal_mst_all = np.array(dat['mst'])  #  masa estelar (en log10).
halo_ids_all = np.array(dat['cross_sub2halo']) # relación galaxia–halo.
halo_pos_all = np.array(dat['halo_pos']) % boxsize
halo_mass_all = np.array(dat['halo_mass'])

# Función de shuffling y cálculo de función de correlación
def procesar_catalogo(N_top, output_dir, bin_width):
    # Normalizar nombre de archivo (punto → guion bajo)
    tag = f"{N_top}_galaxia_bw{bin_width:.2f}".replace('.', '_')
    file_path = os.path.join(output_dir, f"xi_data_{tag}.npz")
#  Selección de galaxias.
    indices = np.argsort(gal_mst_all)[::-1][:N_top]
    gal_pos = gal_pos_all[indices]
    gal_type = gal_type_all[indices]
    halo_ids = halo_ids_all[indices]
#  Asignación de nuevos IDs a halos usados
    halos_usados = np.unique(halo_ids)
    old_to_new = {old: new for new, old in enumerate(halos_usados)}
    halo_ids = np.array([old_to_new[h] for h in halo_ids])
    halo_pos = halo_pos_all[halos_usados]
    halo_mass = halo_mass_all[halos_usados]
# Bin de masa de halos.
    log_mass = np.log10(halo_mass)
    mass_bins = np.arange(np.min(log_mass), np.max(log_mass) + bin_width, bin_width)
    bin_indices = np.digitize(log_mass, mass_bins)

    halos = []
    for i in range(len(halo_pos)):
        mask = (halo_ids == i)
        gal_pos_i = gal_pos[mask]
        gal_type_i = gal_type[mask]
        if np.sum(gal_type_i == 0) == 0:
            continue
        central = gal_pos_i[gal_type_i == 0][0]
        sats = gal_pos_i[gal_type_i == 1]
        rel_sats = sats - central
        rel_sats -= boxsize * np.round(rel_sats / boxsize)
        # Construcción de estructura de halos: Guarda posición del halo, galaxia central y posiciones relativas de los satélite.
        halos.append({
            "id": i,
            "bin": bin_indices[i],
            "halo_pos": halo_pos[i],
            "central": central,
            "rel_satellites": rel_sats
        })

    original_galaxies, shuffled_galaxies = [], []
    # Shuffling por bin de masa: Intercambia galaxias dentro del mismo bin de masa, preservando estadísticas de masa.
    for bin_num in np.unique(bin_indices):
        halos_bin = [h for h in halos if h["bin"] == bin_num]
        if len(halos_bin) < 2:
            continue
        indices = np.arange(len(halos_bin))
        np.random.shuffle(indices)
        for orig, idx in zip(halos_bin, indices):
            shuf = halos_bin[idx]
            # Construcción de nuevas posiciones: Cambia la galaxia central de cada halo, manteniendo estructura relativa de los satélites.
            new_central = orig['halo_pos'] + (shuf['central'] - shuf['halo_pos'])
            new_central %= boxsize
            new_sats = new_central + shuf['rel_satellites']
            new_sats %= boxsize
            shuffled_galaxies.append(new_central)
            shuffled_galaxies.extend(new_sats)
            original_galaxies.append(orig['central'])
            original_galaxies.extend(orig['rel_satellites'] + orig['central'])

    if len(original_galaxies) < 2 or len(shuffled_galaxies) < 2:
        print(f"Muy pocas galaxias en el bin_width={bin_width:.2f} (N_top={N_top}) → archivo no guardado.")
        return

    original_galaxies = np.array(original_galaxies)
    shuffled_galaxies = np.array(shuffled_galaxies)
    x_o, y_o, z_o = original_galaxies.T
    x_s, y_s, z_s = shuffled_galaxies.T
# Cálculo de función de correlación: Calcula funciónes de correlacion antes y después del intercambio usando Corrfunc.
    xi_o = xi(boxsize, nthreads, bin_edges, x_o, y_o, z_o)['xi']
    xi_s = xi(boxsize, nthreads, bin_edges, x_s, y_s, z_s)['xi']
    ratio = (xi_s + 1e-10) / (xi_o + 1e-10)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
# Guardado de resultados.
    np.savez(file_path, r=bin_centers, xi_original=xi_o, xi_shuffled=xi_s, ratio=ratio)
    print(f"Guardado: {file_path}")

# Función para comparar Bin Widths 
# Corre procesar_catalogo() 100 veces por cada bin_width, usando una semilla diferente para obtener distintas permutaciones aleatorias.
def comparar_binwidths(N_top=20000, bin_widths=[0.05, 0.1, 0.2], num_shufflings=100):
    output_dir = f"/Users/hakeem/Desktop/Python/shuffling_flexible/N{N_top}"
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10,7))
    # Corre procesar_catalogo() 100 veces por cada bin_width, usando una semilla diferente para obtener distintas permutaciones aleatorias.
    for bw in bin_widths:
        print(f"\n Analizando bin_width = {bw}")
        ratios = []
        bin_centers = None
        for i in range(num_shufflings):
            np.random.seed(i)
            procesar_catalogo(N_top, output_dir, bin_width=bw)
            tag = f"{N_top}_galaxia_bw{bw:.2f}".replace('.', '_')
            file_path = os.path.join(output_dir, f"xi_data_{tag}.npz")
            if not os.path.exists(file_path):
                continue
            data = np.load(file_path)
            if bin_centers is None:
                bin_centers = data['r']
            ratios.append(data['ratio'])

        if len(ratios) == 0:
            print(f"No se generaron resultados para bin_width = {bw}")
            continue
# Calcula la media y desviación estándar del cociente para representar incertidumbre estadística.
        ratios = np.array(ratios)
        mean_ratio = np.mean(ratios, axis=0)
        std_ratio = np.std(ratios, axis=0)
# Visualización del resultado: Gráfico de líneas logarítmico con sombreado de error para cada bin_width.
        plt.semilogx(bin_centers, mean_ratio, marker='o', label=f'bw = {bw}')
        plt.fill_between(bin_centers, mean_ratio - std_ratio, mean_ratio + std_ratio, alpha=0.2)

    plt.axhline(1, linestyle='--', color='black')
    plt.xlim(0.1, 20)
    plt.ylim(0.55, 1.05)
    plt.xlabel("r [Mpc/h]")
    plt.ylabel(r"R = ($\xi_{\mathrm{shuffled}} / \xi_{\mathrm{original}}$)")
    plt.title(f"Comparación del efecto del ancho de bin en masa de halo")
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.tight_layout()

    out_plot = f"~/Desktop/comparacion_binwidths_N{N_top}.png"
    plt.savefig(os.path.expanduser(out_plot), dpi=300)
    plt.show()
    print(f"Gráfico final guardado en: {out_plot}")

# Ejecución para varias muestras
# Ejecuta el análisis para una muestra de 20,000 galaxias, explorando 5 valores distintos de ancho de bin, con 100 repeticiones por caso.
for N in [20000]:
    comparar_binwidths(N_top=N, bin_widths=[0.01, 0.04, 0.07, 0.1, 0.2], num_shufflings=100)
