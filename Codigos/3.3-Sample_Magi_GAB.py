# Galaxy Assembly bias en función de la magnitud absoluta en la banda i
# Importar librerias
import numpy as np # Necesario  para cálculos numéricos.
import matplotlib.pyplot as plt # Necesario para graficar.
from Corrfunc.theory.xi import xi # Permite calcular la función de correlación de dos puntos (Corrfunc).
import os # Necesario para manejo del sistema de archivos.

# Parámetros globales de caja de simulación
boxsize = 205 # tamaño de la simulación.
nthreads = 4 # número de hilos para Corrfunc.
bin_edges = np.logspace(np.log10(0.1), np.log10(50), 20)  # separaciones espaciales en escala logarítmica.
# Carpeta donde se guardan los resultados.
output_dir = "/Users/hakeem/Desktop/Python/shuffling_flexible"
os.makedirs(output_dir, exist_ok=True)

# Cargar catálogo original de halos y galaxias
dat = np.load('/Users/hakeem/Desktop/Modular/cat_z00.npy', allow_pickle=True).item()

# Datos del catálogo: Extrae posiciones, tipos (central/satélite), masas estelares, identificadores de halo, magnitudes absolutas en g, r, i.
gal_pos_all = np.array(dat['pos']) % boxsize
gal_type_all = np.array(dat['type'])
gal_mst_all = np.array(dat['mst'])  # log10(M*)
halo_ids_all = np.array(dat['cross_sub2halo'])
halo_pos_all = np.array(dat['halo_pos']) % boxsize
halo_mass_all = np.array(dat['halo_mass'])
mag_abs = np.array(dat['col'])  # columnas: g, r, i
mag_i = mag_abs[:, 2]  # Magnitud en banda i

# Función de procesamiento para un sample específico: Esta función calcula el shuffling para una submuestra dada.
def procesar_sample(indices, tag):
    # Inicialización: Para almacenar los resultados de cada corrida.
    ratios_all = []
    bin_centers = None
# Repite el proceso 100 veces: Usa una semilla diferente para cada shuffling.
    for i in range(100):
        np.random.seed(i)
        # Extrae las galaxias de la submuestra
        gal_pos = gal_pos_all[indices]
        gal_type = gal_type_all[indices]
        halo_ids = halo_ids_all[indices]
# Reasigna los IDs de halo para compactarlos
        halos_usados = np.unique(halo_ids)
        if len(halos_usados) == 0:
            continue

        old_to_new = {old: new for new, old in enumerate(halos_usados)}
        halo_ids = np.array([old_to_new[h] for h in halo_ids])
        halo_pos = halo_pos_all[halos_usados]
        halo_mass = halo_mass_all[halos_usados]

        if len(halo_mass) == 0:
            continue
# Bins de masa de halo: Se agrupan halos en bins estrechos de masa para hacer el shuffling correctamente.
        log_mass = np.log10(halo_mass)
        bin_width = 0.1
        mass_bins = np.arange(np.min(log_mass), np.max(log_mass) + bin_width, bin_width)
        if len(mass_bins) < 2:
            continue
        bin_indices = np.digitize(log_mass, mass_bins)

        halos = []
        # Construcción de objetos halo. Se guarda para cada halo: ID, bin en masa, posición del halo, posición del central, satélites relativos al central.
        for j in range(len(halo_pos)):
            mask = (halo_ids == j)
            gal_pos_i = gal_pos[mask]
            gal_type_i = gal_type[mask]
            if np.sum(gal_type_i == 0) == 0:
                continue
            central = gal_pos_i[gal_type_i == 0][0]
            sats = gal_pos_i[gal_type_i == 1]
            rel_sats = sats - central
            rel_sats -= boxsize * np.round(rel_sats / boxsize)
            halos.append({
                "id": j,
                "bin": bin_indices[j],
                "halo_pos": halo_pos[j],
                "central": central,
                "rel_satellites": rel_sats
            })
# Shuffling por bin de masa: Se mezclan las configuraciones internas entre halos del mismo bin de masa. Se trasladan satélites manteniendo su configuración relativa.
        original_galaxies, shuffled_galaxies = [], []
        for bin_num in np.unique(bin_indices):
            halos_bin = [h for h in halos if h["bin"] == bin_num]
            if len(halos_bin) < 2:
                continue
            indices_bin = np.arange(len(halos_bin))
            np.random.shuffle(indices_bin)
            for orig, idx in zip(halos_bin, indices_bin):
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
        if len(original_galaxies) == 0 or len(shuffled_galaxies) == 0:
            continue
# Cálculo de función de correlación y ratio: Calcula la función de correlación para el original y el shuffled. Se añade un pequeño término para evitar división por cero.
        x_o, y_o, z_o = original_galaxies.T
        x_s, y_s, z_s = shuffled_galaxies.T
        xi_o = xi(boxsize, nthreads, bin_edges, x_o, y_o, z_o)['xi']
        xi_s = xi(boxsize, nthreads, bin_edges, x_s, y_s, z_s)['xi']
        ratio = (xi_s + 1e-10) / (xi_o + 1e-10)

        if bin_centers is None:
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        ratios_all.append(ratio) # Acumula los resultados

    if len(ratios_all) == 0:
        print(f"No se pudo calcular la correlación para {tag}.")
        return None, None, None
# Retorna media y desviación estándar del ratio
    ratios_all = np.array(ratios_all)
    mean_ratio = np.mean(ratios_all, axis=0)
    std_ratio = np.std(ratios_all, axis=0)

    return bin_centers, mean_ratio, std_ratio

# Procesar las 10k, 20k, 30k galaxias más brillantes (menor mag i)

samples = [10000, 20000, 30000]
colores = ['orange', 'grey', 'green']
labels = ['10.000 galaxias', '20.000 galaxias', '30.000 galaxias']
# Grafica los resultados del plot comparativo y lo guarda.
plt.figure(figsize=(9,6))

for sample_size, color, label in zip(samples, colores, labels):
    indices_mag = np.argsort(mag_i)[:sample_size]  # Más brillantes: menor magnitud i
    r_mag, mean_mag, std_mag = procesar_sample(indices_mag, tag=f"mag_top_{sample_size}")

    if r_mag is not None:
        plt.semilogx(r_mag, mean_mag, color=color, label=f'{label} (Magnitud i)')
        plt.fill_between(r_mag, mean_mag - std_mag, mean_mag + std_mag, color=color, alpha=0.3)
# Define el estilo del gráfico: Añade línea de referencia en 1 (donde no hay efecto del shuffling) / Etiqueta bien el gráfico y lo guarda como imagen en el escritorio.
plt.axhline(1, linestyle='--', color='black')
plt.xlim(0.1, 20)
plt.ylim(0.64, 1.06)
plt.xlabel(r"$r$ [$\mathrm{Mpc}/h$]", fontsize=12)
plt.ylabel(r"$R = (\xi_{\mathrm{shuffled}} / \xi_{\mathrm{original}})$", fontsize=12)
plt.title("Sesgo de ensamblaje de galaxias en función de la magnitud absoluta $i$")
plt.legend()
plt.grid(True, alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.expanduser("~/Desktop/GAB_MagI_MostBrilliant.png"), dpi=300)
plt.show()

print("Gráfico de magnitud i brillante generado y guardado en el escritorio.")
