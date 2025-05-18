# Galaxy Assembly Bias (GAB) en función del color (g‒r) de las galaxias más brillantes en banda i
# Importacion de librerias
import numpy as np # operaciones matemáticas.
import matplotlib.pyplot as plt #  hacer gráficos
from Corrfunc.theory.xi import xi # calcular la función de correlación.
import os # gestión de archivos

# Parámetros globales de la caja de simulación 
boxsize = 205 # tamaño de la caja de simulación (en Mpc/h)
nthreads = 4 # número de hilos usados por Corrfunc
bin_edges = np.logspace(np.log10(0.1), np.log10(50), 20) # bins logarítmicos de distancia para la correlación
# carpeta para guardar resultados
output_dir = "/Users/hakeem/Desktop/Python/shuffling_flexible"
os.makedirs(output_dir, exist_ok=True)

# Cargar el catálogo como diccionario: datos original de los halos y galaxias
dat = np.load('/Users/hakeem/Desktop/Modular/cat_z00.npy', allow_pickle=True).item()

# Extrae propiedades del catálogo: Posición, tipo (central/satélite), masa estelar, ID de halo, masa/posición de halo y magnitudes absolutas (col = [g, r, i]).
gal_pos_all = np.array(dat['pos']) % boxsize
gal_type_all = np.array(dat['type'])
gal_mst_all = np.array(dat['mst'])  # log10(M*)
halo_ids_all = np.array(dat['cross_sub2halo'])
halo_pos_all = np.array(dat['halo_pos']) % boxsize
halo_mass_all = np.array(dat['halo_mass'])
mag_abs = np.array(dat['col'])  # columnas: g, r, i

mag_g = mag_abs[:, 0]
mag_r = mag_abs[:, 1]
mag_i = mag_abs[:, 2]
color_gr = mag_g - mag_r  # Color (g - r)

# === SELECCIÓN POR COLOR ===
azul_cut = 0.6
rojo_cut = 0.75

indices_azules = np.where(color_gr < azul_cut)[0]
indices_rojas = np.where(color_gr > rojo_cut)[0]

# Igualar número de galaxias
n_min = min(len(indices_azules), len(indices_rojas))
indices_azules = indices_azules[np.argsort(mag_i[indices_azules])[:n_min]]  # más brillantes
indices_rojas = indices_rojas[np.argsort(mag_i[indices_rojas])[:n_min]]    # más brillantes

print(f"Número de galaxias azules seleccionadas: {len(indices_azules)}")
print(f"Número de galaxias rojas seleccionadas: {len(indices_rojas)}")

# 3. Histogramas de masa estelar y color: Visualiza la distribución de M* y color g–r entre las más brillantes.
plt.figure(figsize=(14, 6))

# Histograma de masa estelar
plt.subplot(1, 2, 1)
plt.hist(gal_mst_all[indices_azules], bins=30, color='blue', edgecolor='black', alpha=0.6, label='Galaxias azules')
plt.hist(gal_mst_all[indices_rojas], bins=30, color='red', edgecolor='black', alpha=0.6, label='Galaxias rojas')
plt.xlabel(r'$\log_{10}(M_*[M_\odot])$', fontsize=12)
plt.ylabel('Número de galaxias')
plt.title('Distribución de masa estelar')
plt.legend()

# Histograma de color g-r
plt.subplot(1, 2, 2)
plt.hist(color_gr[indices_azules], bins=30, color='blue', edgecolor='black', alpha=0.6, label='Galaxias azules')
plt.hist(color_gr[indices_rojas], bins=30, color='red', edgecolor='black', alpha=0.6, label='Galaxias rojas')
plt.xlabel('Color (g - r)', fontsize=12)
plt.ylabel('Número de galaxias')
plt.title('Distribución de color $(g - r)$')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.expanduser("~/Desktop/histogramas_mstar_color_azul_rojo.png"))
plt.show()

print("Histogramas de masa estelar y color generados (sin total).")

# Función de procesamiento para un sample específico
def procesar_sample(indices, tag): # Esta función calcula el Galaxy Assembly Bias para un grupo específico.
    # Incialización
    ratios_all = []
    bin_centers = None
# Bucle de 100 shufflings: Se repite el shuffling 100 veces con diferente semilla.
    for i in range(100):
        np.random.seed(i)
        # Subcatálogo de galaxias
        gal_pos = gal_pos_all[indices]
        gal_type = gal_type_all[indices]
        halo_ids = halo_ids_all[indices]
# Asociación con halos.
        halos_usados = np.unique(halo_ids)
        if len(halos_usados) == 0:
            continue

        old_to_new = {old: new for new, old in enumerate(halos_usados)}
        halo_ids = np.array([old_to_new[h] for h in halo_ids])
        halo_pos = halo_pos_all[halos_usados]
        halo_mass = halo_mass_all[halos_usados]

        if len(halo_mass) == 0:
            continue
# Binning en masa de halo: Se agrupan halos en bins estrechos de masa logarítmica para hacer el shuffling correctamente.
        log_mass = np.log10(halo_mass)
        bin_width = 0.1
        mass_bins = np.arange(np.min(log_mass), np.max(log_mass) + bin_width, bin_width)
        if len(mass_bins) < 2:
            continue
        bin_indices = np.digitize(log_mass, mass_bins)

        halos = []
# Reconstrucción de halos: Se guarda para cada halo: ID, bin, posición, central, satélites relativos.
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

        original_galaxies, shuffled_galaxies = [], []
# Shuffling dentro de cada bin: Se mezclan configuraciones internas de halos del mismo bin, preservando sus posiciones relativas.
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
# Cálculo de correlación y ratio: Calcula la función de correlación de dos puntos para el catálogo original y el shuffled, y su cociente.
        x_o, y_o, z_o = original_galaxies.T
        x_s, y_s, z_s = shuffled_galaxies.T
        xi_o = xi(boxsize, nthreads, bin_edges, x_o, y_o, z_o)['xi']
        xi_s = xi(boxsize, nthreads, bin_edges, x_s, y_s, z_s)['xi']
        ratio = (xi_s + 1e-10) / (xi_o + 1e-10)

        if bin_centers is None:
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        ratios_all.append(ratio)

    if len(ratios_all) == 0:
        print(f"No se pudo calcular la correlación para {tag}.")
        return None, None, None
# Resultados: Devuelve la media y la desviación estándar del ratio.
    ratios_all = np.array(ratios_all)
    mean_ratio = np.mean(ratios_all, axis=0)
    std_ratio = np.std(ratios_all, axis=0)

    return bin_centers, mean_ratio, std_ratio

# Ejecutar análisis para galaxias azules y rojas: Ejecuta la función para los dos grupos.
r_azul, mean_azul, std_azul = procesar_sample(indices_azules, tag="azules")
r_rojo, mean_rojo, std_rojo = procesar_sample(indices_rojas, tag="rojas")

# Muestra y guarda el plot comparativo.
plt.figure(figsize=(9,6))
if r_azul is not None:
    plt.semilogx(r_azul, mean_azul, color='blue', label='Galaxias azules')
    plt.fill_between(r_azul, mean_azul - std_azul, mean_azul + std_azul, color='blue', alpha=0.3)
if r_rojo is not None:
    plt.semilogx(r_rojo, mean_rojo, color='red', label='Galaxias rojas')
    plt.fill_between(r_rojo, mean_rojo - std_rojo, mean_rojo + std_rojo, color='red', alpha=0.3)

plt.axhline(1, linestyle='--', color='black')
plt.xlim(0.1, 20)
plt.ylim(0.54, 1.05)
plt.xlabel(r"$r$ [$\mathrm{Mpc}/h$]", fontsize=12)
plt.ylabel(r"$R = (\xi_{\mathrm{shuffled}} / \xi_{\mathrm{original}})$", fontsize=12)
plt.title("Sesgo de ensamblaje de galaxias en función del color $(g - r)$")
plt.legend()
plt.grid(True, alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.expanduser("~/Desktop/GAB_Color_BlueRed_SIN_TOTAL.png"))
plt.show()

print("Análisis completo finalizado y guardado en el escritorio.")
