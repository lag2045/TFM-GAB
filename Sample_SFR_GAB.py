# Galaxy Assembly Bias usando SFR (Star Formation Rate)
# Importar Librerías 
import numpy as np # cálculos numéricos.
import matplotlib.pyplot as plt # para graficar
from Corrfunc.theory.xi import xi # calcula la función de correlación de dos puntos.
import os # manejo del sistema de archivos.

# Parámetros globales de la caja de simulación
boxsize = 205 # tamaño de la caja de simulación (Mpc/h).
nthreads = 4 # número de hilos para Corrfunc.
bin_edges = np.logspace(np.log10(0.1), np.log10(50), 20) # escala logarítmica de separaciones para calcular la correlación.
# Crea un directorio para guardar los resultados si no existe.
output_dir = "/Users/hakeem/Desktop/Python/shuffling_flexible"
os.makedirs(output_dir, exist_ok=True)

# Cargar catálogo original de halos y galaxias
dat = np.load('/Users/hakeem/Desktop/Modular/cat_z00.npy', allow_pickle=True).item()

# Extrae propiedades clave del diccionario. Datos del catálogo: pos: posiciones de galaxias / type: 0 = central, 1 = satélite / mst: log10 de la masa estelar.
# cross_sub2halo: asociación galaxia-halo / halo_pos y halo_mass: propiedades de halos / sfr: tasa de formación estelar.
gal_pos_all = np.array(dat['pos']) % boxsize
gal_type_all = np.array(dat['type'])
gal_mst_all = np.array(dat['mst'])  # log10(M*)
halo_ids_all = np.array(dat['cross_sub2halo'])
halo_pos_all = np.array(dat['halo_pos']) % boxsize
halo_mass_all = np.array(dat['halo_mass'])
sfr_all = np.array(dat['sfr'])  # Nueva propiedad: Star Formation Rate

# Ordena galaxias por SFR y selecciona las 10,000 con SFR más alta.
indices_sfr = np.argsort(sfr_all)[::-1][:10000]
sfr_sample = sfr_all[indices_sfr]

# Calcular percentiles para separar en alta y baja SFR: Alta SFR: Q3–Q4 (por encima del percentil 75) / Baja SFR: Q1–Q2 (por debajo del percentil 25).
p25, p75 = np.percentile(sfr_sample, [25, 75])
gal_sfr_baja = indices_sfr[sfr_sample < p25]
gal_sfr_alta = indices_sfr[sfr_sample > p75]

print(f"🔵 Galaxias con SFR baja: {len(gal_sfr_baja)}")
print(f"🔴 Galaxias con SFR alta: {len(gal_sfr_alta)}")

# Función de procesamiento para un sample específico: Esta función repite el análisis de correlación para un grupo (alta o baja SFR).
def procesar_sample(indices, tag):
# Iniciálizacion: Se almacenarán aquí los resultados de los 100 shufflings.
    ratios_all = []
    bin_centers = None
# Bucle con 100 repeticiones: Fija semilla para reproducibilidad.
    for i in range(100):
        np.random.seed(i)
# Extrae los datos de la submuestra
        gal_pos = gal_pos_all[indices]
        gal_type = gal_type_all[indices]
        halo_ids = halo_ids_all[indices]
# Reasigna IDs para compactar y obtener propiedades de halos
        halos_usados = np.unique(halo_ids)
        if len(halos_usados) == 0:
            continue

        old_to_new = {old: new for new, old in enumerate(halos_usados)}
        halo_ids = np.array([old_to_new[h] for h in halo_ids])
        halo_pos = halo_pos_all[halos_usados]
        halo_mass = halo_mass_all[halos_usados]

        if len(halo_mass) == 0:
            continue
# Binning por masa de halo: 
        log_mass = np.log10(halo_mass)
        bin_width = 0.1
        mass_bins = np.arange(np.min(log_mass), np.max(log_mass) + bin_width, bin_width)
        if len(mass_bins) < 2:
            continue
        bin_indices = np.digitize(log_mass, mass_bins)

        halos = []
# Estructura de halos: Para cada halo se guarda: su bin de masa. Posición central. Posiciones relativas de satélites.
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
# Sufflings: Dentro de cada bin, se reasignan los contenidos de halos aleatoriamente. Se reubican centrals y satélites respetando sus configuraciones internas.
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
# Cálculo de funciones de correlación: Se calcula la correlación de dos puntos original y shuffled. Se calcula el cociente para medir el assembly bias.
        x_o, y_o, z_o = original_galaxies.T
        x_s, y_s, z_s = shuffled_galaxies.T
        xi_o = xi(boxsize, nthreads, bin_edges, x_o, y_o, z_o)['xi']
        xi_s = xi(boxsize, nthreads, bin_edges, x_s, y_s, z_s)['xi']
        ratio = (xi_s + 1e-10) / (xi_o + 1e-10)

        if bin_centers is None:
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        ratios_all.append(ratio) # Almacenamiento

    if len(ratios_all) == 0:
        print(f"⚠️ No se pudo calcular la correlación para {tag}.")
        return None, None, None
# Resultado final de los 100 shufflings
    ratios_all = np.array(ratios_all)
    mean_ratio = np.mean(ratios_all, axis=0)
    std_ratio = np.std(ratios_all, axis=0)

    return bin_centers, mean_ratio, std_ratio

# Ejecutar análisis para SFR baja y alta
r_baja, mean_baja, std_baja = procesar_sample(gal_sfr_baja, tag="sfr_baja")
r_alta, mean_alta, std_alta = procesar_sample(gal_sfr_alta, tag="sfr_alta")

# Grafica los resultados del plot comparativo y lo guarda en escritorio.
plt.figure(figsize=(9,6))
if r_baja is not None:
    plt.semilogx(r_baja, mean_baja, color='green', label='SFR baja')
    plt.fill_between(r_baja, mean_baja - std_baja, mean_baja + std_baja, color='green', alpha=0.3, label=r'SFR baja ±1$\sigma$')
if r_alta is not None:
    plt.semilogx(r_alta, mean_alta, color='orange', label='SFR alta')
    plt.fill_between(r_alta, mean_alta - std_alta, mean_alta + std_alta, color='orange', alpha=0.3, label=r'SFR alta ±1$\sigma$')

plt.axhline(1, linestyle='--', color='black')
plt.xlabel("Separación r [Mpc/h]", fontsize=12)
plt.ylabel(r"Ratio $\xi_{\mathrm{shuffle}} / \xi_{\mathrm{original}}$", fontsize=12)
plt.title("Comparación del Galaxy Assembly Bias por SFR")
plt.legend()
plt.grid(True, alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.expanduser("~/Desktop/shuffling_comparacion_sfr.png"), dpi=300)
plt.show()
