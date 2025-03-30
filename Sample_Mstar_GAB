# Importa librerías para cálculos numéricos, gráficos y la función de correlación de Corrfunc.
import numpy as np
import matplotlib.pyplot as plt
from Corrfunc.theory.xi import xi
import os

#Define el tamaño de la caja de simulación (en Mpc/h), el número de hilos para Corrfunc y los bordes de los bins espaciales para r.
boxsize = 205
nthreads = 4
bin_edges = np.logspace(np.log10(0.1), np.log10(50), 20)
# Crea un directorio para guardar resultados.
output_dir = "/Users/hakeem/Desktop/Python/shuffling_flexible"
os.makedirs(output_dir, exist_ok=True)

# Cargar catálogo original de halos y galaxias
dat = np.load('/Users/hakeem/Desktop/Modular/cat_z00.npy', allow_pickle=True).item()

# Extrae y almacena las propiedades relevantes del catálogo: posición, tipo (central/satélite), masa estelar, IDs de halos, posiciones y masas de halos.
gal_pos_all = np.array(dat['pos']) % boxsize
gal_type_all = np.array(dat['type'])
gal_mst_all = np.array(dat['mst'])  # log10(M*)
halo_ids_all = np.array(dat['cross_sub2halo'])
halo_pos_all = np.array(dat['halo_pos']) % boxsize
halo_mass_all = np.array(dat['halo_mass'])

# Ordena por M* y selecciona las 10,000 más masivas.
indices_mst = np.argsort(gal_mst_all)[::-1][:10000]
mst_sample = gal_mst_all[indices_mst]

# Divide el muestreo en cuartiles: baja M* (< Q1) y alta M* (> Q3).
p25, p75 = np.percentile(mst_sample, [25, 75])
gal_mst_baja = indices_mst[mst_sample < p25]
gal_mst_alta = indices_mst[mst_sample > p75]

print(f"🔵 Galaxias con M* baja: {len(gal_mst_baja)}")
print(f"🔴 Galaxias con M* alta: {len(gal_mst_alta)}")
 
# Esta función es el núcleo del análisis, ejecutada 100 veces para cada sample (muestra).
def procesar_sample(indices, tag):
    ratios_all = [] # Inicializa las listas: Almacena los resultados de las correlaciones.
    bin_centers = None

# Bucle de 100 shufflings: Fija una semilla diferente cada vez para reproducibilidad.
    for i in range(100):
        np.random.seed(i)
        #Extrae la submuestra seleccionada
        gal_pos = gal_pos_all[indices]
        gal_type = gal_type_all[indices]
        halo_ids = halo_ids_all[indices]

# Reorganiza los IDs y selecciona las propiedades de los halos correspondientes
        halos_usados = np.unique(halo_ids)
        if len(halos_usados) == 0:
            continue

        old_to_new = {old: new for new, old in enumerate(halos_usados)}
        halo_ids = np.array([old_to_new[h] for h in halo_ids])
        halo_pos = halo_pos_all[halos_usados]
        halo_mass = halo_mass_all[halos_usados]

        if len(halo_mass) == 0:
            continue
# Bin de masa de halo (con bin_width = 0.01)
        log_mass = np.log10(halo_mass)
        bin_width = 0.1
        mass_bins = np.arange(np.min(log_mass), np.max(log_mass) + bin_width, bin_width)
        if len(mass_bins) < 2:
            continue
        bin_indices = np.digitize(log_mass, mass_bins)

        halos = []
# Construye estructura de halos con centrals y satélites
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
# Shuffling dentro de cada bin de masa: A cada halo se le asigna la configuración de otro halo del mismo bin / Se conservan las posiciones relativas de los satélites.
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
# Calcula la función de correlación y su cociente
        x_o, y_o, z_o = original_galaxies.T
        x_s, y_s, z_s = shuffled_galaxies.T
        xi_o = xi(boxsize, nthreads, bin_edges, x_o, y_o, z_o)['xi']
        xi_s = xi(boxsize, nthreads, bin_edges, x_s, y_s, z_s)['xi']
        ratio = (xi_s + 1e-10) / (xi_o + 1e-10)

        if bin_centers is None:
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        ratios_all.append(ratio)

    if len(ratios_all) == 0:
        print(f"⚠️ No se pudo calcular la correlación para {tag}.")
        return None, None, None
# Almacena los resultados
    ratios_all = np.array(ratios_all)
# Devuelve promedio y desviación estándar del ratio
    mean_ratio = np.mean(ratios_all, axis=0)
    std_ratio = np.std(ratios_all, axis=0)

    return bin_centers, mean_ratio, std_ratio

# Ejecuta la función para Masa estelar (M*) alta y baja.
r_alta, mean_alta, std_alta = procesar_sample(gal_mst_alta, tag="mst_alta")
r_baja, mean_baja, std_baja = procesar_sample(gal_mst_baja, tag="mst_baja")

# Grafica los resultados del plot comparativo
plt.figure(figsize=(9,6))
if r_baja is not None:
    plt.semilogx(r_baja, mean_baja, color='purple', label='M* baja')
    plt.fill_between(r_baja, mean_baja - std_baja, mean_baja + std_baja, color='purple', alpha=0.3, label=r'M* baja ±1$\sigma$')
if r_alta is not None:
    plt.semilogx(r_alta, mean_alta, color='goldenrod', label='M* alta')
    plt.fill_between(r_alta, mean_alta - std_alta, mean_alta + std_alta, color='goldenrod', alpha=0.3, label=r'M* alta ±1$\sigma$')

plt.axhline(1, linestyle='--', color='black')
plt.xlabel("Separación r [Mpc/h]", fontsize=12)
plt.ylabel(r"Ratio $\xi_{\mathrm{shuffle}} / \xi_{\mathrm{original}}$", fontsize=12)
plt.title("Comparación del Galaxy Assembly Bias por masa estelar")
plt.legend()
plt.grid(True, alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.expanduser("~/Desktop/shuffling_comparacion_mstar.png"), dpi=300)
plt.show()
