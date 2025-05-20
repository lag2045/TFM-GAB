# Importa librerías para cálculos numéricos, gráficos y la función de correlación de Corrfunc.
import numpy as np  # Necesario para análisis numérico
import matplotlib.pyplot as plt # Necesario para generar gráficos
from Corrfunc.theory.xi import xi # Permite calular la función de correlación
import os  # Manejo de archivos y carpetas

#Define el tamaño de la caja de simulación (en Mpc/h), el número de hilos para Corrfunc y los bordes de los bins espaciales para r.
boxsize = 205 # Define el tamaño de la caja periódica.
nthreads = 4  # Número de hilos para Corrfunc
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
# Bin de masa de halo (con bin_width = 0.1)
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
        print(f"No se pudo calcular la correlación para {tag}.")
        return None, None, None
# Almacena los resultados
    ratios_all = np.array(ratios_all)
    # Devuelve promedio y desviación estándar del ratio
    mean_ratio = np.mean(ratios_all, axis=0)
    std_ratio = np.std(ratios_all, axis=0)

    return bin_centers, mean_ratio, std_ratio

# Procesar 10k, 20k, 30k galaxias (más masivas directamente)

samples = [10000, 20000, 30000]
colores = ['yellow', 'red', 'green']
labels = ['10.000 galaxias', '20.000 galaxias', '30.000 galaxias']
# Grafica los resultados del plot comparativo
plt.figure(figsize=(9,6))

for sample_size, color, label in zip(samples, colores, labels):
    # Selección directa de las galaxias más masivas
    indices_mst = np.argsort(gal_mst_all)[::-1][:sample_size]

    r_mst, mean_mst, std_mst = procesar_sample(indices_mst, tag=f"mst_top_{sample_size}")

    if r_mst is not None:
        plt.semilogx(r_mst, mean_mst, color=color, label=f'{label} (M$_\star$)')
        plt.fill_between(r_mst, mean_mst - std_mst, mean_mst + std_mst, color=color, alpha=0.3)
# Define el estilo del gráfico: Añade línea de referencia en 1 (donde no hay efecto del shuffling) / Etiqueta bien el gráfico y lo guarda como imagen en el escritorio.
plt.axhline(1, linestyle='--', color='black')
plt.xlim(0.1, 20)
plt.ylim(0.6, 1.06)
plt.xlabel(r"$r$ [$\mathrm{Mpc}/h$]", fontsize=12)
plt.ylabel(r"$R = (\xi_{\mathrm{shuffled}} / \xi_{\mathrm{original}})$", fontsize=12)
plt.title("Sesgo de ensamblaje de galaxias en función de la masa estelar")
plt.legend()
plt.grid(True, alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.expanduser("~/Desktop/GAB_Mstar_MostMassive.png"), dpi=300)
plt.show()

print("Gráfico combinado generado usando solo las galaxias más masivas.")
