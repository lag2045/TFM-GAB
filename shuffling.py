import numpy as np

# Parámetros de la simulación
num_halos = 100  # Número de halos
num_galaxias_por_halo = 10  # Número de galaxias por halo
caja_tamaño = 10.0  # Tamaño de la caja (dimensiones de 0 a caja_tamaño)

# Generar posiciones iniciales de halos y galaxias
halos_posiciones = np.random.uniform(0, caja_tamaño, (num_halos, 3))  # Posiciones (x, y, z)
halos_masas = np.random.uniform(12, 15, num_halos)  # Masas en logaritmo

# Generar posiciones relativas de galaxias dentro de cada halo
galaxias_posiciones_relativas = np.random.uniform(-0.5, 0.5, (num_halos, num_galaxias_por_halo, 3))
galaxias_velocidades_relativas = np.random.uniform(-100, 100, (num_halos, num_galaxias_por_halo, 3))

# Calcular posiciones absolutas de las galaxias
galaxias_posiciones = halos_posiciones[:, np.newaxis, :] + galaxias_posiciones_relativas

# Aplicar condiciones periódicas a las galaxias
galaxias_posiciones %= caja_tamaño  # Si sale de un lado, entra por el otro

# Función para realizar el shuffling
def realizar_shuffling(halos_posiciones, galaxias_posiciones_relativas, galaxias_velocidades_relativas):
    """
    Intercambia las galaxias entre halos mientras respeta las condiciones periódicas.
    """
    # Crear un índice aleatorio para hacer el intercambio
    indices_shuffle = np.random.permutation(len(halos_posiciones))

    # Intercambiar posiciones y velocidades relativas de galaxias entre halos
    galaxias_posiciones_relativas_shuffled = galaxias_posiciones_relativas[indices_shuffle]
    galaxias_velocidades_relativas_shuffled = galaxias_velocidades_relativas[indices_shuffle]

    # Calcular nuevas posiciones absolutas de galaxias
    galaxias_posiciones_nuevas = halos_posiciones[:, np.newaxis, :] + galaxias_posiciones_relativas_shuffled
    galaxias_posiciones_nuevas %= caja_tamaño  # Aplicar condiciones periódicas

    return galaxias_posiciones_nuevas, galaxias_velocidades_relativas_shuffled

# Ejecutar el shuffling
galaxias_posiciones_nuevas, galaxias_velocidades_relativas_nuevas = realizar_shuffling(
    halos_posiciones, galaxias_posiciones_relativas, galaxias_velocidades_relativas
)

# Validar que todas las galaxias están dentro de la caja
assert np.all(galaxias_posiciones_nuevas >= 0) and np.all(galaxias_posiciones_nuevas < caja_tamaño), \
    "Error: Algunas galaxias están fuera de la caja."

# Visualizar los resultados
print("Posiciones iniciales de los halos:")
print(halos_posiciones)
print("\nPosiciones absolutas de galaxias antes del shuffling:")
print(galaxias_posiciones)
print("\nPosiciones absolutas de galaxias después del shuffling:")
print(galaxias_posiciones_nuevas)