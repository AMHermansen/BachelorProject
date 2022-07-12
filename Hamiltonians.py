import numpy as np
import matplotlib.pyplot as plt


# Used for control to solve_hamiltonian
def hamiltonian_ho(positions, momenta, h_params):
    """
    Harmonic Oscillator.
    :param positions:
    :param momenta:
    :param h_params: Tuple containing (Mass, Spring constant (k))
    :return:
    """
    mass, k = h_params
    return momenta[0] ** 2 / (2 * mass) + k * positions[0] ** 2 / 2


# Used for control to solve_hamiltonian
def hamiltonian_orbit(positions, momenta, h_params):
    """
    Classical Newtonian one body potential in polar coordinates.
    :param positions:
    :param momenta:
    :param h_params: Tuple containing (Central mass, orbit mass) (G is forced normalized to 1)
    :return:
    """
    M, m = h_params
    return (momenta[0] ** 2 + momenta[1] ** 2 * positions[0] ** (-2)) / (2 * m) - M * m / positions[0]


def hamiltonian_einstein_infeld_hoffmann(positions, momenta, h_params):
    """
    Einstein-Infled-Hoffmann formula (insert Wolfram Alpha link)
    :param positions:
    :param momenta:
    :param h_params: Tuple containing (G, c, mass
    :return:
    """
    G, c, mass_1, mass_2 = h_params
    mass_1_inv = 1 / mass_1
    mass_2_inv = 1 / mass_2
    squared_momentum = momenta[0] ** 2 + momenta[1] ** 2
    distance = (positions[0] ** 2 + positions[1] ** 2) ** 0.5
    return (
            (squared_momentum / 2) * (mass_1_inv + mass_2_inv) - G * mass_1 * mass_2 / distance
            + squared_momentum ** 2 / (8 * c ** 2) * (mass_1_inv ** 3 + mass_2_inv ** 3)
            - G / (2 * c ** 2 * distance) * (3 * squared_momentum * (mass_2 * mass_1_inv + mass_1 * mass_2_inv)
                                             + 7 * squared_momentum
                                             + ((momenta[0] * positions[0] + momenta[1] * positions[1]) / distance) ** 2)
            + G ** 2 * mass_1 * mass_2 * (mass_1 + mass_2) / (2 * c ** 2 * distance ** 2)
    )


def hamiltonian_double_orbit(positions, momenta, h_params):
    """
    Classical two body hamiltonian in spherical coordinates
    :param positions:
    :param momenta:
    :param h_params: Tuple containing (mass_1, mass_2) G normalized to 1
    :return:
    """
    mass_1, mass_2 = h_params
    return ((momenta[0] ** 2 + momenta[1] ** 2 * positions[0] ** (-2)) / (2 * mass_1)
            + (momenta[2] ** 2 + momenta[3] ** 2 * positions[2] ** (-2)) / (2 * mass_2)
            - mass_1 * mass_2 / ((positions[0] * np.cos(positions[1]) - positions[2] * np.cos(positions[3])) ** 2
                       + (positions[0] * np.sin(positions[1]) - positions[2] * np.sin(positions[3])) ** 2) ** 0.5)


def hamiltonian_reduced_two_body(positions, momenta, h_params):
    """
    Reduced two body hamiltonian
    :param positions:
    :param momenta:
    :param h_params: Tuple containing (G, c, mass_1, mass_2)
    :return:
    """
    G, c, mass_1, mass_2 = h_params
    mass_2_inv = 1 / mass_2
    mass_1_inv = 1 / mass_1
    p2 = momenta[0] ** 2 + momenta[1] ** 2
    distance = (positions[0] ** 2 + positions[1] ** 2) ** 0.5
    return p2 * (mass_2_inv + mass_1_inv) / 2 - G * mass_2 * mass_1 / distance


def hamiltonian_two_body(positions, momenta, h_params):
    """
    Two body problem in Cartesian coordinates
    :param positions:
    :param momenta:
    :param h_params: Tuple containing (G, mass_1, mass_2)
    :return:
    """
    G, mass_1, mass_2 = h_params
    energy_1 = (momenta[0] ** 2 + momenta[1] ** 2) / (2 * mass_1)
    energy_2 = (momenta[2] ** 2 + momenta[3] ** 2) / (2 * mass_2)
    distance = ((positions[0] - positions[2]) ** 2 + (positions[1] - positions[3]) ** 2) ** 0.5
    return energy_1 + energy_2 - G * mass_1 * mass_2 / distance


def hamiltonian_newton_kinetic(positions, momenta, h_params):
    """
    Two body Newtonian kinetic energy.
    For consistency with other Hamiltonians, G is included in h_params
    :param positions:
    :param momenta:
    :param h_params: Tuple containing (G, mass_1, mass_2)
    :return: The two body Newtonian kinetic energy.
    """
    G, mass_1, mass_2 = h_params
    return (momenta[0]**2 + momenta[1]**2) / (2 * mass_1) + (momenta[2]**2 + momenta[3]**2) / (2 * mass_2)


def hamiltonian_sr_kinetic(positions, momenta, h_params):
    """
    Two body SR kinetic energy.
    For consistency with other Hamiltonians, G is included in h_params
    :param positions:
    :param momenta:
    :param h_params:  Tuple containing (G, mass_1, mass_2)
    :return: The two body SR kinetic energy (Rest mass energy isn't included).
    """
    G, mass_1, mass_2 = h_params
    energy_1 = (mass_1**2 + momenta[0]**2 + momenta[1]**2)**0.5 - mass_1
    energy_2 = (mass_2**2 + momenta[2]**2 + momenta[3]**2)**0.5 - mass_2
    return energy_1 + energy_2


def hamiltonian_sr_newton_pot(positions, momenta, h_params):
    """
    SR Kinetic energy with a classical Newtonian potential energy. (Used to evaluate precession with a 1/r potential)
    :param positions:
    :param momenta:
    :param h_params: Tuple containing (G, mass_1, mass_2)
    :return:
    """
    G, mass_1, mass_2 = h_params
    energy_1 = (mass_1**2 + momenta[0]**2 + momenta[1]**2)**0.5
    energy_2 = (mass_2**2 + momenta[2]**2 + momenta[3]**2)**0.5
    distance = ((positions[0] - positions[2])**2 + (positions[1] - positions[3])**2)**0.5
    return energy_1 + energy_2 - G * mass_2 * mass_1 / distance


def hamiltonian_post_minkowski1(positions, momenta, h_params):
    """
    PM1 Hamiltonian.
    :param positions: Positions of the particles. First two are for the first object, last 2 are for second object.
    :param momenta: First two for the first particle. Last 2 for the last object. Need momenta[0] = - momenta[2],
    momenta[1] = - momenta[3]
    :param h_params: Newton constant (G), Speed of light (c), mass of first object, mass of second object.
    :return: Numerical value for the first order post minkowski approximation.
    """
    G, mass_1, mass_2 = h_params
    energy_1 = (mass_1**2 + momenta[0]**2 + momenta[1]**2)**0.5
    energy_2 = (mass_2**2 + momenta[2]**2 + momenta[3]**2)**0.5
    momentum_abs_sq = - momenta[0] * momenta[2] - momenta[1] * momenta[3]
    distance = ((positions[0] - positions[2])**2 + (positions[1] - positions[3])**2) ** 0.5
    c_1 = mass_1**2 * mass_2**2 - 2 * (energy_1 * energy_2 + momentum_abs_sq)**2

    return energy_1 + energy_2 + (G * c_1) / (energy_1 * energy_2 * distance) - mass_1 - mass_2


def hamiltonian_post_minkowski2(positions, momenta, h_params):
    """
    PM2 Hamiltonian
    :param positions:  Positions of the particles. First two are for the first object, last 2 are for second object.
    :param momenta: First two for the first particle. Last 2 for the last object. Need momenta[0] = - momenta[2],
    momenta[1] = - momenta[3]
    :param h_params: Newton constant (G), Speed of light (c), mass of first object, mass of second object.
    :return: Numerical value for the second order post minkowski approximation.
    """
    G, mass_1, mass_2 = h_params
    energy_1 = (mass_1 ** 2 + momenta[0] ** 2 + momenta[1] ** 2) ** 0.5
    energy_2 = (mass_2 ** 2 + momenta[2] ** 2 + momenta[3] ** 2) ** 0.5
    dotted_momenta = energy_1 * energy_2 - momenta[0] * momenta[2] - momenta[1] * momenta[3]
    distance = ((positions[0] - positions[2])**2 + (positions[1] - positions[3])**2) ** 0.5
    c_1 = mass_1 ** 2 * mass_2 ** 2 - 2 * (energy_1 * energy_2 - momenta[0] * momenta[2] - momenta[1] * momenta[3]) ** 2
    c_mass_1 = 3 * mass_1 ** 2 * (mass_1 ** 2 * mass_2 ** 2 - 5 * dotted_momenta ** 2)
    c_mass_2 = 3 * mass_2 ** 2 * (mass_1 ** 2 * mass_2 ** 2 - 5 * dotted_momenta ** 2)
    energy_total = energy_1 + energy_2
    xi = energy_1 * energy_2 / (energy_total ** 2)
    return (energy_1 + energy_2 - mass_1 - mass_2
            + (G * c_1) / (energy_1 * energy_2 * distance)
            + G ** 2 / (distance ** 2 * energy_1 * energy_2) * ((c_mass_1 / mass_1 + c_mass_2 / mass_2) / 4
                                                                + (c_1 ** 2 * (xi - 1) / (
                                2 * energy_total ** 3 * xi ** 2)
                                                                   - 4 * c_1 * dotted_momenta / (energy_total * xi))))


def hamiltonian_post_minkowski3(positions, momenta, h_params):
    """
    PM3 Hamiltonian - No Radiation Reaction term is included
    :param positions:  Positions of the particles. First two are for the first object, last 2 are for second object.
    :param momenta: First two for the first particle. Last 2 for the last object. Need momenta[0] = - momenta[2],
    momenta[1] = - momenta[3]
    :param h_params: Newton constant (G), Speed of light (c), mass of first object, mass of second object.
    :return: Numerical value for the third order post minkowski approximation.
    """
    G, mass_1, mass_2 = h_params
    m = mass_1 + mass_2
    energy_1 = (mass_1 ** 2 + momenta[0] ** 2 + momenta[1] ** 2) ** 0.5
    energy_2 = (mass_2 ** 2 + momenta[2] ** 2 + momenta[3] ** 2) ** 0.5
    E = energy_1 + energy_2
    gamma = E / m
    dotted_momenta = energy_1 * energy_2 - momenta[0] * momenta[2] - momenta[1] * momenta[3]
    sigma = dotted_momenta / (mass_1 * mass_2)
    distance = ((positions[0] - positions[2]) ** 2 + (positions[1] - positions[3]) ** 2) ** 0.5
    xi = energy_1 * energy_2 / (E ** 2)
    nu = mass_1 * mass_2 / m ** 2
    c_1 = nu ** 2 * m ** 2 / (gamma ** 2 * xi) * (1 - 2 * sigma ** 2)
    c_2 = ((nu ** 2 * m ** 3 / (gamma ** 2 * xi))
           * (3 / 4 * (1 - 5 * sigma ** 2)
              - 4 * nu * sigma * (1 - 2 * sigma ** 2) / (gamma * xi)
              - nu ** 2 * (1 - xi) * (1 - 2 * sigma ** 2) ** 2 / (2 * gamma ** 3 * xi ** 2)))
    c_3 = ((nu ** 2 * m ** 4 / (gamma ** 2 * xi))
           * (1 / 12 * (3 - 6 * nu + 206 * nu * sigma - 54 * sigma ** 2 + 108 * nu * sigma ** 2 + 4 * nu * sigma ** 3)
              - (4 * nu * (3 + 12 * sigma ** 2 - 4 * sigma ** 4) * np.log(
                        ((sigma - 1) / 2) ** 0.5 + ((sigma + 1) / 2) ** 0.5)
                 / (sigma ** 2 - 1) ** 0.5)
              - 3 * nu * gamma * (1 - 2 * sigma ** 2) * (1 - 5 * sigma ** 2) / (2 * (1 + gamma) * (1 + sigma))
              - 3 * nu * sigma * (7 - 20 * sigma ** 2) / (2 * gamma * xi)
              - (nu ** 2 * (
                                3 + 8 * gamma - 3 * xi - 15 * sigma ** 2 - 80 * gamma * sigma ** 2 + 15 * xi * sigma ** 2) * (
                             1 - 2 * sigma ** 2)
                 / (4 * gamma ** 3 * xi ** 2))
              + 2 * nu ** 3 * (3 - 4 * xi) * sigma * (1 - 2 * sigma ** 2) ** 2 / (gamma ** 4 * xi ** 3)
              + nu ** 4 * (1 - 2 * xi) * (1 - 2 * sigma ** 2) ** 3 / (2 * gamma ** 6 * xi ** 4)))
    return (energy_1 + energy_2 - mass_1 - mass_2
            + c_1 * (G / distance) + c_2 * (G / distance) ** 2 + c_3 * (G / distance) ** 3)