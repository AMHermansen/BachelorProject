import numpy as np
from Utilities import split_position_momentum, get_gamma_factor, get_total_angular_momentum


def scattering_classical(solution, h_params):
    G, mass_1, mass_2 = h_params
    reduced_mass = (mass_1 ** (-1) + mass_2 ** (-1)) ** (-1)
    b = solution.y[1, 0] - solution.y[3, 0]
    v_infinity = (solution.y[4, 0] / mass_1 - solution.y[6, 0] / mass_2)
    return 2 * np.arctan(G * (mass_1 + mass_2) / (v_infinity ** 2 * b))


def _get_constants(solution, hamiltonian, h_params, position_pair_coordinates, momentum_pair_coordinates):
    G, mass_1, mass_2 = h_params
    positions, momenta = split_position_momentum(solution=solution)
    no_mass_energy = hamiltonian(positions=positions, momenta=momenta, h_params=h_params)[0]
    E = no_mass_energy + mass_1 + mass_2
    gamma_factor = get_gamma_factor(positions=positions, momenta=momenta, hamiltonian=hamiltonian, h_params=h_params)
    L = get_total_angular_momentum(solution=solution,
                                   position_pair_coordinates=position_pair_coordinates,
                                   momentum_pair_coordinates=momentum_pair_coordinates)[0]
    mu = (mass_1 ** (-1) + mass_2 ** (-1)) ** (-1)
    M = mass_1 + mass_2
    nu = mu / M
    Gamma = E / M  # Not to confuse with gamma_factor
    return E, gamma_factor, mu, M, nu, Gamma, L


def _get_f1(E, gamma_factor, mu, M, nu, Gamma):
    return 2 * mu**2 * M * (2 * gamma_factor**2 - 1) / Gamma


def _get_f2(E, gamma_factor, mu, M, nu, Gamma):
    return 3/2 * mu**2 * M**2 * (5 * gamma_factor**2 - 1) / Gamma


def _get_f3(E, gamma_factor, mu, M, nu, Gamma):
    return mu**2 * M**3 * (
           Gamma * (18 * gamma_factor**2 - 1) / 2 - 4 * nu * gamma_factor * (14*gamma_factor**2 + 25) / (3 * Gamma)
           + 3/2 * (Gamma - 1) / (gamma_factor**2 - 1) * (2 * gamma_factor**2 - 1) * (5 * gamma_factor**2 - 1)
           - 8 * nu * (4*gamma_factor**4 - 12*gamma_factor**2 - 3) / (Gamma * (gamma_factor**2 - 1)**0.5)
               * np.log(((gamma_factor - 1) / 2)**0.5 + ((gamma_factor + 1) / 2)**0.5)
    )


def get_all_f(E, gamma_factor, mu, M, nu, Gamma):
    return (_get_f1(E, gamma_factor, mu, M, nu, Gamma),
            _get_f2(E, gamma_factor, mu, M, nu, Gamma),
            _get_f3(E, gamma_factor, mu, M, nu, Gamma),
            )


def scattering_pm1(solution, hamiltonian, h_params, position_pair_coordinates, momentum_pair_coordinates):
    """
    Verified formula. 05/07/22
    :param solution: The solution for which the theoretical scattering is to be computed. (Need the initial values)
    :param hamiltonian: The Hamiltonian for which the theoretical scattering is to be computed. Should be compatible be
    compatible with solution.
    :param h_params: Parameters for Hamiltonian
    :param position_pair_coordinates: List of pairs of (x,y) position coordinates.
    :param momentum_pair_coordinates: List of pairs of (x,y) momentum coordinates
    :return: Theoretical post Minkowskian scattering angle to first order.
    """
    positions, momenta = split_position_momentum(solution=solution)
    gamma_factor = get_gamma_factor(positions=positions, momenta=momenta, hamiltonian=hamiltonian, h_params=h_params)
    total_angular_momentum = get_total_angular_momentum(solution=solution,
                                                        position_pair_coordinates=position_pair_coordinates,
                                                        momentum_pair_coordinates=momentum_pair_coordinates)[0]
    G, mass_1, mass_2 = h_params
    reduced_mass = (mass_1 ** (-1) + mass_2 ** (-1)) ** (-1)
    mass_sum = mass_1 + mass_2
    chi_1 = (2 * gamma_factor ** 2 - 1) / (gamma_factor ** 2 - 1) ** 0.5
    return (chi_1 * (G * mass_sum * reduced_mass / total_angular_momentum)) * 2


def scattering_pm2(solution, hamiltonian, h_params, position_pair_coordinates, momentum_pair_coordinates):
    """
    Verified formula. 05/07/22
    :param solution: The solution for which the theoretical scattering angle is to be compouted.
    :param hamiltonian: The Hamiltonian for which the theoretical scattering angle is to be computed.
    :param h_params: Parameters of the Hamiltonian.
    :param position_pair_coordinates: List of (x,y) position coordinate pairs.
    :param momentum_pair_coordinates: List of (x,y) momentum coordinate pairs
    :return: Post Minkowskian scattering angle to 2nd order
    """
    G, mass_1, mass_2 = h_params
    positions, momenta = split_position_momentum(solution=solution)
    no_mass_energy = hamiltonian(positions=positions, momenta=momenta, h_params=h_params)[0]
    total_energy = no_mass_energy + mass_1 + mass_2
    gamma_factor = get_gamma_factor(positions=positions, momenta=momenta, hamiltonian=hamiltonian, h_params=h_params)
    total_angular_momentum = get_total_angular_momentum(solution=solution,
                                                        position_pair_coordinates=position_pair_coordinates,
                                                        momentum_pair_coordinates=momentum_pair_coordinates)[0]
    reduced_mass = (mass_1 ** (-1) + mass_2 ** (-1)) ** (-1)
    mass_sum = mass_1 + mass_2
    scattering_sum_factor = G * mass_sum * reduced_mass / total_angular_momentum
    chi_1 = (2 * gamma_factor**2 - 1) / (gamma_factor**2 - 1)**0.5
    chi_2 = (3 * np.pi / 8) * (5 * gamma_factor**2 - 1) / (total_energy / mass_sum)
    return (chi_1 * scattering_sum_factor + chi_2 * scattering_sum_factor**2) * 2


def scattering_pm3(solution, hamiltonian, h_params, position_pair_coordinates, momentum_pair_coordinates):
    """
    Not verified
    :param solution: The solution for which the theoretical scattering angle is to be compouted.
    :param hamiltonian: The Hamiltonian for which the theoretical scattering angle is to be computed.
    :param h_params: Parameters of the Hamiltonian.
    :param position_pair_coordinates: List of (x,y) position coordinate pairs.
    :param momentum_pair_coordinates: List of (x,y) momentum coordinate pairs
    :return: Post Minkowskian scattering angle to 3rd order
    """
    G, mass_1, mass_2 = h_params
    E, gamma_factor, mu, M, nu, Gamma, L = _get_constants(solution, hamiltonian, h_params,
                                                          position_pair_coordinates, momentum_pair_coordinates)
    scattering_sum_factor = G / L

    p_0 = mu / Gamma * (gamma_factor**2 - 1)**0.5  # Observe formula from (1901.07102) gives p_0**2
    # f_1 = 2 * mu**2 * M * (2 * gamma_factor**2 - 1) / Gamma
    # f_2 = 3/2 * mu**2 * M**2 * (5 * gamma_factor**2 - 1) / Gamma
    # f_3 = mu**2 * M**3 * (
    #     Gamma * (18 * gamma_factor**2 - 1) / 2 - 4 * nu * gamma_factor * (14*gamma_factor**2 + 25) / (3 * Gamma)
    #     + 3/2 * (Gamma - 1) / (gamma_factor**2 - 1) * (2 * gamma_factor**2 - 1) * (5 * gamma_factor**2 - 1)
    #     - 8 * nu * (4*gamma_factor**4 - 12*gamma_factor**2 - 3) / (Gamma * (gamma_factor**2 - 1)**0.5)
    #         * np.log(((gamma_factor - 1) / 2)**0.5 + ((gamma_factor + 1) / 2)**0.5)
    # )
    f_1, f_2, f_3 = get_all_f(E, gamma_factor, mu, M, nu, Gamma)
    return 2 * (scattering_sum_factor * (f_1 / 2 * p_0) + scattering_sum_factor**2 * (np.pi * f_2 / 4)
                + scattering_sum_factor**3 * (p_0 * f_3 + f_1 * f_2 / (2 * p_0) - f_1**3 / (24 * p_0**3)))


def theoretical_perihelion_shift(solution, hamiltonian, h_params, position_pair_coordinates, momentum_pair_coordinates):
    positions, momenta = split_position_momentum(solution=solution)
    G, mass_1, mass_2 = h_params
    no_mass_energy = hamiltonian(positions=positions, momenta=momenta, h_params=h_params)[0]
    total_energy = no_mass_energy + mass_1 + mass_2
    gamma_factor = get_gamma_factor(positions=positions, momenta=momenta, hamiltonian=hamiltonian, h_params=h_params)
    mass_sum = mass_1 + mass_2
    reduced_mass = (mass_1**(-1) + mass_2**(-1)) ** (-1)
    total_angular_momentum = get_total_angular_momentum(solution=solution,
                                                        position_pair_coordinates=position_pair_coordinates,
                                                        momentum_pair_coordinates=momentum_pair_coordinates)[0]
    return (3 * np.pi
            * (G * mass_sum * reduced_mass / total_angular_momentum)**2
            * (total_energy / mass_sum) * (5 * gamma_factor**2 - 1))


