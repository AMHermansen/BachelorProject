import numpy as np
import einsteinpy.utils.dual as dual
from einsteinpy.utils.dual import _deriv
from scipy.integrate import solve_ivp
from scipy.signal import argrelextrema


def convert_to_dual(positions, momenta):
    positions_dual = list()
    momenta_dual = list()
    for position in positions:
        positions_dual.append(dual.DualNumber(position, 0.0))

    for momentum in momenta:
        momenta_dual.append(dual.DualNumber(momentum, 0.0))
    return positions_dual, momenta_dual


def _diff_hamiltonian_q(hamiltonian, positions, momenta, wrt_q, h_params):
    positions_dual, momenta_dual = convert_to_dual(positions=positions, momenta=momenta)
    positions_dual[wrt_q].deriv = 1.0
    return _deriv(lambda q: hamiltonian(positions_dual, momenta_dual, h_params), positions[wrt_q])


def _diff_hamiltonian_p(hamiltonian, positions, momenta, wrt_p, h_params):
    positions_dual, momenta_dual = convert_to_dual(positions=positions, momenta=momenta)
    momenta_dual[wrt_p].deriv = 1.0
    return _deriv(lambda q: hamiltonian(positions_dual, momenta_dual, h_params=h_params), momenta[wrt_p])


def _get_hamilton_eq(hamiltonian, positions, momenta, h_params):
    assert len(positions) == len(momenta)
    hamilton_p_deriv = list()
    hamilton_q_deriv = list()
    for i in range(len(positions)):
        hamilton_p_deriv.append(_diff_hamiltonian_p(hamiltonian=hamiltonian,
                                                    positions=positions, momenta=momenta,
                                                    wrt_p=i, h_params=h_params))
    for i in range(len(momenta)):
        hamilton_q_deriv.append(- _diff_hamiltonian_q(hamiltonian=hamiltonian,
                                                      positions=positions, momenta=momenta,
                                                      wrt_q=i, h_params=h_params))

    return np.concatenate([np.array(hamilton_p_deriv), np.array(hamilton_q_deriv)])


def hamiltonian_system(t, coordinates, hamiltonian, h_params):
    positions = coordinates[:len(coordinates) // 2]
    momenta = coordinates[len(coordinates) // 2:]
    return _get_hamilton_eq(hamiltonian=hamiltonian, positions=positions, momenta=momenta, h_params=h_params)


def solve_hamiltonian(hamiltonian, t_span, initial, h_params, **kwargs):
    """"
    :param hamiltonian: The hamiltonian that is to solved. Should take the stacked position-momentum
    coordinates as input.
    :param t_span: The time span for which the solution is calculated.
    :param initial: The initial position in phase-space, i.e. a stacked position-momentum np.array
    :param h_params: Parameters for the hamiltonian examples could be masses.
    :param kwargs: kwargs solve_ivp
    :return:
    """
    return(solve_ivp(lambda t, coord: hamiltonian_system(t, coord, hamiltonian=hamiltonian, h_params=h_params),
                     t_span=t_span, y0=initial, **kwargs))


def get_solutions(hamiltonians, **kwargs):
    return [solve_hamiltonian(hamiltonian, **kwargs) for hamiltonian in hamiltonians]


def get_angular_momentum(positions, momenta):
    x = positions[0, :]
    y = positions[1, :]
    px = momenta[0, :]
    py = momenta[1, :]
    return x * py - y * px


def get_total_angular_momentum(solution, position_pair_coordinates, momentum_pair_coordinates):
    total_angular_momentum = 0
    for position_coordinates, momentum_coordinates in zip(position_pair_coordinates, momentum_pair_coordinates):
        position = np.array([solution.y[position_coordinates[0], :], solution.y[position_coordinates[1], :]])
        momentum = np.array([solution.y[momentum_coordinates[0], :], solution.y[momentum_coordinates[1], :]])
        total_angular_momentum += get_angular_momentum(positions=position, momenta=momentum)
    return total_angular_momentum


def get_gamma_factor(positions, momenta, hamiltonian, h_params):
    """
    Formula verified. 05/07/22
    :param positions: Position for the hamiltonian
    :param momenta: momentum for the hamiltonian
    :param hamiltonian: The hamiltonian for which the gamma factor is to be calculated. (Used for total energy)
    :param h_params: Parameters for the hamiltonian
    :return: Relativistic gamma_factor.
    """
    G, mass_1, mass_2 = h_params
    no_mass_energy = hamiltonian(positions=positions, momenta=momenta, h_params=h_params)[0]
    energy = no_mass_energy + mass_1 + mass_2
    return (energy ** 2 - mass_1 ** 2 - mass_2 ** 2) / (2 * mass_1 * mass_2)


def split_position_momentum(solution):
    number_of_coordinates = len(solution.y[:, 0]) // 2  # Position + momentum coordinates // 2
    return solution.y[:number_of_coordinates, :], solution.y[number_of_coordinates:, :]


def get_perihelion_shift(solution, shift=0):  # Moved to Utilities
    """

    :param solution: The solution, for which the perihelion shift is to be measured. Assumes x,y for the shift are on
    (0, 1)
    :param shift: Since the solution is discretized, the maximum of the distance, might not be the correct maximum.
    Shift lets you move to neighbouring points. (-1, 0, or 1) is recommended
    :return:
    """
    r2 = solution.y[0, :] ** 2 + solution.y[1, :] ** 2
    phi = np.arctan(solution.y[1, :] / solution.y[0, :])
    perihelion = argrelextrema(r2, np.greater)
    perihelion_shifted = [pos + shift for pos in perihelion]
    return phi[perihelion_shifted]


def scattering_angle(solution, position_coordinates):
    initial_x = solution.y[position_coordinates[0], 0]
    initial_y = solution.y[position_coordinates[1], 0]
    final_x = solution.y[position_coordinates[0], -1]
    final_y = solution.y[position_coordinates[1], -1]
    initial_angle = np.arctan2(initial_y, initial_x)
    final_angle = np.arctan2(final_y, final_x)
    return initial_angle, final_angle + 2 * np.pi


def get_scattering(angle_tuple):
    return (angle_tuple[1] - angle_tuple[0]) - np.pi


def time_averaged_mean(array, time):
    number_of_elements = len(array)
    total_time = time[-1] - time[0]
    indices = np.arange(number_of_elements - 1)
    time_averaged_values = array[indices] * (time[indices + 1] - time[indices]) / total_time
    return np.sum(time_averaged_values)


