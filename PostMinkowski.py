import numpy as np
import matplotlib.pyplot as plt
from Hamiltonians import (hamiltonian_post_minkowski1, hamiltonian_post_minkowski2, hamiltonian_post_minkowski3,
                          hamiltonian_two_body, hamiltonian_sr_kinetic, hamiltonian_newton_kinetic)
from Utilities import (get_solutions, get_angular_momentum, get_perihelion_shift, scattering_angle, get_scattering,
                       time_averaged_mean)  # Might use later
from GeneratePlots import (generate_orbit_plot, generate_energy_plot, generate_split_energy_plot,
                           generate_angular_momentum_plot)
from Formulas import scattering_classical, scattering_pm1, scattering_pm2, scattering_pm3, theoretical_perihelion_shift
from Hamiltonians import hamiltonian_sr_newton_pot  # Maybe used in future


# def hamiltonian_post_minkowski1(positions, momenta, h_params):  # Moved into Hamiltonians
#     """
#     :param positions: Positions of the particles. First two are for the first object, last 2 are for second object.
#     :param momenta: First two for the first particle. Last 2 for the last object. Need momenta[0] = - momenta[2],
#     momenta[1] = - momenta[3]
#     :param h_params: Newton constant (G), Speed of light (c), mass of first object, mass of second object.
#     :return: Numerical value for the first order post minkowski approximation.
#     """
#     G, mass_1, mass_2 = h_params
#     energy_1 = (mass_1**2 + momenta[0]**2 + momenta[1]**2)**0.5
#     energy_2 = (mass_2**2 + momenta[2]**2 + momenta[3]**2)**0.5
#     momentum_abs_sq = - momenta[0] * momenta[2] - momenta[1] * momenta[3]
#     distance = ((positions[0] - positions[2])**2 + (positions[1] - positions[3])**2) ** 0.5
#     c_1 = mass_1**2 * mass_2**2 - 2 * (energy_1 * energy_2 + momentum_abs_sq)**2
#
#     return energy_1 + energy_2 + (G * c_1) / (energy_1 * energy_2 * distance) - mass_1 - mass_2
#
#
# def hamiltonian_post_minkowski2(positions, momenta, h_params):  # Moved into Hamiltonians
#     """
#     :param positions:  Positions of the particles. First two are for the first object, last 2 are for second object.
#     :param momenta: First two for the first particle. Last 2 for the last object. Need momenta[0] = - momenta[2],
#     momenta[1] = - momenta[3]
#     :param h_params: Newton constant (G), Speed of light (c), mass of first object, mass of second object.
#     :return: Numerical value for the second order post minkowski approximation.
#     """
#     G, mass_1, mass_2 = h_params
#     energy_1 = (mass_1 ** 2 + momenta[0] ** 2 + momenta[1] ** 2) ** 0.5
#     energy_2 = (mass_2 ** 2 + momenta[2] ** 2 + momenta[3] ** 2) ** 0.5
#     dotted_momenta = energy_1 * energy_2 - momenta[0] * momenta[2] - momenta[1] * momenta[3]
#     distance = ((positions[0] - positions[2])**2 + (positions[1] - positions[3])**2) ** 0.5
#     c_1 = mass_1 ** 2 * mass_2 ** 2 - 2 * (energy_1 * energy_2 - momenta[0] * momenta[2] - momenta[1] * momenta[3]) ** 2
#     c_mass_1 = 3 * mass_1 ** 2 * (mass_1 ** 2 * mass_2 ** 2 - 5 * dotted_momenta ** 2)
#     c_mass_2 = 3 * mass_2 ** 2 * (mass_1 ** 2 * mass_2 ** 2 - 5 * dotted_momenta ** 2)
#     energy_total = energy_1 + energy_2
#     xi = energy_1 * energy_2 / (energy_total ** 2)
#     return (energy_1 + energy_2 - mass_1 - mass_2
#             + (G * c_1) / (energy_1 * energy_2 * distance)
#             + G ** 2 / (distance ** 2 * energy_1 * energy_2) * ((c_mass_1 / mass_1 + c_mass_2 / mass_2) / 4
#                                                                 + (c_1 ** 2 * (xi - 1) / (
#                                 2 * energy_total ** 3 * xi ** 2)
#                                                                    - 4 * c_1 * dotted_momenta / (energy_total * xi))))
#
#
# def hamiltonian_post_minkowski3(positions, momenta, h_params):  # Moved into Hamiltonians
#     G, mass_1, mass_2 = h_params
#     m = mass_1 + mass_2
#     energy_1 = (mass_1 ** 2 + momenta[0] ** 2 + momenta[1] ** 2) ** 0.5
#     energy_2 = (mass_2 ** 2 + momenta[2] ** 2 + momenta[3] ** 2) ** 0.5
#     E = energy_1 + energy_2
#     gamma = E / m
#     dotted_momenta = energy_1 * energy_2 - momenta[0] * momenta[2] - momenta[1] * momenta[3]
#     sigma = dotted_momenta / (mass_1 * mass_2)
#     distance = ((positions[0] - positions[2]) ** 2 + (positions[1] - positions[3]) ** 2) ** 0.5
#     xi = energy_1 * energy_2 / (E ** 2)
#     nu = mass_1 * mass_2 / m ** 2
#     c_1 = nu ** 2 * m ** 2 / (gamma ** 2 * xi) * (1 - 2 * sigma ** 2)
#     c_2 = ((nu ** 2 * m ** 3 / (gamma ** 2 * xi))
#            * (3 / 4 * (1 - 5 * sigma ** 2)
#               - 4 * nu * sigma * (1 - 2 * sigma ** 2) / (gamma * xi)
#               - nu ** 2 * (1 - xi) * (1 - 2 * sigma ** 2) ** 2 / (2 * gamma ** 3 * xi ** 2)))
#     c_3 = ((nu ** 2 * m ** 4 / (gamma ** 2 * xi))
#            * (1 / 12 * (3 - 6 * nu + 206 * nu * sigma - 54 * sigma ** 2 + 108 * nu * sigma ** 2 + 4 * nu * sigma ** 3)
#               - (4 * nu * (3 + 12 * sigma ** 2 - 4 * sigma ** 4) * np.log(
#                         ((sigma - 1) / 2) ** 0.5 + ((sigma + 1) / 2) ** 0.5)
#                  / (sigma ** 2 - 1) ** 0.5)
#               - 3 * nu * gamma * (1 - 2 * sigma ** 2) * (1 - 5 * sigma ** 2) / (2 * (1 + gamma) * (1 + sigma))
#               - 3 * nu * sigma * (7 - 20 * sigma ** 2) / (2 * gamma * xi)
#               - (nu ** 2 * (
#                                 3 + 8 * gamma - 3 * xi - 15 * sigma ** 2 - 80 * gamma * sigma ** 2 + 15 * xi * sigma ** 2) * (
#                              1 - 2 * sigma ** 2)
#                  / (4 * gamma ** 3 * xi ** 2))
#               + 2 * nu ** 3 * (3 - 4 * xi) * sigma * (1 - 2 * sigma ** 2) ** 2 / (gamma ** 4 * xi ** 3)
#               + nu ** 4 * (1 - 2 * xi) * (1 - 2 * sigma ** 2) ** 3 / (2 * gamma ** 6 * xi ** 4)))
#     return (energy_1 + energy_2 - mass_1 - mass_2
#             + c_1 * (G / distance) + c_2 * (G / distance) ** 2 + c_3 * (G / distance) ** 3)
#
#
# def hamiltonian_two_body(positions, momenta, h_params):  # Moved into Hamiltonians
#     G, mass_1, mass_2 = h_params
#     energy_1 = (momenta[0] ** 2 + momenta[1] ** 2) / (2 * mass_1)
#     energy_2 = (momenta[2] ** 2 + momenta[3] ** 2) / (2 * mass_2)
#     distance = ((positions[0] - positions[2]) ** 2 + (positions[1] - positions[3]) ** 2) ** 0.5
#     return energy_1 + energy_2 - G * mass_1 * mass_2 / distance
#
#
# def hamiltonian_sr_newton_pot(positions, momenta, h_params):  # Moved into Hamiltonians
#     G, mass_1, mass_2 = h_params
#     energy_1 = (mass_1**2 + momenta[0]**2 + momenta[1]**2)**0.5
#     energy_2 = (mass_2**2 + momenta[2]**2 + momenta[3]**2)**0.5
#     distance = ((positions[0] - positions[2])**2 + (positions[1] - positions[3])**2)**0.5
#     return energy_1 + energy_2 - G * mass_2 * mass_1 / distance
#
#
# def hamiltonian_sr_kinetic(positions, momenta, h_params):  # Moved into Hamiltonians
#     G, mass_1, mass_2 = h_params
#     energy_1 = (mass_1**2 + momenta[0]**2 + momenta[1]**2)**0.5 - mass_1
#     energy_2 = (mass_2**2 + momenta[2]**2 + momenta[3]**2)**0.5 - mass_2
#     return energy_1 + energy_2
#
#
# def hamiltonian_newton_kinetic(positions, momenta, h_params):  # Moved into Hamiltonians
#     G, mass_1, mass_2 = h_params
#     return (momenta[0]**2 + momenta[1]**2) / (2 * mass_1) + (momenta[2]**2 + momenta[3]**2) / (2 * mass_2)


# def get_solutions(hamiltonians, **kwargs):  # Moved to Utilities
#     return [solve_hamiltonian(hamiltonian, **kwargs) for hamiltonian in hamiltonians]


# def generate_orbit_plot(solutions, legends, x_indices, y_indices,
#                         x_label='x coordinate [au]', y_label='y coordinate [au]'):
#     for solution, labels in zip(solutions, legends):
#         for x_coordinate, y_coordinate, label in zip(x_indices, y_indices, labels):
#             plt.plot(solution.y[x_coordinate, :], solution.y[y_coordinate, :], label=label)
#     plt.xlabel(x_label)
#     plt.ylabel(y_label)
#     plt.legend()


# def generate_energy_plot(solutions, hamiltonians, all_h_params, legends,
#                          x_label='Time [au]', y_label='Energy [au]'):
#     for solution, hamiltonian, h_params, label in zip(solutions, hamiltonians, all_h_params, legends):
#         number_of_coordinates = len(solution.y[:, 0]) // 2  # Number of position/momentum coordinates. Half the total number
#         position_coordinates = solution.y[:number_of_coordinates, :]
#         momentum_coordinates = solution.y[number_of_coordinates:, :]
#         initial_energy = hamiltonian(solution.y[:number_of_coordinates, 0],
#                                      solution.y[number_of_coordinates:, 0],
#                                      h_params)
#         normalized_energy = hamiltonian(position_coordinates, momentum_coordinates, h_params) / initial_energy
#         plt.plot(solution.t, normalized_energy, label=label)
#     plt.xlabel(x_label)
#     plt.ylabel(y_label)
#     plt.legend()


# def generate_split_energy_plot(solutions, hamiltonians, kinetic_energies, all_h_params, legends,
#                                x_label='Time [au]', y_label='Energy [au]', plot_kinetic=True, plot_potential=True):
#     all_normalized_kinetic = list()
#     all_normalized_potential = list()
#     all_times = list()
#     for solution, hamiltonian, kinetic_energy, h_params, label in zip(solutions, hamiltonians, kinetic_energies, all_h_params, legends):
#         number_of_coordinates = len(solution.y[:, 0]) // 2
#         position_coordinates, momentum_coordinates = split_position_momentum(solution=solution)
#         initial_energy = hamiltonian(solution.y[:number_of_coordinates, 0],
#                                      solution.y[number_of_coordinates:, 0],
#                                      h_params)
#         normalized_energy = hamiltonian(position_coordinates, momentum_coordinates, h_params) / np.abs(initial_energy)
#         normalized_kinetic = kinetic_energy(position_coordinates, momentum_coordinates, h_params) / np.abs(initial_energy)
#         normalized_potential = normalized_energy - normalized_kinetic
#         plt.plot(solution.t, normalized_energy, label=f'{label} Total energy')
#         if plot_kinetic:
#             plt.plot(solution.t, normalized_kinetic, label=f'{label} Kinetic energy')
#         if plot_potential:
#             plt.plot(solution.t, normalized_potential, label=f'{label} Potential energy')
#         all_normalized_kinetic.append(normalized_kinetic)
#         all_normalized_potential.append(normalized_potential)
#         all_times.append(solution.t)
#     plt.xlabel(x_label)
#     plt.ylabel(y_label)
#     plt.legend()
#     return all_normalized_kinetic, all_normalized_potential, all_times


# def generate_angular_momentum_plot(solutions, legends, position_pair_coordinates, momentum_pair_coordinates,
#                                    x_label='Time [au]', y_label='Angular momentum [au]'):
#     """
#
#     :param solutions: Solutions from solve_hamiltonian
#     :param legends: Tuple of legends for the plots
#     :param position_pair_coordinates: Position pairs, when multiple x-/y-coordinates
#     :param momentum_pair_coordinates: Momentum pairs, when multiple x-/y-momenta
#     :param x_label:
#     :param y_label:
#     :return:
#     """
#     for solution, label, positions, momenta in zip(solutions,
#                                                    legends,
#                                                    position_pair_coordinates,
#                                                    momentum_pair_coordinates):
#         total_angular_momentum = get_total_angular_momentum(solution=solution,
#                                                             position_pair_coordinates=positions,
#                                                             momentum_pair_coordinates=momenta)
#         plt.plot(solution.t, total_angular_momentum / total_angular_momentum[0], label=label)
#     plt.xlabel(x_label)
#     plt.ylabel(y_label)
#     plt.legend()


# def get_angular_momentum(positions, momenta):
#     x = positions[0, :]
#     y = positions[1, :]
#     px = momenta[0, :]
#     py = momenta[1, :]
#     return x * py - y * px
#
#
# def get_total_angular_momentum(solution, position_pair_coordinates, momentum_pair_coordinates):
#     total_angular_momentum = 0
#     for position_coordinates, momentum_coordinates in zip(position_pair_coordinates, momentum_pair_coordinates):
#         position = np.array([solution.y[position_coordinates[0], :], solution.y[position_coordinates[1], :]])
#         momentum = np.array([solution.y[momentum_coordinates[0], :], solution.y[momentum_coordinates[1], :]])
#         total_angular_momentum += get_angular_momentum(positions=position, momenta=momentum)
#     return total_angular_momentum
#
#
# def get_gamma_factor(positions, momenta, hamiltonian, h_params):
#     """
#     Formula verified. 05/07/22
#     :param positions: Position for the hamiltonian
#     :param momenta: momentum for the hamiltonian
#     :param hamiltonian: The hamiltonian for which the gamma factor is to be calculated. (Used for total energy)
#     :param h_params: Parameters for the hamiltonian
#     :return: Relativistic gamma_factor.
#     """
#     G, mass_1, mass_2 = h_params
#     no_mass_energy = hamiltonian(positions=positions, momenta=momenta, h_params=h_params)[0]
#     energy = no_mass_energy + mass_1 + mass_2
#     return (energy ** 2 - mass_1 ** 2 - mass_2 ** 2) / (2 * mass_1 * mass_2)
#
#
# def split_position_momentum(solution):
#     number_of_coordinates = len(solution.y[:, 0]) // 2  # Position + momentum coordinates // 2
#     return solution.y[:number_of_coordinates, :], solution.y[number_of_coordinates:, :]


# def scattering_classical(solution, h_params):
#     G, mass_1, mass_2 = h_params
#     reduced_mass = (mass_1 ** (-1) + mass_2 ** (-1)) ** (-1)
#     b = solution.y[1, 0] - solution.y[3, 0]
#     v_infinity = (solution.y[4, 0] / mass_1 - solution.y[6, 0] / mass_2)
#     return 2 * np.arctan(G * (mass_1 + mass_2) / (v_infinity ** 2 * b))
#
#
# def scattering_pm1(solution, hamiltonian, h_params, position_pair_coordinates, momentum_pair_coordinates):
#     """
#     Verified formula. 05/07/22
#     :param solution: The solution for which the theoretical scattering is to be computed. (Need the initial values)
#     :param hamiltonian: The Hamiltonian for which the theoretical scattering is to be computed. Should be compatible be
#     compatible with solution.
#     :param h_params: Parameters for Hamiltonian
#     :param position_pair_coordinates: List of pairs of (x,y) position coordinates.
#     :param momentum_pair_coordinates: List of pairs of (x,y) momentum coordinates
#     :return: Theoretical post Minkowskian scattering angle to first order.
#     """
#     positions, momenta = split_position_momentum(solution=solution)
#     gamma_factor = get_gamma_factor(positions=positions, momenta=momenta, hamiltonian=hamiltonian, h_params=h_params)
#     total_angular_momentum = get_total_angular_momentum(solution=solution,
#                                                         position_pair_coordinates=position_pair_coordinates,
#                                                         momentum_pair_coordinates=momentum_pair_coordinates)[0]
#     G, mass_1, mass_2 = h_params
#     reduced_mass = (mass_1 ** (-1) + mass_2 ** (-1)) ** (-1)
#     mass_sum = mass_1 + mass_2
#     chi_1 = (2 * gamma_factor ** 2 - 1) / (gamma_factor ** 2 - 1) ** 0.5
#     return (chi_1 * (G * mass_sum * reduced_mass / total_angular_momentum)) * 2
#
#
# def scattering_pm2(solution, hamiltonian, h_params, position_pair_coordinates, momentum_pair_coordinates):
#     """
#     Verified formula. 05/07/22
#     :param solution: The solution for which the theoretical scattering angle is to be compouted.
#     :param hamiltonian: The Hamiltonian for which the theoretical scattering angle is to be computed.
#     :param h_params: Parameters of the Hamiltonian.
#     :param position_pair_coordinates: List of (x,y) position coordinate pairs.
#     :param momentum_pair_coordinates: List of (x,y) momentum coordinate pairs
#     :return: Post Minkowskian scattering angle to 2nd order
#     """
#     G, mass_1, mass_2 = h_params
#     positions, momenta = split_position_momentum(solution=solution)
#     no_mass_energy = hamiltonian(positions=positions, momenta=momenta, h_params=h_params)[0]
#     total_energy = no_mass_energy + mass_1 + mass_2
#     gamma_factor = get_gamma_factor(positions=positions, momenta=momenta, hamiltonian=hamiltonian, h_params=h_params)
#     total_angular_momentum = get_total_angular_momentum(solution=solution,
#                                                         position_pair_coordinates=position_pair_coordinates,
#                                                         momentum_pair_coordinates=momentum_pair_coordinates)[0]
#     reduced_mass = (mass_1 ** (-1) + mass_2 ** (-1)) ** (-1)
#     mass_sum = mass_1 + mass_2
#     scattering_sum_factor = G * mass_sum * reduced_mass / total_angular_momentum
#     chi_1 = (2 * gamma_factor**2 - 1) / (gamma_factor**2 - 1)**0.5
#     chi_2 = (3 * np.pi / 8) * (5 * gamma_factor**2 - 1) / (total_energy / mass_sum)
#     return (chi_1 * scattering_sum_factor + chi_2 * scattering_sum_factor**2) * 2
#
#
# def scattering_pm3(solution, hamiltonian, h_params, position_pair_coordinates, momentum_pair_coordinates):
#     """
#     Not verified
#     :param solution: The solution for which the theoretical scattering angle is to be compouted.
#     :param hamiltonian: The Hamiltonian for which the theoretical scattering angle is to be computed.
#     :param h_params: Parameters of the Hamiltonian.
#     :param position_pair_coordinates: List of (x,y) position coordinate pairs.
#     :param momentum_pair_coordinates: List of (x,y) momentum coordinate pairs
#     :return: Post Minkowskian scattering angle to 3rd order
#     """
#     G, mass_1, mass_2 = h_params
#     positions, momenta = split_position_momentum(solution=solution)
#     no_mass_energy = hamiltonian(positions=positions, momenta=momenta, h_params=h_params)[0]
#     E = no_mass_energy + mass_1 + mass_2
#     gamma_factor = get_gamma_factor(positions=positions, momenta=momenta, hamiltonian=hamiltonian, h_params=h_params)
#     L = get_total_angular_momentum(solution=solution,
#                                                         position_pair_coordinates=position_pair_coordinates,
#                                                         momentum_pair_coordinates=momentum_pair_coordinates)[0]
#     mu = (mass_1 ** (-1) + mass_2 ** (-1)) ** (-1)
#     M = mass_1 + mass_2
#     nu = mu / M
#     Gamma = E / M  # Not to confuse with gamma_factor
#     scattering_sum_factor = G / L
#     p_0 = mu / Gamma * (gamma_factor**2 - 1)**0.5  # Observe formula from (1901.07102) gives p_0**2
#     f_1 = 2 * mu**2 * M * (2 * gamma_factor**2 - 1) / Gamma
#     f_2 = 3/2 * mu**2 * M**2 * (5 * gamma_factor**2 - 1) / Gamma
#     f_3 = mu**2 * M**3 * (
#         Gamma * (18 * gamma_factor**2 - 1) / 2 - 4 * nu * gamma_factor * (14*gamma_factor**2 + 25) / (3 * Gamma)
#         + 3/2 * (Gamma - 1) / (gamma_factor**2 - 1) * (2 * gamma_factor**2 - 1) * (5 * gamma_factor**2 - 1)
#         - 8 * nu * (4*gamma_factor**4 - 12*gamma_factor**2 - 3) / (Gamma * (gamma_factor**2 - 1)**0.5)
#             * np.log(((gamma_factor - 1) / 2)**0.5 + ((gamma_factor + 1) / 2)**0.5)
#     )
#     return 2 * (scattering_sum_factor * (f_1 / 2 * p_0) + scattering_sum_factor**2 * (np.pi * f_2 / 4)
#                 + scattering_sum_factor**3 * (p_0 * f_3 + f_1 * f_2 / (2 * p_0) - f_1**3 / (24 * p_0**3)))
#
#
# def theoretical_perihelion_shift(solution, hamiltonian, h_params, position_pair_coordinates, momentum_pair_coordinates):
#     positions, momenta = split_position_momentum(solution=solution)
#     G, mass_1, mass_2 = h_params
#     no_mass_energy = hamiltonian(positions=positions, momenta=momenta, h_params=h_params)[0]
#     total_energy = no_mass_energy + mass_1 + mass_2
#     gamma_factor = get_gamma_factor(positions=positions, momenta=momenta, hamiltonian=hamiltonian, h_params=h_params)
#     mass_sum = mass_1 + mass_2
#     reduced_mass = (mass_1**(-1) + mass_2**(-1)) ** (-1)
#     total_angular_momentum = get_total_angular_momentum(solution=solution,
#                                                         position_pair_coordinates=position_pair_coordinates,
#                                                         momentum_pair_coordinates=momentum_pair_coordinates)[0]
#     return (3 * np.pi
#             * (G * mass_sum * reduced_mass / total_angular_momentum)**2
#             * (total_energy / mass_sum) * (5 * gamma_factor**2 - 1))


# def get_perihelion_shift(solution, shift=0):  # Moved to Utilities
#     """
#
#     :param solution: The solution, for which the perihelion shift is to be measured. Assumes x,y for the shift are on
#     (0, 1)
#     :param shift: Since the solution is discretized, the maximum of the distance, might not be the correct maximum.
#     Shift lets you move to neighbouring points. (-1, 0, or 1) is recommended
#     :return:
#     """
#     r2 = solution.y[0, :] ** 2 + solution.y[1, :] ** 2
#     phi = np.arctan(solution.y[1, :] / solution.y[0, :])
#     perihelion = argrelextrema(r2, np.greater)
#     perihelion_shifted = [pos + shift for pos in perihelion]
#     return phi[perihelion_shifted]
#
#
# def scattering_angle(solution, position_coordinates):  # Moved to Utilities
#     initial_x = solution.y[position_coordinates[0], 0]
#     initial_y = solution.y[position_coordinates[1], 0]
#     final_x = solution.y[position_coordinates[0], -1]
#     final_y = solution.y[position_coordinates[1], -1]
#     initial_angle = np.arctan2(initial_y, initial_x)
#     final_angle = np.arctan2(final_y, final_x)
#     return initial_angle, final_angle + 2 * np.pi
#
#
# def get_scattering(angle_tuple):  # Moved to Utilities
#     return (angle_tuple[1] - angle_tuple[0]) - np.pi
#
#
# def time_averaged_mean(array, time):  # Moved to Utilities
#     number_of_elements = len(array)
#     total_time = time[-1] - time[0]
#     indices = np.arange(number_of_elements - 1)
#     time_averaged_values = array[indices] * (time[indices + 1] - time[indices]) / total_time
#     return np.sum(time_averaged_values)


def post_minkowski_analysis_bound_orbit(r, p, t_span, mass_1, mass_2=1., min_n_steps=10 ** 3, G=1):
    factor_1 = mass_2 / (mass_1 + mass_2)
    factor_2 = - mass_1 / (mass_1 + mass_2)
    r_1 = factor_1 * r
    r_2 = factor_2 * r
    max_step = t_span[1] / min_n_steps

    h_params = G, mass_2, mass_1
    initial = np.array([r_1, 0, r_2, 0, 0, p, 0, -p])

    solution_pm1, solution_pm2, solution_classical = get_solutions((hamiltonian_post_minkowski1,
                                                                    hamiltonian_post_minkowski2,
                                                                    hamiltonian_two_body),
                                                                   t_span=t_span,
                                                                   initial=initial,
                                                                   h_params=h_params,
                                                                   method='DOP853',
                                                                   dense_output=True,
                                                                   max_step=max_step
                                                                   )

    generate_orbit_plot(solutions=(solution_pm1, solution_pm2, solution_classical),
                        legends=(('PM1 Particle1', 'PM1 Particle2'),
                                 ('PM2 Particle1', 'PM2 Particle2'),
                                 ('Newton Particle1', 'Newton Particle2')),
                        x_indices=(0, 2), y_indices=(1, 3))
    plot_scale = 1.1
    plt.title('PM1 Problems')
    plt.xlim((r_2 * plot_scale, r_1 * plot_scale))
    plt.ylim((r_2 * plot_scale, r_1 * plot_scale))
    plt.show()

    generate_orbit_plot(solutions=(solution_pm1, solution_pm2, solution_classical),
                        legends=(('PM1 Particle1', 'PM1 Particle2'),
                                 ('PM2 Particle1', 'PM2 Particle2'),
                                 ('Newton Particle1', 'Newton Particle2')),
                        x_indices=(0, 2), y_indices=(1, 3))
    plot_scale2 = 10
    plt.title('PM1 Problems zoomed out')
    plt.xlim((r_2 * plot_scale2, r_1 * plot_scale2))
    plt.ylim((r_2 * plot_scale2, r_1 * plot_scale2))
    plt.show()

    generate_energy_plot(solutions=(solution_pm1, solution_pm2, solution_classical, solution_pm1, solution_pm2),
                         hamiltonians=(hamiltonian_post_minkowski1, hamiltonian_post_minkowski2, hamiltonian_two_body),
                         all_h_params=(h_params, h_params, h_params),
                         legends=('PM1', 'PM2', 'Classical'))
    plt.show()

    normalized_kinetic, normalized_potential, all_times = generate_split_energy_plot(solutions=(solution_pm1,
                                                                                     solution_pm2,
                                                                                     solution_classical),
                                                                                     hamiltonians=(
                                                                                     hamiltonian_post_minkowski1,
                                                                                     hamiltonian_post_minkowski2,
                                                                                     hamiltonian_two_body),
                                                                                     kinetic_energies=(
                                                                                     hamiltonian_sr_kinetic,
                                                                                     hamiltonian_sr_kinetic,
                                                                                     hamiltonian_newton_kinetic),
                                                                                     all_h_params=(
                                                                                     h_params, h_params, h_params),
                                                                                     legends=('PM1', 'PM2', 'classical')
                                                                                     )
    plt.show()

    # printing average energies
    # for kinetic, potential, times, label in zip(normalized_kinetic,
    #                                             normalized_potential,
    #                                             all_times,
    #                                             ['pm1', 'pm2', 'classical']):
    #     print(f'{label} Average kinetic: {np.mean(kinetic)}')
    #     print(f'{label} Time average kinetic {time_averaged_mean(kinetic, times)}')
    #     print(f'{label} Average potential: {np.mean(potential)}')
    #     print(f'{label} Time average potential {time_averaged_mean(potential, times)}')

    generate_angular_momentum_plot(solutions=(solution_pm1, solution_pm2, solution_classical),
                                   legends=('PM1', 'PM2', 'Classical'),
                                   position_pair_coordinates=(((0, 1), (2, 3)), ((0, 1), (2, 3)), ((0, 1), (2, 3))),
                                   momentum_pair_coordinates=(((4, 5), (6, 7)), ((4, 5), (6, 7)), ((4, 5), (6, 7))))
    plt.show()

    # Precession of the orbits.
    reduced_mass = 1 / (1 / mass_1 + 1 / mass_2)

    angular_momentum = (get_angular_momentum(solution_pm1.y[0:2, :], solution_pm1.y[4:6, :])
                        + get_angular_momentum(solution_pm1.y[2:4, :], solution_pm1.y[6:8, :]))

    pred_delta_phi = (6 * np.pi * (h_params[0] * (mass_1 + mass_2) * reduced_mass) ** 2
                      / angular_momentum[0] ** 2) * (np.arange(len(get_perihelion_shift(solution=solution_pm1))) + 1)

    pred_delta_phi_relativistic1 = (theoretical_perihelion_shift(solution=solution_pm1,
                                                                 hamiltonian=hamiltonian_post_minkowski1,
                                                                 h_params=h_params,
                                                                 position_pair_coordinates=((0, 1), (2, 3)),
                                                                 momentum_pair_coordinates=((4, 5), (6, 7)))
                                    * (np.arange(len(get_perihelion_shift(solution=solution_pm1))) + 1))

    pred_delta_phi_relativistic2 = (theoretical_perihelion_shift(solution=solution_pm2,
                                                                 hamiltonian=hamiltonian_post_minkowski2,
                                                                 h_params=h_params,
                                                                 position_pair_coordinates=((0, 1), (2, 3)),
                                                                 momentum_pair_coordinates=((4, 5), (6, 7)))
                                    * (np.arange(len(get_perihelion_shift(solution=solution_pm1))) + 1))

    print('predicted: ', pred_delta_phi)
    print('predicted pm1 relativistic: ', pred_delta_phi_relativistic1)
    print('predicted pm2 relativistic: ', pred_delta_phi_relativistic2)  # Maybe doesn't give pm2
    print('pm1: ', get_perihelion_shift(solution=solution_pm1))
    print('pm2: ', get_perihelion_shift(solution=solution_pm2))

    for shift in [-1, 0, 1]:
        print(f'Relative Error pm1 {shift=}: ',
              (get_perihelion_shift(solution=solution_pm1, shift=shift) - pred_delta_phi) / pred_delta_phi)
    for shift in [-1, 0, 1]:
        print(f'Relative Error pm2 {shift=}: ',
              (get_perihelion_shift(solution=solution_pm2, shift=shift) - pred_delta_phi) / pred_delta_phi)


def post_minkowski_analysis_scattering(r, b, p, mass_1, t_span, mass_2=1., minimal_steps=10 ** 3, no_pm1=False):
    max_step = t_span[1] / minimal_steps
    factor_1 = mass_2 / (mass_1 + mass_2)
    factor_2 = - mass_1 / (mass_1 + mass_2)
    initial = np.array([factor_1 * np.sqrt(r ** 2 - b ** 2), factor_1 * b,  # Position 1
                        factor_2 * np.sqrt(r ** 2 - b ** 2), factor_2 * b,  # Position 2
                        - p, 0,  # Momentum 1
                        p, 0])  # Momentum 2
    G = 1
    h_params = G, mass_1, mass_2

    all_solutions = get_solutions((hamiltonian_post_minkowski1, hamiltonian_post_minkowski2,
                                   hamiltonian_post_minkowski3, hamiltonian_two_body),
                                  t_span=t_span, initial=initial, h_params=h_params,
                                  method='DOP853', dense_output=True, max_step=max_step
                                  )
    all_hamiltonians = [hamiltonian_post_minkowski1, hamiltonian_post_minkowski2,
                        hamiltonian_post_minkowski3, hamiltonian_two_body]
    solution_names = ('PM1', 'PM2', 'PM3', 'Newton')

    all_hamiltonians_no_pm1 = all_hamiltonians[1:]
    all_solutions_no_pm1 = all_solutions[1:]
    solution_names_no_pm1 = solution_names[1:]

    used_solutions, used_hamiltonians, used_names = (
        (all_solutions_no_pm1, all_hamiltonians_no_pm1, solution_names_no_pm1)
        if no_pm1
        else (all_solutions, all_hamiltonians, solution_names))

    used_h_params = [h_params for _ in used_names]

    orbit_labels = [(f'{sol_name} Particle1', f'{sol_name} Particle2') for sol_name in used_names]
    generate_orbit_plot(solutions=used_solutions,
                        legends=orbit_labels,
                        x_indices=(0, 2), y_indices=(1, 3))
    plt.show()

    generate_energy_plot(solutions=used_solutions,
                         hamiltonians=used_hamiltonians,
                         all_h_params=used_h_params,
                         legends=used_names
                         )
    plt.show()

    position_pairs = [((0, 1), (2, 3)) for _ in used_names]
    momentum_pairs = [((4, 5), (6, 7)) for _ in used_names]
    generate_angular_momentum_plot(solutions=used_solutions,
                                   legends=used_names,
                                   position_pair_coordinates=position_pairs,
                                   momentum_pair_coordinates=momentum_pairs
                                   )
    plt.show()
    solution_pm1, solution_pm2, solution_pm3, solution_classical = all_solutions
    if not no_pm1:
        scattering_angle_pm1 = scattering_angle(solution=solution_pm1, position_coordinates=(0, 1))
    scattering_angle_pm2 = scattering_angle(solution=solution_pm2, position_coordinates=(0, 1))
    scattering_angle_pm3 = scattering_angle(solution=solution_pm3, position_coordinates=(0, 1))
    scattering_angle_classical = scattering_angle(solution=solution_classical, position_coordinates=(0, 1))

    if not no_pm1:
        print(f'{scattering_angle_pm1=}')
    print(f'{scattering_angle_pm2=}')
    print(f'{scattering_angle_pm3=}')
    print(f'{scattering_angle_classical=}')

    if not no_pm1:
        print('pm1 scattering angle: ', get_scattering(scattering_angle_pm1))
    print('pm2 scattering angle: ', get_scattering(scattering_angle_pm2))
    print('pm3 scattering angle: ', get_scattering(scattering_angle_pm3))
    print('Classical scattering angle: ', get_scattering(scattering_angle_classical))

    if not no_pm1:
        print('Theoretical scattering pm1: ', scattering_pm1(solution=solution_pm1,
                                                             hamiltonian=hamiltonian_post_minkowski1,
                                                             h_params=h_params,
                                                             position_pair_coordinates=((0, 1), (2, 3)),
                                                             momentum_pair_coordinates=((4, 5), (6, 7))))
    print('Theoretical scattering pm2: ', scattering_pm2(solution=solution_pm2,
                                                         hamiltonian=hamiltonian_post_minkowski2,
                                                         h_params=h_params,
                                                         position_pair_coordinates=((0, 1), (2, 3)),
                                                         momentum_pair_coordinates=((4, 5), (6, 7))))
    print('Theoretical scattering classical: ', scattering_classical(solution=solution_classical, h_params=h_params))


def main():
    # post_minkowski_analysis_bound_orbit(r=8 * 10**1, p=0.08, t_span=(0, 7.5 * 10**3), mass_1=1)
    m = 10 ** (-4)
    post_minkowski_analysis_scattering(r=10 ** 4, b=1.5 * 10 ** 2, p=0.71 * m, mass_1=m, t_span=(0, 4 * 10 ** 4),
                                       mass_2=10.)  # Sprednings_vinkler
    # m = 10 ** (-5)
    # post_minkowski_analysis_scattering(r=10 ** 3, b=0.8 * 10 ** 1, p=0.767499 * m, mass_1=m, t_span=(0, 2 * 10 ** 3), mass_2=1., no_pm1=True)
    # post_minkowski_analysis_scattering(r=10 ** 4, b=10 ** 2, p=5*m, mass_1=m, mass_2=10**3)
    pass


if __name__ == '__main__':
    main()
