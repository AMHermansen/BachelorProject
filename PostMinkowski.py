import numpy as np
import matplotlib.pyplot as plt
from HamiltonianOrbits import solve_hamiltonian
from scipy.signal import argrelextrema


def get_solutions(hamiltonians, **kwargs):
    return [solve_hamiltonian(hamiltonian, **kwargs) for hamiltonian in hamiltonians]


def generate_orbit_plot(solutions, legends, x_indices, y_indices,
                        x_label='x coordinate [au]', y_label='y coordinate [au]'):
    for solution, labels in zip(solutions, legends):
        for x_coordinate, y_coordinate, label in zip(x_indices, y_indices, labels):
            plt.plot(solution.y[x_coordinate, :], solution.y[y_coordinate, :], label=label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()


def generate_energy_plot(solutions, hamiltonians, all_h_params, legends,
                         x_label='Time [au]', y_label='Energy [au]'):
    for solution, hamiltonian, h_params, label in zip(solutions, hamiltonians, all_h_params, legends):
        number_of_coordinates = len(solution.y[:, 0]) // 2  # Number of position/momentum coordinates. Half the total number
        position_coordinates = solution.y[:number_of_coordinates, :]
        momentum_coordinates = solution.y[number_of_coordinates:, :]
        initial_energy = hamiltonian(solution.y[:number_of_coordinates, 0], solution.y[number_of_coordinates:, 0], h_params)
        normalized_energy = hamiltonian(position_coordinates, momentum_coordinates, h_params) / initial_energy
        plt.plot(solution.t, normalized_energy, label=label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()


def generate_split_energy_plot(solutions, hamiltonians, kinetic_energies, all_h_params, legends,
                          x_label='Time [au]', y_label='Energy [au]', plot_kinetic=True, plot_potential=True):
    all_normalized_kinetic = list()
    all_normalized_potential = list()
    all_times = list()
    for solution, hamiltonian, kinetic_energy, h_params, label in zip(solutions, hamiltonians, kinetic_energies, all_h_params, legends):
        number_of_coordinates = len(solution.y[:, 0]) // 2
        position_coordinates = solution.y[:number_of_coordinates, :]
        momentum_coordinates = solution.y[number_of_coordinates:, :]
        initial_energy = hamiltonian(solution.y[:number_of_coordinates, 0], solution.y[number_of_coordinates:, 0], h_params)
        normalized_energy = hamiltonian(position_coordinates, momentum_coordinates, h_params) / np.abs(initial_energy)
        normalized_kinetic = kinetic_energy(position_coordinates, momentum_coordinates, h_params) / np.abs(initial_energy)
        normalized_potential = normalized_energy - normalized_kinetic
        plt.plot(solution.t, normalized_energy, label=f'{label} Total energy')
        if plot_kinetic:
            plt.plot(solution.t, normalized_kinetic, label=f'{label} Kinetic energy')
        if plot_potential:
            plt.plot(solution.t, normalized_potential, label=f'{label} Potential energy')
        all_normalized_kinetic.append(normalized_kinetic)
        all_normalized_potential.append(normalized_potential)
        all_times.append(solution.t)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    return all_normalized_kinetic, all_normalized_potential, all_times


def generate_angular_momentum_plot(solutions, legends, position_pair_coordinates, momentum_pair_coordinates,
                                   x_label='Time [au]', y_label='Angular momentum [au]'):
    """

    :param solutions: Solutions from solve_hamiltonian
    :param legends: Tuple of legends for the plots
    :param position_pair_coordinates: Position pairs, when multiple x-/y-coordinates
    :param momentum_pair_coordinates: Momentum pairs, when multiple x-/y-momenta
    :param x_label:
    :param y_label:
    :return:
    """
    for solution, label, positions, momenta in zip(solutions,
                                                   legends,
                                                   position_pair_coordinates,
                                                   momentum_pair_coordinates):
        angular_momentum = 0
        for position_coordinates, momentum_coordinates in zip(positions, momenta):
            position = np.array([solution.y[position_coordinates[0], :], solution.y[position_coordinates[1], :]])
            momentum = np.array([solution.y[momentum_coordinates[0], :], solution.y[momentum_coordinates[1], :]])
            angular_momentum += get_angular_momentum(positions=position, momenta=momentum)
        plt.plot(solution.t, angular_momentum / angular_momentum[0], label=label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()


def get_angular_momentum(positions, momenta):
    x = positions[0, :]
    y = positions[1, :]
    px = momenta[0, :]
    py = momenta[1, :]
    return x * py - y * px


def get_perihelion_shift(solution, shift=0):
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


def time_averaged_mean(array, time):
    number_of_elements = len(array)
    total_time = time[-1] - time[0]
    indices = np.arange(number_of_elements - 1)
    time_averaged_values = array[indices] * (time[indices + 1] - time[indices]) / total_time
    return np.sum(time_averaged_values)


def hamiltonian_post_minkowski1(positions, momenta, h_params):
    """
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
    c_1 = mass_1**2 * mass_2**2 - 2 * (energy_1 * energy_2 - momenta[0] * momenta[2] - momenta[1] * momenta[3])**2
    c_mass_1 = 3 * mass_1 ** 2 * (mass_1 ** 2 * mass_2 ** 2 - 5 * dotted_momenta ** 2)
    c_mass_2 = 3 * mass_2 ** 2 * (mass_1 ** 2 * mass_2 ** 2 - 5 * dotted_momenta ** 2)
    energy_total = energy_1 + energy_2
    xi = energy_1 * energy_2 / (energy_total ** 2)
    return (energy_1 + energy_2 - mass_1 - mass_2
            + (G * c_1) / (energy_1 * energy_2 * distance)
            + G ** 2 / (distance ** 2 * energy_1 * energy_2) * ((c_mass_1 / mass_1 + c_mass_2 / mass_2) / 4
                                                                + (c_1**2 * (xi - 1) / (2 * energy_total**3 * xi**2)
                                                                   - 4*c_1 * dotted_momenta / (energy_total * xi))))


def hamiltonian_two_body(positions, momenta, h_params):
    G, mass_1, mass_2 = h_params
    energy_1 = (momenta[0]**2 + momenta[1]**2) / (2 * mass_1)
    energy_2 = (momenta[2]**2 + momenta[3]**2) / (2 * mass_2)
    distance = ((positions[0] - positions[2])**2 + (positions[1] - positions[3])**2)**0.5
    return energy_1 + energy_2 - G * mass_1 * mass_2 / distance


def hamiltonian_sr_newton_pot(positions, momenta, h_params):
    G, mass_1, mass_2 = h_params
    energy_1 = (mass_1**2 + momenta[0]**2 + momenta[1]**2)**0.5
    energy_2 = (mass_2**2 + momenta[2]**2 + momenta[3]**2)**0.5
    distance = ((positions[0] - positions[2])**2 + (positions[1] - positions[3])**2)**0.5
    return energy_1 + energy_2 - G * mass_2 * mass_1 / distance


def hamiltonian_sr_kinetic(positions, momenta, h_params):
    G, mass_1, mass_2 = h_params
    energy_1 = (mass_1**2 + momenta[0]**2 + momenta[1]**2)**0.5 - mass_1
    energy_2 = (mass_2**2 + momenta[2]**2 + momenta[3]**2)**0.5 - mass_2
    return energy_1 + energy_2


def hamiltonian_newton_kinetic(positions, momenta, h_params):
    G, mass_1, mass_2 = h_params
    return (momenta[0]**2 + momenta[1]**2) / (2 * mass_1) + (momenta[2]**2 + momenta[3]**2) / (2 * mass_2)


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

    for kinetic, potential, times, label in zip(normalized_kinetic,
                                                normalized_potential,
                                                all_times,
                                                ['pm1', 'pm2', 'classical']):
        print(f'{label} Average kinetic: {np.mean(kinetic)}')
        print(f'{label} Time average kinetic {time_averaged_mean(kinetic, times)}')
        print(f'{label} Average potential: {np.mean(potential)}')
        print(f'{label} Time average potential {time_averaged_mean(potential, times)}')

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

    print('predicted pm1: ', pred_delta_phi)
    print('pm1: ', get_perihelion_shift(solution=solution_pm1))
    print('pm2: ', get_perihelion_shift(solution=solution_pm2))

    for shift in [-1, 0, 1]:
        print(f'Relative Error pm1 {shift=}: ',
              (get_perihelion_shift(solution=solution_pm1, shift=shift) - pred_delta_phi) / pred_delta_phi)
    for shift in [-1, 0, 1]:
        print(f'Relative Error pm2 {shift=}: ',
              (get_perihelion_shift(solution=solution_pm2, shift=shift) - pred_delta_phi) / pred_delta_phi)


def post_minkowski_analysis_scattering(r, b, p, mass_1, t_span, mass_2=1., minimal_steps=10**3):
    max_step = t_span[1] / minimal_steps
    factor_1 = mass_2 / (mass_1 + mass_2)
    factor_2 = - mass_1 / (mass_1 + mass_2)
    initial = np.array([factor_1 * np.sqrt(r**2 - b**2), factor_1 * b,  # Position 1
                        factor_2 * np.sqrt(r**2 - b**2), factor_2 * b,  # Position 2
                        - p, 0,  # Momentum 1
                        p, 0])  # Momentum 2
    G = 1
    h_params = G, mass_1, mass_2

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
    plt.show()

    generate_energy_plot(solutions=(solution_pm1, solution_pm2, solution_classical, solution_pm1, solution_pm2),
                         hamiltonians=(hamiltonian_post_minkowski1, hamiltonian_post_minkowski2, hamiltonian_two_body),
                         all_h_params=(h_params, h_params, h_params),
                         legends=('PM1', 'PM2', 'Classical'))
    plt.show()

    generate_angular_momentum_plot(solutions=(solution_pm1, solution_pm2, solution_classical),
                                   legends=('PM1', 'PM2', 'Classical'),
                                   position_pair_coordinates=(((0, 1), (2, 3)), ((0, 1), (2, 3)), ((0, 1), (2, 3))),
                                   momentum_pair_coordinates=(((4, 5), (6, 7)), ((4, 5), (6, 7)), ((4, 5), (6, 7))))
    plt.show()


def main():
    post_minkowski_analysis_bound_orbit(r=8 * 10**1, p=0.06, t_span=(0, 2.5 * 10**3), mass_1=1)
    # m = 10**(-4)
    # post_minkowski_analysis_scattering(r=10**4, b=10**2, p=0.5*m, mass_1=m, t_span=(0, 10**5), mass_2=10.)
    # post_minkowski_analysis_scattering(r=10 ** 4, b=10 ** 2, p=5*m, mass_1=m, mass_2=10**3)
    pass


if __name__ == '__main__':
    main()
