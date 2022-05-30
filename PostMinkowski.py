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
                         x_label='Time [au]', y_label='Energy [au]'):  # Doesn't work
    for solution, hamiltonian, h_params, label in zip(solutions, hamiltonians, all_h_params, legends):
        number_of_coordiantes = len(solution.y[:, 0])
        position_coordinates = solution.y[:number_of_coordiantes, :]
        momentum_coordinates = solution.y[number_of_coordiantes:, :]
        initial_energy = hamiltonian(solution.y[:number_of_coordiantes, 0], solution.y[number_of_coordiantes:, 0], h_params)
        normalized_energy = hamiltonian(position_coordinates, momentum_coordinates, h_params) / initial_energy
        plt.plot(solution.t, normalized_energy, label=label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)


def generate_angular_momentum_plot(solutions, legends, position_pair_coordinates, momentum_pair_coordinates,
                                   x_label='Time [au]', y_label='Energy [au]'):
    for solution, label, positions, momenta in zip(solutions,
                                                   legends,
                                                   position_pair_coordinates,
                                                   momentum_pair_coordinates):
        angular_momentum = 0
        angular_momentum_initial = 0
        for position_coordinates, momentum_coordinates in zip(positions, momenta):
            position = (solution.y[position_coordinates[0], :], solution.y[position_coordinates[0], :])
            momentum = (solution.y[momentum_coordinates[0], :], solution.y[momentum_coordinates[0], :])
            angular_momentum += get_angular_momentum(positions=position, momenta=momentum)
            position = (solution.y[position_coordinates[0], 0], solution.y[position_coordinates[0], 0])
            momentum = (solution.y[momentum_coordinates[0], 0], solution.y[momentum_coordinates[0], 0])
            angular_momentum_initial += get_angular_momentum(positions=position, momenta=momentum)
        plt.plot(solution.t, angular_momentum / angular_momentum_initial, label=label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)


def get_angular_momentum(positions, momenta):
    x = positions[0, :]
    y = positions[1, :]
    px = momenta[0, :]
    py = momenta[1, :]
    return x * py - y * px


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

    return energy_1 + energy_2 + (G * c_1) / (energy_1 * energy_2 * distance)


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
    return (energy_1 + energy_2
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


def post_minkowski_analysis_bound_orbit():

    def get_perihelion_shift(solution):
        r2 = solution.y[0, :] ** 2 + solution.y[1, :] ** 2
        phi = np.arctan(solution.y[1, :] / solution.y[0, :])
        perihelion = argrelextrema(r2, np.greater)
        perihelion_left = [pos - 1 for pos in perihelion]
        perihelion_right = [pos + 1 for pos in perihelion]
        return phi[perihelion_left], phi[perihelion], phi[perihelion_right]

    t_span = (0, 4*10**5)
    max_step = t_span[1] / 1000
    r_1 = 10**3
    r_2 = -10**3
    mass_1 = 1
    mass_2 = 1
    G = 1
    h_params = G, mass_2, mass_1
    p_1 = 0.008
    initial = np.array([r_1, 0, r_2, 0, 0, p_1, 0, -p_1])

    reduced_mass = 1 / (1 / mass_1 + 1 / mass_2)

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
    # Duplicate code made into a function. Left for emergency use.
    # solution_pm1 = solve_hamiltonian(hamiltonian=hamiltonian_post_minkowski1,
    #                                  t_span=t_span, initial=initial, h_params=h_params,
    #                                  method='DOP853', dense_output=True, max_step=max_step)
    # solution_classical = solve_hamiltonian(hamiltonian=hamiltonian_two_body,
    #                                        t_span=t_span, initial=initial, h_params=h_params,
    #                                        method='DOP853', dense_output=True, max_step=max_step)
    # solution_sr = solve_hamiltonian(hamiltonian=hamiltonian_sr_newton_pot,
    #                                 t_span=t_span, initial=initial, h_params=h_params,
    #                                 method='DOP853', dense_output=True, max_step=max_step)

    # plt.plot(solution_pm1.y[0, :], solution_pm1.y[1, :], 'r-', label='particle 1 PM1')
    # plt.plot(solution_pm1.y[2, :], solution_pm1.y[3, :], 'b-', label='particle 2 PM1')
    # plt.plot(solution_classical.y[0, :], solution_classical.y[1, :], 'y--', label='particle 1 classical')
    # plt.plot(solution_classical.y[2, :], solution_classical.y[3, :], 'c--', label='particle 2 classical')
    # # plt.plot(solution_sr.y[0, :], solution_sr.y[1, :], 'm--', label='particle 1 SR')
    # # plt.plot(solution_sr.y[2, :], solution_sr.y[3, :], 'g--', label='particle 2 SR')
    # plt.legend()
    # plt.title(f"Two body PM1 {mass_1=} {mass_2=}")
    # plt.xlabel("X-Coordinate [au]")
    # plt.ylabel("Y-Coordinate [au]")
    # plt.show()

    generate_orbit_plot(solutions=(solution_pm1, solution_pm2, solution_classical),
                        legends=(('PM1 Particle1', 'PM1 Particle2'),
                                 ('PM2 Particle1', 'PM2 Particle2'),
                                 ('Newton Particle1', 'Newton Particle2')),
                        x_indices=(0, 2), y_indices=(1, 3))
    scale = 1.1
    plt.xlim((r_2 * scale, r_1 * scale))
    plt.ylim((r_2 * scale, r_1 * scale))
    plt.show()

    plt.plot((hamiltonian_post_minkowski1(solution_pm1.y[:4, :], solution_pm1.y[4:, :], h_params=h_params)
              / hamiltonian_post_minkowski1(solution_pm1.y[:4, 0], solution_pm1.y[4:, 0], h_params=h_params)),
             'g-', label='PM1 Energy')
    plt.plot((hamiltonian_two_body(solution_classical.y[:4, :], solution_classical.y[4:, :], h_params=h_params)
             / hamiltonian_two_body(solution_classical.y[:4, 0], solution_classical.y[4:, 0], h_params=h_params)),
             'c-', label='Newton Energy')
    plt.legend()
    # generate_energy_plot(solutions=(solution_pm1, solution_pm2, solution_classical),
    #                      hamiltonians=(hamiltonian_post_minkowski1, hamiltonian_post_minkowski2, hamiltonian_two_body),
    #                      all_h_params=(h_params, h_params, h_params),
    #                      legends=('PM1', 'PM2', 'Classical'))
    plt.show()

    angular_pm1 = (get_angular_momentum(solution_pm1.y[0:2, :], solution_pm1.y[4:6, :])
                   + get_angular_momentum(solution_pm1.y[2:4, :], solution_pm1.y[6:8, :]))
    angular_classical = (get_angular_momentum(solution_classical.y[0:2, :], solution_classical.y[4:6, :])
                         + get_angular_momentum(solution_classical.y[2:4, :], solution_classical.y[6:8, :]))

    plt.plot(solution_pm1.t, angular_pm1 / angular_pm1[0], 'g-', label='PM1 Angular')
    plt.plot(solution_classical.t, angular_classical / angular_classical[0], 'c-', label='Newton Angular')
    plt.legend()
    plt.show()

    # Precession of the orbits.
    pred_delta_phi = (6 * np.pi * (h_params[0] * (mass_1 + mass_2) * reduced_mass) ** 2
                      / angular_pm1[0] ** 2) * (np.arange(len(get_perihelion_shift(solution=solution_pm1)[1])) + 1)
    print("predicted: ", pred_delta_phi)
    print("pm1: ", get_perihelion_shift(solution=solution_pm1))
    print("pm2: ", get_perihelion_shift(solution=solution_pm2))
    print("Relative Error pm1: ",
          (get_perihelion_shift(solution=solution_pm1)[1]
           - pred_delta_phi)
          / pred_delta_phi)
    print("Relative Error pm2: ",
          (get_perihelion_shift(solution=solution_pm2)[1]
           - pred_delta_phi)
          / pred_delta_phi)


def post_minkowski_analysis_scattering(r, b, p, mass_1, mass_2=1.):
    t_span = (0, 2*10**5)
    minimal_steps = 10**4
    max_step = t_span[1] / minimal_steps
    factor_1 = mass_2 / (mass_1 + mass_2)
    factor_2 = - mass_1 / (mass_1 + mass_2)
    initial = np.array([factor_1 * np.sqrt(r**2 - b**2), factor_1 * b,  # Position 1
                        factor_2 * np.sqrt(r**2 - b**2), factor_2 * b,  # Position 2
                        - p, 0,  # Momentum 1
                        p, 0])  # Momentum 2
    G = 1
    h_params = G, mass_1, mass_2

    solution_pm1 = solve_hamiltonian(hamiltonian=hamiltonian_post_minkowski1,
                                     t_span=t_span, initial=initial, h_params=h_params,
                                     method='DOP853', dense_output=True, max_step=max_step)
    solution_classical = solve_hamiltonian(hamiltonian=hamiltonian_two_body,
                                           t_span=t_span, initial=initial, h_params=h_params,
                                           method='DOP853', dense_output=True, max_step=max_step)
    solution_sr = solve_hamiltonian(hamiltonian=hamiltonian_sr_newton_pot,
                                    t_span=t_span, initial=initial, h_params=h_params,
                                    method='DOP853', dense_output=True, max_step=max_step)

    plt.plot(solution_pm1.y[0, :], solution_pm1.y[1, :], 'r-', label='particle 1 PM1')
    plt.plot(solution_pm1.y[2, :], solution_pm1.y[3, :], 'b-', label='particle 2 PM1')
    plt.legend()
    plt.title(f"Two body PM1 {mass_1=} {mass_2=} {b=} {p=}")
    plt.xlabel("X-Coordinate [au]")
    plt.ylabel("Y-Coordinate [au]")
    plt.show()


def main():
    post_minkowski_analysis_bound_orbit()
    # m = 10**(-4)
    # post_minkowski_analysis_scattering(r=10**4, b=10**2, p=0.5*m, mass_1=m, mass_2=10.)
    # post_minkowski_analysis_scattering(r=10 ** 4, b=10 ** 2, p=5*m, mass_1=m, mass_2=10**3)
    pass


if __name__ == '__main__':
    main()
