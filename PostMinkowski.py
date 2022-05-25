import numpy as np
import matplotlib.pyplot as plt
from HamiltonianOrbits import solve_hamiltonian
from scipy.signal import argrelextrema


def get_solutions(hamiltonians, **kwargs):
    return [solve_hamiltonian(hamiltonian, **kwargs) for hamiltonian in hamiltonians]


def generate_orbit_plot(solutions, legends, x_indices, y_indices,
                        x_label='x coordinate [au]', y_label='y coordinate [au]'):
    for solution in solutions:
        for x_coordinate, y_coordinate, legend in zip(x_indices, y_indices, legends):
            plt.plot(solution.y[x_coordinate, :], solution.y[y_coordinate, :], label=legend)
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
    :return:
    """
    G, mass_1, mass_2 = h_params
    energy_1 = (mass_1**2 + momenta[0]**2 + momenta[1]**2)**0.5
    energy_2 = (mass_2**2 + momenta[2]**2 + momenta[3]**2)**0.5
    momentum_abs_sq = - momenta[0] * momenta[2] - momenta[1] * momenta[3]
    distance = ((positions[0] - positions[2])**2 + (positions[1] - positions[3])**2) ** 0.5
    c_1 = mass_1**2 * mass_2**2 - 2 * (energy_1 * energy_2 + momentum_abs_sq)**2

    return energy_1 + energy_2 + (G * c_1) / (energy_1 * energy_2 * distance)


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

    t_span = (0, 3*10**7)
    max_step = t_span[1] / 1000
    r_1 = 10**4
    r_2 = -10**4
    mass_1 = 1
    mass_2 = 1
    G = 1
    h_params = G, mass_2, mass_1
    p_1 = 0.003
    initial = np.array([r_1, 0, r_2, 0, 0, p_1, 0, -p_1])

    reduced_mass = 1 / (1 / mass_1 + 1 / mass_2)

    solution_pm1, solution_classical, solution_sr = get_solutions((hamiltonian_post_minkowski1,
                                                                   hamiltonian_two_body,
                                                                   hamiltonian_sr_newton_pot),
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

    plt.plot(solution_pm1.y[0, :], solution_pm1.y[1, :], 'r-', label='particle 1 PM1')
    plt.plot(solution_pm1.y[2, :], solution_pm1.y[3, :], 'b-', label='particle 2 PM1')
    plt.plot(solution_classical.y[0, :], solution_classical.y[1, :], 'y--', label='particle 1 classical')
    plt.plot(solution_classical.y[2, :], solution_classical.y[3, :], 'c--', label='particle 2 classical')
    # plt.plot(solution_sr.y[0, :], solution_sr.y[1, :], 'm--', label='particle 1 SR')
    # plt.plot(solution_sr.y[2, :], solution_sr.y[3, :], 'g--', label='particle 2 SR')
    plt.legend()
    plt.title(f"Two body PM1 {mass_1=} {mass_2=}")
    plt.xlabel("X-Coordinate [au]")
    plt.ylabel("Y-Coordinate [au]")
    plt.show()

    plt.plot((hamiltonian_post_minkowski1(solution_pm1.y[:4, :], solution_pm1.y[4:, :], h_params=h_params)
              / hamiltonian_post_minkowski1(solution_pm1.y[:4, 0], solution_pm1.y[4:, 0], h_params=h_params)),
             'g-', label='PM1 Energy')
    plt.plot((hamiltonian_sr_newton_pot(solution_sr.y[:4, :], solution_sr.y[4:, :], h_params=h_params)
              / hamiltonian_sr_newton_pot(solution_sr.y[:4, 0], solution_sr.y[4:, 0], h_params=h_params)),
             'b-', label='SR Energy')
    plt.plot((hamiltonian_two_body(solution_classical.y[:4, :], solution_classical.y[4:, :], h_params=h_params)
             / hamiltonian_two_body(solution_classical.y[:4, 0], solution_classical.y[4:, 0], h_params=h_params)),
             'c-', label='Newton Energy')
    plt.legend()
    plt.show()

    angular_pm1 = (get_angular_momentum(solution_pm1.y[0:2, :], solution_pm1.y[4:6, :])
                   + get_angular_momentum(solution_pm1.y[2:4, :], solution_pm1.y[6:8, :]))
    angular_classical = (get_angular_momentum(solution_classical.y[0:2, :], solution_classical.y[4:6, :])
                         + get_angular_momentum(solution_classical.y[2:4, :], solution_classical.y[6:8, :]))
    angular_sr = (get_angular_momentum(solution_sr.y[0:2, :], solution_sr.y[4:6, :])
                  + get_angular_momentum(solution_sr.y[2:4, :], solution_sr.y[6:8, :]))

    plt.plot(solution_pm1.t, angular_pm1 / angular_pm1[0], 'g-', label='PM1 Angular')
    plt.plot(solution_sr.t, angular_sr / angular_sr[0], 'b-', label='SR Angular')
    plt.plot(solution_classical.t, angular_classical / angular_classical[0], 'c-', label='Newton Angular')
    plt.legend()
    plt.show()

    pred_delta_phi = (6 * np.pi * (h_params[0] * (mass_1 + mass_2) * reduced_mass) ** 2
                      / angular_pm1[0] ** 2) * (np.arange(len(get_perihelion_shift(solution=solution_pm1)[1])) + 1)

    print("predicted: ", pred_delta_phi)
    print("pm1: ", get_perihelion_shift(solution=solution_pm1))
    print("sr: ", get_perihelion_shift(solution=solution_sr))
    print("Relative Error: ",
          (get_perihelion_shift(solution=solution_pm1)[1]
           - pred_delta_phi)
          / get_perihelion_shift(solution=solution_pm1)[1])


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
    # plt.plot(solution_classical.y[0, :], solution_classical.y[1, :], 'y--', label='particle 1 classical')
    # plt.plot(solution_classical.y[2, :], solution_classical.y[3, :], 'c--', label='particle 2 classical')
    # plt.plot(solution_sr.y[0, :], solution_sr.y[1, :], 'm--', label='particle 1 SR')
    # plt.plot(solution_sr.y[2, :], solution_sr.y[3, :], 'g--', label='particle 2 SR')
    plt.legend()
    plt.title(f"Two body PM1 {mass_1=} {mass_2=} {b=} {p=}")
    plt.xlabel("X-Coordinate [au]")
    plt.ylabel("Y-Coordinate [au]")
    plt.show()


def main():
    post_minkowski_analysis_bound_orbit()
    m = 10**(-4)
    post_minkowski_analysis_scattering(r=10**4, b=10**2, p=0.5*m, mass_1=m, mass_2=10.)
    post_minkowski_analysis_scattering(r=10 ** 4, b=10 ** 2, p=5*m, mass_1=m, mass_2=10**3)
    pass


if __name__ == '__main__':
    main()
