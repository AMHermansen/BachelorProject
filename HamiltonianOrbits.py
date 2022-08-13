import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from Utilities import solve_hamiltonian
from Hamiltonians import (hamiltonian_ho, hamiltonian_orbit, hamiltonian_double_orbit,
                          hamiltonian_einstein_infeld_hoffmann, hamiltonian_reduced_two_body)


def harmonic_oscillator(t_span=(0, 10), initial=np.array([10.0, 0.0]), h_params=np.array([3.0, 2.0])):
    solution = solve_hamiltonian(hamiltonian=hamiltonian_ho, t_span=t_span, initial=initial, h_params=h_params,
                                 method='DOP853', dense_output=True)
    xn = np.linspace(t_span[0], t_span[1], 100)
    plt.plot(xn, 10 * np.cos((2 / 3) ** 0.5 * xn), 'ko', label='exact')
    plt.plot(xn, solution.sol(xn)[0, :], 'r-', label='numerical')
    plt.xlabel(r'Time $\left[\sqrt{\frac{m}{k}} \right]$')
    plt.ylabel('Position [au]')
    plt.title('Harmonic oscillator, m=3, k=2')
    plt.legend()
    plt.show()


def orbit(t_span=(0, 10 ** 4), initial=np.array([10 ** 3, 0.0, 0.0, 0.8]), h_params=np.array([1000.0, 0.001])):
    solution = solve_hamiltonian(hamiltonian=hamiltonian_orbit, t_span=t_span, initial=initial, h_params=h_params,
                                 method='DOP853', dense_output=True, max_step=100)

    energy = hamiltonian_orbit(initial[:len(initial) // 2], initial[len(initial) // 2:], h_params=h_params)
    angular_momentum = initial[-1]
    M, m = h_params

    eccentricity = (1 + (2 * energy * angular_momentum ** 2) / (m ** 3 * M ** 2)) ** (1 / 2)
    C = m ** 2 * M / angular_momentum ** 2
    phi_n = np.linspace(0, 2 * np.pi, 100)
    rn = 1 / (C * (1 - eccentricity * np.cos(phi_n)))
    plt.plot(rn * np.cos(phi_n), rn * np.sin(phi_n), 'ko', label='exact')
    tn = np.linspace(t_span[0], t_span[1], 10)
    plt.plot(solution.y[0, :] * np.cos(solution.y[1, :]),
             solution.y[0, :] * np.sin(solution.y[1, :]),
             'r-', label='numerical')
    plt.xlabel('x-coordinate [au]')
    plt.ylabel('y-coordinate [au]')
    plt.title('One body problem')
    plt.legend()
    plt.show()


def two_bodies():
    p_phi = 4
    m = 0.5
    M = 2.0
    r_1 = 200
    r_2 = M / m * r_1
    t_span = (0, 8 * 10 ** 4)
    initial = np.array([r_1, 0.0, r_2, np.pi,  # positions
                        0.0, p_phi * (m / M), 0.0, p_phi])  # momenta
    h_params = np.array([M, m])
    solution = solve_hamiltonian(hamiltonian_double_orbit, t_span, initial=initial, h_params=h_params,
                                 method='DOP853', dense_output=True)

    tn = np.linspace(t_span[0], t_span[1], 10 ** 3)

    r1, phi1, r2, phi2 = solution.sol(tn)[:4, :]

    plt.plot(r1 * np.cos(phi1),
             r1 * np.sin(phi1),
             label='M body')
    plt.plot(r2 * np.cos(phi2),
             r2 * np.sin(phi2),
             label='m body')
    plt.xlim(-900, 300)
    plt.ylim(-600, 600)
    plt.xlabel('x-coordinate [au]')
    plt.ylabel('y-coordinate [au]')
    plt.title(f'two body problem {m=} {M=}')
    plt.legend()
    plt.show()

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(solution.t,
             (hamiltonian_double_orbit(solution.y[:4, :], solution.y[4:, :], h_params=h_params) /
              hamiltonian_double_orbit(solution.y[:4, 0], solution.y[4:, 0], h_params=h_params)),
             'k-',
             label='Energy')
    ax2.plot(solution.t,
             (solution.y[5, :] + solution.y[7, :]) / (solution.y[5, 0] + solution.y[7, 0]),
             'r-',
             label='angular momentum')
    ax1.set_xlabel('Time [au]')
    ax1.set_ylabel('Energy [$E_0$]')
    ax2.set_ylabel('Angular momentum [$l_0$]')
    ax1.legend(loc='upper center')
    ax2.legend(loc='upper right')
    plt.title('conserved quantities')
    plt.show()
    print(solution.y[1, 0])
    print(solution.y[1, -1])


def eih_plots():
    t_span = (0, 3 * 10 ** 6)
    mass_2 = 1
    mass_1 = 1
    reduced_m = (1 / mass_1 + 1 / mass_2) ** (-1)
    r_0 = 4000
    initial = np.array([r_0, 0, 0, 0.013 * reduced_m])
    h_params = np.array([1, 1, mass_1, mass_2])
    max_step = t_span[1] / 10000
    solution_eih = solve_hamiltonian(hamiltonian=hamiltonian_einstein_infeld_hoffmann, t_span=t_span,
                                     initial=initial, h_params=h_params,
                                     method='DOP853', dense_output=True, max_step=max_step)

    solution_newton = solve_hamiltonian(hamiltonian=hamiltonian_reduced_two_body, t_span=t_span,
                                        initial=initial, h_params=h_params,
                                        method='DOP853', dense_output=True, max_step=max_step)

    t_dense = np.linspace(t_span[0], t_span[1], 2 * 10 ** 6)
    r2_dense = solution_eih.sol(t_dense)[0, :] ** 2 + solution_eih.sol(t_dense)[1, :] ** 2
    phi_dense = np.arctan(solution_eih.sol(t_dense)[1, :] / solution_eih.sol(t_dense)[0, :])
    r2 = solution_eih.y[0, :] ** 2 + solution_eih.y[1, :] ** 2
    phi = np.arctan(solution_eih.y[1, :] / solution_eih.y[0, :])
    angular_momentum = initial[0] * initial[3]
    pred_delta_phi2 = (6 * np.pi * (h_params[0] * (mass_1 + mass_2) * reduced_m) ** 2
                       / angular_momentum ** 2)
    a = (r_0 - solution_newton.y[0, np.argmin(solution_newton.y[0, :])]) / 2
    b = (solution_newton.y[1, np.argmax(solution_newton.y[1, :])]
         - solution_newton.y[1, np.argmin(solution_newton.y[1, :])]) / 2
    pred_delta_phi = 6 * np.pi * (mass_2 + mass_1) * a / b ** 2
    obs_delta_phi = phi[argrelextrema(r2, np.greater)]
    obs_delta_phi_dense = phi_dense[argrelextrema(r2_dense, np.greater)]

    print(f"{obs_delta_phi=}")
    print(f"{obs_delta_phi_dense=}")
    print(f"{pred_delta_phi=}")
    print(f"{pred_delta_phi2=}")
    print(f"{(pred_delta_phi * np.arange(1, len(obs_delta_phi) + 1) - obs_delta_phi) / obs_delta_phi}")

    ratio_1 = (1 + mass_1 / mass_2) ** (-1)
    ratio_2 = (1 + mass_2 / mass_1) ** (-1)

    plt.plot(solution_eih.y[0, :] * ratio_1, solution_eih.y[1, :] * ratio_1, 'r-', label='particle 1 EIH')
    plt.plot(-solution_eih.y[0, :] * ratio_2, -solution_eih.y[1, :] * ratio_2, 'b-', label='particle 2 EIH')
    plt.plot(solution_newton.y[0, :] * ratio_1, solution_newton.y[1, :] * ratio_1, 'k--', label='particle 1 Newton')
    plt.plot(-solution_newton.y[0, :] * ratio_2, -solution_newton.y[1, :] * ratio_2, 'y--',
             label='particle 2 Newton')
    plt.legend()
    plt.title(f"Two body EIH {mass_1=} {mass_2=}")
    # plt.xlim((-500, 500))
    # plt.ylim((-500, 500))
    plt.xlabel("X-Coordinate [$\\mu ?$]")
    plt.ylabel("Y-Coordinate [$\\mu ?$]")
    plt.show()

    plt.plot((hamiltonian_einstein_infeld_hoffmann(solution_eih.y[:2, :], solution_eih.y[2:, :], h_params=h_params)
              / hamiltonian_einstein_infeld_hoffmann(solution_eih.y[:2, 0], solution_eih.y[2:, 0],
                                                     h_params=h_params)),
             'g-')
    plt.show()

    plt.plot(solution_eih.t,
             ((solution_eih.y[0, :] * solution_eih.y[3, :] - solution_eih.y[1, :] * solution_eih.y[2, :])
              / (solution_eih.y[0, 0] * solution_eih.y[3, 0] - solution_eih.y[1, 0] * solution_eih.y[2, 0]))
             )
    plt.show()


def main():

    harmonic_oscillator()
    # orbit()
    # two_bodies()
    # eih_plots()


if __name__ == '__main__':
    main()
