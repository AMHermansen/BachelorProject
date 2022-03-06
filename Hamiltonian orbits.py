import numpy as np
import matplotlib.pyplot as plt
import einsteinpy.utils.dual as dual
from einsteinpy.utils.dual import _deriv
from scipy.integrate import solve_ivp


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


def main():
    def hamiltonian_ho(positions, momenta, h_params):
        mass, k = h_params
        return momenta[0]**2 / (2 * mass) + k * positions[0]**2 / 2

    def hamiltonian_orbit(positions, momenta, h_params):
        M, m = h_params
        return (momenta[0]**2 + momenta[1]**2 * positions[0]**(-2)) / (2 * m) - M * m / positions[0]

    def hamiltonian_double_orbit(positions, momenta, h_params):
        M, m = h_params
        return ((momenta[0]**2 + momenta[1]**2 * positions[0]**(-2)) / (2 * M)
                + (momenta[2]**2 + momenta[3]**2 * positions[2]**(-2)) / (2 * m)
                - M * m / ((positions[0] * np.cos(positions[1]) - positions[2] * np.cos(positions[3]))**2
                           + (positions[0] * np.sin(positions[1]) - positions[2] * np.sin(positions[3]))**2)**0.5)

    def harmonic_oscillator():
        t_span = (0, 10)
        initial = np.array([10.0, 0.0])
        h_params = np.array([3.0, 2.0])

        solution = solve_hamiltonian(hamiltonian=hamiltonian_ho, t_span=t_span, initial=initial, h_params=h_params,
                                     method='DOP853', dense_output=True)
        xn = np.linspace(t_span[0], t_span[1], 100)
        plt.plot(xn, 10 * np.cos((2 / 3) ** 0.5 * xn), 'ko', label='exact')
        plt.plot(xn, solution.sol(xn)[0, :], 'r-', label='numerical')
        plt.xlabel('Time [au]')
        plt.ylabel('Position [au]')
        plt.title('Harmonic oscillator, m=3, k=2')
        plt.legend()
        plt.show()

    def orbit():
        t_span = (0, 10**4)
        initial = np.array([10**3, 0.0, 0.0, 0.8])
        h_params = np.array([1000.0, 0.001])

        solution = solve_hamiltonian(hamiltonian=hamiltonian_orbit, t_span=t_span, initial=initial, h_params=h_params,
                                     method='DOP853', dense_output=True, max_step=100)

        energy = hamiltonian_orbit(initial[:len(initial) // 2], initial[len(initial) // 2:], h_params=h_params)
        l = initial[-1]
        M, m = h_params

        eccentricity = (1 + (2 * energy * l**2) / (m**3 * M**2))**(1/2)
        C = m**2 * M / l**2
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
        t_span = (0, 8*10**4)
        initial = np.array([r_1, 0.0, r_2, np.pi,  # positions
                            0.0, p_phi * (m / M), 0.0, p_phi])  # momenta
        h_params = np.array([M, m])
        solution = solve_hamiltonian(hamiltonian_double_orbit, t_span, initial=initial, h_params=h_params,
                                     method='DOP853', dense_output=True)

        tn = np.linspace(t_span[0], t_span[1], 10**3)

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

    # harmonic_oscillator()
    # orbit()
    two_bodies()


if __name__ == '__main__':
    main()
