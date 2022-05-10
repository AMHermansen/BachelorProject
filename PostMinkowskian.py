import numpy as np
import matplotlib.pyplot as plt
from HamiltonianOrbits import solve_hamiltonian


def hamiltonian_einstein_infeld_hoffmann(positions, momenta, h_params):
    G, c, M, m = h_params
    m_inv = 1 / m
    M_inv = 1 / M
    p2 = momenta[0] ** 2 + momenta[1] ** 2
    r = (positions[0] ** 2 + positions[1] ** 2) ** 0.5
    return (
            (p2 / 2) * (m_inv + M_inv) - G * m * M / r
            + p2 ** 2 / (8 * c ** 2) * (m_inv ** 3 + M_inv ** 3)
            - G / (2 * c ** 2 * r) * (3 * p2 * (M * m_inv + m * M_inv) + 7 * p2
                                      + ((momenta[0] * positions[0] + momenta[1] * positions[1]) / r) ** 2)
            + G ** 2 * m * M * (m + M) / (2 * c ** 2 * r ** 2) 
    )


def post_minkowski_hamilton(positions, momenta, h_params):
    G, c, mass_1, mass_2 = h_params
    c_1 = mass_1**2 * mass_2**2 - 2 * ()


def main():
    pass


if __name__ == '__main__':
    main()
