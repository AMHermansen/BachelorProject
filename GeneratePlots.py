import numpy as np
import matplotlib.pyplot as plt
from Utilities import get_total_angular_momentum, split_position_momentum


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
        initial_energy = hamiltonian(solution.y[:number_of_coordinates, 0],
                                     solution.y[number_of_coordinates:, 0],
                                     h_params)
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
        position_coordinates, momentum_coordinates = split_position_momentum(solution=solution)
        initial_energy = hamiltonian(solution.y[:number_of_coordinates, 0],
                                     solution.y[number_of_coordinates:, 0],
                                     h_params)
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
        total_angular_momentum = get_total_angular_momentum(solution=solution,
                                                            position_pair_coordinates=positions,
                                                            momentum_pair_coordinates=momenta)
        plt.plot(solution.t, total_angular_momentum / total_angular_momentum[0], label=label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()


