import numpy as np
import matplotlib.pyplot as plt
from Hamiltonians import (hamiltonian_post_minkowski1, hamiltonian_post_minkowski2, hamiltonian_post_minkowski3,
                          hamiltonian_two_body, hamiltonian_sr_kinetic, hamiltonian_newton_kinetic,
                          hamiltonian_fake_pm1, hamiltonian_fake_pm2, hamiltonian_fake_pm3)
from Utilities import (get_solutions, get_angular_momentum, get_perihelion_shift, scattering_angle, get_scattering,
                       all_scattering_angle, all_scattering_angle_velocities,
                       time_averaged_mean)  # Might use later
from GeneratePlots import (generate_orbit_plot, generate_energy_plot, generate_split_energy_plot,
                           generate_angular_momentum_plot)
from Formulas import (scattering_classical, scattering_pm1, scattering_pm2, scattering_pm3, theoretical_perihelion_shift,
                      all_fake_scattering)
from Hamiltonians import hamiltonian_sr_newton_pot  # Maybe used in future

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


    # Plotting average energies
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

    pred_delta_phi = (6 * np.pi * (G * (mass_1 + mass_2) * reduced_mass) ** 2
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
        (all_solutions_no_pm1, all_hamiltonians_no_pm1, solution_names_no_pm1) if no_pm1
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
    print('Theoretical scattering pm3: ', scattering_pm3(solution=solution_pm3,
                                                         hamiltonian=hamiltonian_post_minkowski3,
                                                         h_params=h_params,
                                                         position_pair_coordinates=((0, 1), (2, 3)),
                                                         momentum_pair_coordinates=((4, 5), (6, 7))))
    print('Theoretical scattering classical: ', scattering_classical(solution=solution_classical, h_params=h_params))



def fake_pm_analysis_scattering(r, b, p, mass_1, t_span, mass_2=1., minimal_steps=10 ** 3, no_pm1=False):
    max_step = t_span[1] / minimal_steps
    factor_1 = mass_2 / (mass_1 + mass_2)
    factor_2 = - mass_1 / (mass_1 + mass_2)
    initial_real = np.array([factor_1 * np.sqrt(r ** 2 - b ** 2), factor_1 * b,  # Position 1
                             factor_2 * np.sqrt(r ** 2 - b ** 2), factor_2 * b,  # Position 2
                             - p, 0,  # Momentum 1
                             p, 0])  # Momentum 2
    initial_fake = np.array([np.sqrt(r**2 - b**2), b,
                             - p, 0])
    number_of_coordinates_fake = len(initial_fake) // 2  # number of position / momenta coordinates, is half the total.
    G = 1
    fake_h_params = G, mass_1, mass_2, p
    real_h_params = G, mass_1, mass_2
    all_fake_hamiltonians = (hamiltonian_fake_pm1, hamiltonian_fake_pm2, hamiltonian_fake_pm3)
    all_real_hamiltonians = (hamiltonian_post_minkowski1, hamiltonian_post_minkowski2, hamiltonian_post_minkowski3)

    all_solutions_fake = get_solutions(hamiltonians=all_fake_hamiltonians,
                                       t_span=t_span, initial=initial_fake, h_params=fake_h_params,
                                       method='DOP853', dense_output=True, max_step=max_step
                                       )
    all_solutions_real = get_solutions(hamiltonians=all_real_hamiltonians,
                                       t_span=t_span, initial=initial_real, h_params=real_h_params,
                                       method='DOP853', dense_output=True, max_step=max_step
                                       )
    solution_names_fake = ('Fake PM1', 'Fake PM2', 'Fake PM3')
    solution_names_real = ('PM1', 'PM2', 'PM3')
    orbit_labels_fake = [(f'{sol_name} Particle1', f'{sol_name} Particle2') for sol_name in solution_names_fake]
    orbit_labels_real = [(f'{sol_name} Particle1', f'{sol_name} Particle2') for sol_name in solution_names_real]

    generate_orbit_plot(solutions=all_solutions_fake,
                        legends=orbit_labels_fake,
                        x_indices=(0,), y_indices=(1,))
    plt.show()
    generate_orbit_plot(solutions=all_solutions_real,
                        legends=orbit_labels_real,
                        x_indices=(0,), y_indices=(1,))
    plt.show()
    scattering_fake1, scattering_fake2, scattering_fake3 = all_scattering_angle(solutions=all_solutions_fake,
                                                                                position_coordinates=(0, 1))
    v_scattering_fake1, v_scattering_fake2, v_scattering_fake3 = all_scattering_angle_velocities(solutions=all_solutions_fake,
                                                                                                 position_coordinates=(0, 1))

    print(f'{scattering_fake1=}')
    print(f'{scattering_fake2=}')
    print(f'{scattering_fake3=}')

    print(f'{v_scattering_fake1=}')
    print(f'{v_scattering_fake2=}')
    print(f'{v_scattering_fake3=}')

    print('Fake pm1 scattering angle: ', get_scattering(scattering_fake1))
    print('Fake pm2 scattering angle: ', get_scattering(scattering_fake2))
    print('Fake pm3 scattering angle: ', get_scattering(scattering_fake3))

    print('Fake pm1 v-scattering angle: ', get_scattering(v_scattering_fake1))
    print('Fake pm2 v-scattering angle: ', get_scattering(v_scattering_fake2))
    print('Fake pm3 v-scattering angle: ', get_scattering(v_scattering_fake3))

    t_scattering_fake1, t_scattering_fake2, t_scattering_fake3 = all_fake_scattering(positions=initial_fake[:number_of_coordinates_fake],
                                                                                     momenta=initial_fake[number_of_coordinates_fake:],
                                                                                     h_params=fake_h_params)

    print(f'{t_scattering_fake1=}')
    print(f'{t_scattering_fake2=}')
    print(f'{t_scattering_fake3=}')


def main():
    # post_minkowski_analysis_bound_orbit(r=8 * 10**1, p=0.08, t_span=(0, 7.5 * 10**3), mass_1=1)
    # m = 10 ** (-4)
    # post_minkowski_analysis_scattering(r=10 ** 4, b=1.5 * 10 ** 2, p=0.71 * m, mass_1=m, t_span=(0, 4 * 10 ** 4),
    #                                    mass_2=10.)  # Sprednings_vinkler
    # m = 10 ** (-5)
    # post_minkowski_analysis_scattering(r=10 ** 3, b=0.8 * 10 ** 1, p=0.767499 * m, mass_1=m, t_span=(0, 2 * 10 ** 3), mass_2=1., no_pm1=True)
    # post_minkowski_analysis_scattering(r=10 ** 4, b=10 ** 2, p=5*m, mass_1=m, mass_2=10**3)
    m = 10 ** (-5)
    fake_pm_analysis_scattering(r=10 ** 3, b=1.5 * 10 ** 1, p=0.71 * m, mass_1=m, t_span=(0, 4 * 10 ** 3), mass_2=1.)
    pass


if __name__ == '__main__':
    main()
