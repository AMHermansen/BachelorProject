import numpy as np
from einsteinpy.geodesic import Timelike
from einsteinpy.plotting import GeodesicPlotter
from einsteinpy.integrators import GeodesicIntegrator
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema


def precesion():
    position = [200, np.pi/2, 0.]
    momentum = [0., 0., 13]
    a = 0

    geod = Timelike(
        metric="Schwarzschild",
        metric_params=(a,),
        position=position,
        momentum=momentum,
        steps=1.2*10**4,
        delta=1,
        order=2,
        omega=1,
        return_cartesian=False
    )
    print(geod.trajectory[1][argrelextrema(geod.trajectory[1][:, 1], np.greater)[0], 3])
    gpl = GeodesicPlotter()
    gpl.plot2D(geod, coordinates=(1, 2))
    gpl.show()


def scattering(r=1*10**2, b=10**1, v=0.425):
    def gamma(velocity):
        return 1 / np.sqrt(1 - velocity**2)

    def metric_rr(rad):
        return 1 / (1 - 2 / rad)

    phi = np.arcsin(b/r)
    p = v * gamma(v)  # Uses "conjugate momentum", need to fix this
    print(p)
    position = np.array([r, np.pi/2, phi])
    momentum = np.array([-p * np.sqrt(1 - (b / r)**2) * metric_rr(r)**0.5, 0, b * p])
    a = 0

    geod = Timelike(
        metric="Schwarzschild",
        metric_params=(a,),
        position=position,
        momentum=momentum,
        steps=1.5 * 10 ** 3,
        delta=0.2,
        return_cartesian=True,
        order=4,
        omega=0.003
    )
    gpl = GeodesicPlotter()
    gpl.plot2D(geod, coordinates=(1, 2))
    gpl.show()
    print(geod)


def custom_metric_test():
    def metric_contra1(coords, *g_prms):
        return -np.array([
                        [(1 - 2/coords[1]) ** (-1), 0, 0, 0],
                        [0, (-1 - 2/coords[1]) ** (-1), 0, 0],
                        [0, 0, (-coords[1]**2) ** (-1), 0],
                        [0, 0, 0, (-(coords[1] * np.sin(coords[2]))**2) ** (-1)]
                        ])

    def metric_contra2(coords, *g_prms):
        return -1 * np.array([
            [(1 - 2 / coords[1]) ** (-1), 0, 0, 0],
            [0, (-1 - 2 / coords[1] - (2 / coords[1]) ** 2) ** (-1), 0, 0],
            [0, 0, -(coords[1] ** 2) ** (-1), 0],
            [0, 0, 0, -((coords[1] * np.sin(coords[2])) ** 2) ** (-1)]
        ])

    def metric_contra3(coords, *g_prms):
        return - np.array([
            [(1 - 2 / coords[1]) ** (-1), 0, 0, 0],
            [0, (-1 - 2 / coords[1] - (2 / coords[1]) ** 2 - (2 / coords[1]) ** 3) ** (-1), 0, 0],
            [0, 0, (-coords[1] ** 2) ** (-1), 0],
            [0, 0, 0, (-(coords[1] * np.sin(coords[2])) ** 2) ** (-1)]
        ])

    def metric_contra_exact(coords, *g_prms):
        return - np.array([
            [(1 - 2 / coords[1]) ** (-1), 0, 0, 0],
            [0, -(1 - 2 / coords[1]), 0, 0],
            [0, 0, (-coords[1] ** 2) ** (-1), 0],
            [0, 0, 0, (-(coords[1] * np.sin(coords[2])) ** 2) ** (-1)]
        ])

    def compute_path(metric, q3, p3, steps=3*10**3, return_cartesian=True, pos_3=True, momentum_3=True):
        """
        :param metric: The contravariant metric for which the geodesic is to be calculated
        (Predominantly positive metric is expected)
        :param q3: The 3-position or the 4 position if pos_3=False. For the initial location of the particle
        :param p3: The 3-momentum (covariant velocity) or the 4 position if momentum_4=False.
        Used for the initial condition, if 4-momentum is used, it must satisfy the regularity condition:
        $p_\mu p_\nu g^{\mu \nu} = 1$. Lightlike isn't implemented.
        geodesic is to be calculated.
        :param steps: The number of steps, that are used in the Fantasy integrator.
        :param return_cartesian: If the coordinates returned are cartesian.
        (Only works if the original coordinates are spherical)
        :param pos_3 toggles if the initial condition is a 3-position or 4-position
        :param momentum_3 toggles if the initial condition is a 3-momentum or 4-momentum
        :return:
        """
        if pos_3:
            q4 = np.array([0, *q3])
        else:
            q4 = q3
        if momentum_3:
            p4 = np.array([0, *p3])
        else:
            p4 = p3
        A = metric(q4)[0, 0]
        B = 2*(metric(q4)[0, 1] * p4[1] + metric(q4)[0, 2] * p4[2] + metric(q4)[0, 3] * p4[3])
        C = 1  # Timelike constant
        for i in range(1, 4):
            for j in range(1, 4):
                C += metric(q4)[i, j] * p4[i] * p4[j]
        p4[0] = (- B + np.sqrt(B**2 - 4 * A * C)) / (2 * A)


        geodint = GeodesicIntegrator(
            metric=metric,
            metric_params=(0,),
            q0=q4,
            p0=p4,
        )
        for _ in range(steps):
            geodint.step()

        vecs = np.array(geodint.results, dtype=float)

        position1, momentum1 = vecs[:, 0], vecs[:, 1]
        # Ignoring (For a correct solution it should be a duplicate)
        # position2, momentum2 = vecs[:, 2], vecs[:, 3]
        if return_cartesian:
            t, r, th, ph = position1.T
            x = r * np.sin(th) * np.cos(ph)
            y = r * np.sin(th) * np.sin(ph)
            z = r * np.cos(th)
            position1 = np.vstack((t, x, y, z))
        return steps, position1, momentum1

    steps1, pos1, momentum1 = compute_path(metric_contra1,
                                           q3=np.array([40, np.pi/2, 0]),
                                           p3=np.array([0, 0, 3.853]))
    x1, y1 = pos1[1, :], pos1[2, :]

    steps2, pos2, momentum2 = compute_path(metric_contra2,
                                           q3=np.array([40, np.pi / 2, 0]),
                                           p3=np.array([0, 0, 3.853]))
    x2, y2 = pos2[1, :], pos2[2, :]

    steps3, pos3, momentum3 = compute_path(metric_contra3,
                                           q3=np.array([40, np.pi / 2, 0]),
                                           p3=np.array([0, 0, 3.853]))
    x3, y3 = pos3[1, :], pos3[2, :]

    steps_exact, pos_exact, momentum_exact = compute_path(metric_contra_exact,
                                                          q3=np.array([40, np.pi / 2, 0]),
                                                          p3=np.array([0, 0, 3.853]))
    x_exact, y_exact = pos_exact[1, :], pos_exact[2, :]

    blackhole = plt.Circle((0, 0), 2, color='k')
    fig, ax = plt.subplots()
    ax.plot(x1, y1, 'b--', label='First oder')
    ax.plot(x2, y2, 'g--', label='Second order')
    ax.plot(x3, y3, 'r--', label='Third order')
    ax.plot(x_exact, y_exact, 'k--', label='Exact solution')
    ax.add_patch(blackhole)
    plt.title('$r_0$=40, L=3.853')
    plt.xlim((-50, 50))
    plt.ylim((-50, 50))
    plt.xlabel('x coordinate $[M]$')
    plt.ylabel('y coordinate $[M]$')
    plt.legend()
    plt.show()


def main():
    precesion()
    # scattering()
    # custom_metric_test()
    pass


if __name__ == '__main__':
    main()
