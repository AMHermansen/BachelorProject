import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def S_metric(mu, nu, t, r, theta, phi):
    if mu == nu == 0:
        return -(1 - (1/r))
    elif mu == nu == 1:
        return (1 - (1/r))**(-1)
    elif mu == nu == 2:
        return r**2
    elif mu == nu == 3:
        return r**2 * np.sin(theta)**2
    else:
        return 0


def christoffel_symbols(i, j, k, t, r, theta, phi):
    """
    Expression for the christoffel symbol in Schwarzschild metric. \Gamma^i_{j,k}. The units G=c=r_0=1 are used
    :param i: Upper index (0,1,2,3)
    :param j: Lower index (0,1,2,3)
    :param k: Lower index (0,1,2,3)
    :param t: Time coordinate
    :param r: Radial coordinate
    :param theta: Angular coordinate (primarily use theta = np.pi/2)
    :param phi: Angular coordinate
    :return: Numerical expression for the i-j-k Christoffel symbol, with parameters (t, r, theta, phi)
    """
    if (i, j, k) == (1, 0, 0):
        return (r - 1) / (2 * r**3)
    elif ((i, j, k) == (0, 1, 0)) or ((i, k, j) == (0, 1, 0)):
        return 1 / (2 * r * (r - 1))
    elif (i, j, k) == (1, 1, 1):
        return - 1 / (2 * r * (r - 1))
    elif ((i, j, k) == (2, 2, 1)) or ((i, k, j) == (2, 2, 1)):
        return 1 / r
    elif ((i, j, k) == (3, 3, 1)) or ((i, k, j) == (3, 3, 1)):
        return 1 / r
    elif (i, j, k) == (1, 2, 2):
        return - (r - 1)
    elif ((i, j, k) == (2, 2, 1)) or ((i, k, j) == (2, 2, 1)):
        return np.cos(theta) / np.sin(theta)  # cot(theta)
    elif (i, j, k) == (1, 3, 3):
        return (1 - r) * np.sin(theta)**2
    elif (i, j, k) == (2, 3, 3):
        return - np.sin(theta) * np.cos(theta)
    else:
        return 0


def main():
    def ODE(tau, coords):
        """

        :param tau: Proper time
        :param coords: t, r, theta, phi, u_t, u_r, u_theta, u_phi
        :return: The geodesic equation written as an 8d first order ODE.
        """
        ddot_t = 0  # double derivative
        ddot_r = 0  # double derivative
        ddot_theta = 0  # double derivative
        ddot_phi = 0  # double derivative
        for j in range(4):
            for k in range(4):
                ddot_t -= christoffel_symbols(0, j, k, coords[0], coords[1], coords[2], coords[3])
                ddot_r -= christoffel_symbols(1, j, k, coords[0], coords[1], coords[2], coords[3])
                ddot_theta -= christoffel_symbols(2, j, k, coords[0], coords[1], coords[2], coords[3])
                ddot_phi -= christoffel_symbols(3, j, k, coords[0], coords[1], coords[2], coords[3])
        return np.array([
            coords[4],
            coords[5],
            coords[6],
            coords[7],
            ddot_t,
            ddot_r,
            ddot_theta,
            ddot_phi
        ])
    position = np.array([40, np.pi/2, 0])
    velocity = np.array([0, 0, 3.83405])
    initial_condition = get_full_state(position, velocity, S_metric)
    print(initial_condition)
    geodesic = solve_ivp(ODE, (0, 10**2), initial_condition, method="DOP853")
    plt.plot(geodesic.y[1] * np.cos(geodesic.y[3]), geodesic.y[1] * np.sin(geodesic.y[3]), 'b--')
    plt.show()


def get_full_state(q3, p3, metric):
    q4 = np.array([0., q3[0], q3[1], q3[2]])  # set t=0
    # solve second degree polynomial
    A = metric(0, 0, *q4)
    B = 2 * (p3[0] * metric(0, 1, *q4) + p3[1] * metric(0, 1, *q4) + p3[2] * metric(0, 1, *q4))
    C = (p3[0] * p3[0] * metric(1, 1, *q4) + p3[0] * p3[1] * metric(1, 2, *q4) + p3[0] * p3[2] * metric(1, 3, *q4)
         + p3[1] * p3[0] * metric(2, 1, *q4) + p3[1] * p3[1] * metric(2, 2, *q4) + p3[1] * p3[2] * metric(2, 3, *q4)
         + p3[2] * p3[0] * metric(3, 1, *q4) + p3[2] * p3[1] * metric(3, 2, *q4) + p3[2] * p3[2] * metric(3, 3, *q4)
         + 1)  # 1 since time-like geodesic
    print(q4)
    print(A, B, C)
    p4 = np.array([((-B + np.sqrt(B**2 - 4 * A * C))/(2*A)), p3[0], p3[1], p3[2]])
    return np.concatenate([q4, p4])


if __name__ == '__main__':
    position = np.array([0, 40, np.pi / 2, 0])
    for i in range(4):
        for j in range(4):
            print(S_metric(i, j, *position))
    main()

