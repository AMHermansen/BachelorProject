import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate


def ode(t, state, L=1):
    """

    :param t: Time (there is no explicit time dependence in the ODE)
    :param state: (X, dX/d\phi)
    :param L: "Angular" momentum.
    :return: standard function of the ODE.
    G = M = 1.
    """
    return np.array([state[1],
                     L**(-2) - state[0] + 3 * state[0]**2])


def main():
    pass


if __name__ == '__main__':
    main()
