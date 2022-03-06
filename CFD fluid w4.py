import numpy as np
import scipy.linalg as lin
from math import factorial
import matplotlib.animation as ani
import matplotlib.pyplot as plt


def animate(data, inter=200, show=True):
    """ Show a frame-by-frame animation. \n
    Parameters:
        inter: time between frames (ms).
    """
    # create the figure
    fig = plt.figure()

    # the local animation function
    def animate_fn(n):
        # we want a fresh figure everytime
        fig.clf()
        # add subplot, aka axis
        ax = fig.add_subplot(111)
        # call the global function
        if data.ndim == 2:
            plot1D(data[n], ax)
        elif data.ndim == 3:
            plot2d(data[n], ax)
        else:
            raise ValueError('...')

    anim = ani.FuncAnimation(fig, animate_fn, frames=data.shape[0],
                             interval=inter, blit=False)
    if show == True:
        plt.show()
        return

    return anim


def save(an, fname, fps, tt='ffmpeg', bitrate=1800):
    writer = ani.writers[tt](fps=fps, bitrate=bitrate)
    an.save(fname, writer=writer)


def plot1D(data, engine=plt):
    engine.plot(np.linspace(0, 2, len(data)), data)
    engine.set_xlabel('x')
    engine.set_xlim([0, 2])
    engine.set_ylim([-1, 2])


def plot2d(data, engine=plt):
    im = engine.imshow(data.T, cmap='turbo', interpolation='lanczos',
                       origin='lower', vmin=0.0, vmax=1.0)
    plt.colorbar(im)


def find_scheme(a, n=1):
    N = len(a)
    A = np.stack([a**i for i in range(N)])
    b = np.zeros(N)
    b[n] = factorial(n)

    return lin.solve(A, b)


def advection_diffusion_1d(initial, points_x=5, length_x=1, time_range=(0, 0.5), t_steps=1 * 10**2, nu=0.01):
    def construct_operator(f_value):
        dx = length_x / points_x
        scheme1 = find_scheme(np.array([-dx, 0, dx]), n=1)
        scheme2 = find_scheme(np.array([-dx, 0, dx]), n=2)
        temp1 = np.zeros((points_x, points_x))
        temp2 = np.zeros((points_x, points_x))

        assert len(scheme1) == len(scheme2)

        for index in range(len(scheme1)):
            for i in range(points_x):
                temp1[i, (i + index - len(scheme1) // 2) % points_x] = scheme1[index] * f_value[i]
                temp2[i, (i + index - len(scheme2) // 2) % points_x] = scheme2[index]
        f_val_diag = np.diag(f_value)

        return temp1 - nu * temp2

    f_cur = initial
    all_f_vals = []
    delta_t = (time_range[1] - time_range[0]) / t_steps
    for _ in range(t_steps):
        all_f_vals.append(f_cur)
        operator = construct_operator(f_cur) * delta_t + np.identity(points_x)
        f_cur = lin.inv(operator) @ f_cur

    all_f_vals.append(f_cur)
    return np.array(all_f_vals)


def main():
    points_x = 100
    initial = np.sin(np.linspace(0, 4 * np.pi, points_x)) + 1
    result = advection_diffusion_1d(initial, points_x=points_x)
    print(result)
    animate(result)
    pass


if __name__ == '__main__':
    main()
