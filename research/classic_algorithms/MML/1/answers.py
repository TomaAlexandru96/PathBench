# -*- coding: utf-8 -*-

"""
Use this file for your answers.

This file should been in the root of the repository
(do not move it or change the file name)

"""

# NB this is tested on python 2.7. Watch out for integer division

import numpy as np
import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt


def grad_f1(x):
    """
    4 marks

    :param x: input array with shape (2, )
    :return: the gradient of f1, with shape (2, )
    """

    return np.array([
        8 * x[0] - 2 * x[1] - 1,
        8 * x[1] - 2 * x[0] - 1
    ])


def grad_f2(x):
    """
    6 marks

    :param x: input array with shape (2, )
    :return: the gradient of f2, with shape (2, )
    """
    return np.array([
        np.cos(x[0] ** 2 + x[1] ** 2 - 2 * x[0] + 1) * (2 * x[0] - 2) + 6 * x[0] - 2 * x[1] - 2,
        np.cos(x[0] ** 2 + x[1] ** 2 - 2 * x[0] + 1) * 2 * x[1] + 6 * x[1] - 2 * x[0] + 6
    ])


def grad_f3(x):
    """
    This question is optional. The test will still run (so you can see if you are correct by
    looking at the testResults.txt file), but the marks are for grad_f1 and grad_f2 only.

    Do not delete this function.

    :param x: input array with shape (2, )
    :return: the gradient of f3, with shape (2, )
    """

    grad_f3_func = grad(f3)
    return grad_f3_func(x)


def f1(x):
    return 4 * x[0] ** 2 + 4 * x[1] ** 2 - 2 * x[0] * x[1] - x[0] - x[1]


def f2(x):
    return np.sin(x[0] ** 2 + x[1] ** 2 - 2 * x[0] + 1) + \
            3 * x[0] ** 2 + 3 * x[1] ** 2 - 2 * x[0] * x[1] - 2 * x[0] + 6 * x[1] + 3


def f3(x):
    a = np.exp(-((x[0] - 1) ** 2 + x[1] ** 2))
    b = np.exp(-(x[0] * (3 * x[0] - x[1] - 1) + (x[1] + 1) * (-x[0] + 3 * x[1] + 3)))
    c = 0.1 * np.log(
        (0.01 + x[0] ** 2) * (0.01 + x[1] ** 2) - x[0] ** 2 * x[1] ** 2
    )
    return 1 - (a + b - c)


def gradient_descent(iterations, fun, gradient, starting_point, step_size, name):
    x = starting_point
    xs = [starting_point]

    for i in range(iterations):
        x = x - step_size * gradient(x)
        xs.append(x)

    extents = list(map(lambda a: a * 1, [-1, 1, -2, 0]))
    X = np.linspace(extents[0], extents[1])
    Y = np.linspace(extents[2], extents[3])
    X, Y = np.meshgrid(X, Y)

    Z = fun(np.array([X, Y]))

    plot(X, Y, Z, xs, name, step_size)


def plot(X, Y, Z, xs, name, step_size):
    # plt.imshow(Z, extent=extents, origin='lower', cmap='Greens', alpha=0.8)
    scatter = plt.scatter(list(map(lambda a: a[0], xs)), list(map(lambda a: a[1], xs)), c=range(51), s=20, cmap="winter")
    contours = plt.contour(X, Y, Z, levels=30, cmap="seismic")
    plt.title(name + " Gradient descent with step size " + str(step_size))
    plt.xlabel("X")
    plt.ylabel("Y")
    sbar = plt.colorbar(scatter)
    cbar = plt.colorbar()
    #sbar.ax.set_ylabel("iterations")
    #cbar.ax.set_ylabel(name + " value")
    #plt.clabel(contours, inline=True, fontsize=8)
    plt.show()


# print(f3(np.array([0., -1.])))
# print(grad_f3(np.array([-1., 1.])))
# gradient_descent(50, f3, grad_f3, np.array([1., -1.]), 3, "f3")
# step size > 0.3 diverges

if __name__ == '__main__':
    gradient_descent(50, f2, grad_f2, np.array([1., -1.]), 0.003, "")
