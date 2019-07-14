# -*- coding: utf-8 -*-

"""
Use this file for your answers. 

This file should been in the root of the repository 
(do not move it or change the file name) 

"""

import numpy as np


def gauss(x, mean, covariance_matrix):
    k = len(x)
    const = 1.0 / np.sqrt((2 * np.pi) ** k * np.linalg.det(covariance_matrix))
    e_term = np.exp(-0.5 * np.dot(np.dot(np.transpose(x - mean), np.linalg.inv(covariance_matrix)), x - mean))
    return const * e_term


def lml(alpha, beta, Phi, Y):
    """
    4 marks

    :param alpha: float
    :param beta: float
    :param Phi: array of shape (N, M)
    :param Y: array of shape (N, 1)
    :return: the log marginal likelihood, a scalar
    """

    mean = 0.0
    outer_prod = np.dot(Phi, np.transpose(Phi))
    covariance_matrix = alpha * outer_prod + beta * np.identity(np.shape(outer_prod)[0])
    return np.float(np.log(gauss(Y, mean, covariance_matrix)))


def grad_lml(alpha, beta, Phi, Y):
    """
    8 marks (4 for each component)

    :param alpha: float
    :param beta: float
    :param Phi: array of shape (N, M)
    :param Y: array of shape (N, 1)
    :return: array of shape (2,). The components of this array are the gradients
    (d_lml_d_alpha, d_lml_d_beta), the gradients of lml with respect to alpha and beta respectively.
    """

    a_deriv = np.dot(Phi, np.transpose(Phi))
    b_deriv = np.identity(np.shape(Phi)[0])

    mat = alpha * a_deriv + beta * b_deriv
    mat_inv = np.linalg.inv(mat)

    def grad_lml_1(deriv):
        return -0.5 * np.trace(np.dot(mat_inv, deriv))

    def grad_lml_2(deriv):
        return 0.5 * np.dot(np.dot(np.dot(np.dot(np.transpose(Y), mat_inv), deriv), mat_inv), Y)

    a_grad = grad_lml_1(a_deriv) + grad_lml_2(a_deriv)
    b_grad = grad_lml_1(b_deriv) + grad_lml_2(b_deriv)

    return np.array([a_grad, b_grad]).reshape((2, ))


if __name__ == '__main__':
    print(grad_lml(2, 3, np.array([[1, 2], [3, 4], [4, 5]]), np.array([5, 6, 6])))
