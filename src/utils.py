# coding: utf8

import numpy as np
from numpy import power, exp, sqrt, log
from numpy.linalg import inv, det, eigvals

def is_pos_def(M):
    return np.all(eigvals(M) > 0)

def check_vector(v, size, err_message='wrong vector size'):
    if v is None:
        v = np.zeros(size)
        return v
    elif v.size != size:
        raise ValueError(err_message)
    else:
        return v

def check_matrix(M, shape, err_message='wrong matrix shape'):
    if M is None:
        M = np.zeros(shape) # Pourquoi pas un np.eye(shape) ? fonctionne avec matrice rectangulaire aussi
        for i in range(0, min(shape)):
            M[i, i] = 1.0
        return M
    elif M.shape != shape:
        raise ValueError(err_message)
    else:
        return M

def rbf(rbf_value, rbf_center, rbf_width_inv, x):
    """
        calcule h_i rho_{i}(x) où rho_i est la RBF de parametres (rbf_center,rbf_width)
    """
    if (rbf_center.size != x.size or rbf_width_inv.shape != (x.size, x.size)):
        raise ValueError('Dimensions of rbf center and width do not match')

    dim = x.size
    K = sqrt(power(2 * np.pi, -dim) * det(rbf_width_inv))
    v = (x - rbf_center)
    rbf_term = exp(-0.5 * v.dot(rbf_width_inv).dot(v))

    return K * rbf_value * rbf_term

def rbf_derivative(rbf_value, rbf_center, rbf_width_inv, x):
    """
        calcule d h_i rho_{i} / dx
    """
    rbf_vector = rbf(rbf_value, rbf_center, rbf_width_inv, x)[:, np.newaxis]
    v = (x - rbf_center)[np.newaxis, :]

    return rbf_vector.dot(v).dot(rbf_width_inv)
