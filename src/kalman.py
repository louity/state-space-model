# coding: utf8
import numpy as np
from numpy.linalg import inv

def check_matrix(M, dim, err_message='wrong matrix dim'):
    if M is None:
        return np.identity(dim)
    elif M.size != dim:
        raise ValueError(err_message)
    else:
        return M

class LinearStateSpaceModel:
    """
    classe decrivant un modele suivant les equations :
        x_t+1 = A x_t + w_t, w_t zero mean with cov mat Q
        y_t = C x_t + v_t, v_t zero mean with cov mat R
    permet de faire du filtering et du smoothing
    """

    def __init__(self, state_dim=1, output_dim=1, Sigma_0=None, A=None, Q=None, C=None, R=None):
        self.state_dim = state_dim
        self.output_dim = output_dim
        self.Sigma_0 = check_matrix(Sigma_0, state_dim, 'matrix Sigma_0 sie must be equal to state_dim')
        self.A = check_matrix(A, state_dim, 'matrix A size must equal to state_dim')
        self.Q = check_matrix(Q, state_dim, 'matrix Q size must equal to state_dim')
        self.C = check_matrix(C, output_dim, 'matrix C size must equal to output_dim')
        self.R = check_matrix(R, output_dim, 'matrix R size must equal to output_dim')

    def kalman_filtering(self, output_sequence):
        t = len(output_sequence)
        self.output_sequence = output_sequence

        x_0 = np.zeros(self.state_dim)
        P_0 = self.Sigma_0

        self.state_means = [ x_0 ]
        self.state_covariance = [ P_0 ]

        A = self.A
        Q =self.Q
        C = self.C
        R = self.R
        AT = np.transpose(A)
        CT = np.transpose(C)

        for i in range(1, t):
            y = self.output_sequence[i]

            x_1_0 = A.dot(x_0)
            P_1_0 = A * P_0 * AT
            K = P_1_0 * CT * inv(C * P_1_0 * CT + R)






