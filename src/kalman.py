# coding: utf8
import numpy as np
from numpy.linalg import inv
from numpy.random import multivariate_normal as mv_norm

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
        self.output_sequence = None
        self.state_sequence = None

    def kalman_filtering(self, output_sequence=None):
        """
            etant donne une sequence [y_1, ..., y_t], calcule de façon dynamique
            les moyennes et covariances de probabilités gaussiennes
            p(x_k|y_1, ... , y_k) et p(x_k|y_1, ... , y_k-1) pour k=1...t
            stocke les resultats dans self.filtered_state_means et self.filtered_state_covariance
        """
        if output_sequence is None and self.output_sequence is None:
            raise ValueError('Can not do filtering if output_sequence is None')
        elif output_sequence is not None:
            self.output_sequence = output_sequence

        t = len(self.output_sequence)

        # simplify notations
        A = self.A
        Q = self.Q
        C = self.C
        R = self.R
        AT = np.transpose(A)
        CT = np.transpose(C)

        self.filtered_state_means = []
        self.filtered_state_covariance = []

        for i in range(0, t):
            y = self.output_sequence[i]

            if i == 0:
                #initialization
                x_1_0 = np.zeros(self.state_dim)
                P_1_0 = self.Sigma_0
            else:
                x_1_0 = A.dot(self.filtered_state_means[i-1][1])
                P_1_0 = A * self.filtered_state_covariance[i-1][1] * AT + Q

            # kalma gain matrix
            K = P_1_0 * CT * inv(C * P_1_0 * CT + R)
            x_1_1 = x_1_0 + K.dot(y - C.dot(x_1_0))
            P_1_1 = P_1_0 - K  * C * P_1_0

            self.filtered_state_means.append([x_1_0, x_1_1])
            self.filtered_state_covariance.append([P_1_0, P_1_1])

    def kalman_smoothing(self, output_sequence=None):
        """
            etant donne une sequence [y_1, ..., y_T], calcule de façon dynamique
            les moyennes et covariances de probabilités gaussiennes
            p(x_t|y_1, ... , y_T) pour t=1...T
            stocke les resultats dans self.smoothed_state_means et self.smoothed_state_covariance
        """
        self.kalman_filtering(output_sequence)

        T = len(self.output_sequence)

        self.smoothed_state_means = []
        self.smoothed_state_covariance = []

        AT = np.transpose(self.A)

        for i in range(T, 0, -1):
            if i == T:
                x_t_T = self.filtered_state_means[i-1][1]
                P_t_T = self.filtered_state_covariance[i-1][1]
            else:
                P_t_t = self.filtered_state_covariance[i-1][1]
                P_t_plus_1_t = self.filtered_state_covariance[i][0]
                P_t_plus_1_T = self.smoothed_state_covariance[0]
                x_t_t = self.filtered_state_means[i-1][1]
                x_t_plus_1_t = self.filtered_state_means[i][0]
                x_t_plus_1_T = self.smoothed_state_means[0]

                L =  P_t_t * AT * inv(P_t_plus_1_t)
                LT = np.transpose(L)

                x_t_T = x_t_t + L.dot(x_t_plus_1_T - x_t_plus_1_t)
                P_t_T = P_t_t + L * (P_t_plus_1_T - P_t_plus_1_t) * LT

            self.smoothed_state_means.insert(0, x_t_T)
            self.smoothed_state_covariance.insert(0, P_t_T)

    def draw_sample(self, T=1):
        states = []
        outputs = []
        x_1 = mv_norm(np.zeros(self.state_dim), self.Sigma_0)
        y_1 = mv_norm(self.C.dot(x_1), self.R)

        states.append(x_1)
        outputs.append(y_1)

        for i in range(1,T):
            x_i = mv_norm(self.A.dot(states[i-1]), self.Q)
            y_i = mv_norm(self.C.dot(x_i), self.R)
            states.append(x_i)
            outputs.append(y_i)

        self.output_sequence = outputs
        self.state_sequence = states
