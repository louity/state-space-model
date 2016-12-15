# coding: utf8
import numpy as np
from numpy import power, exp
from numpy.linalg import inv, det
from numpy.random import multivariate_normal as mv_norm
import random
import math

DEFAULT_N_RBF = 10

def check_matrix(M, shape, err_message='wrong matrix shape'):
    if M is None:
        M = np.zeros(shape)
        for i in range(0, min(shape)):
            M[i, i] = 1.0
        return M
    elif M.shape != shape:
        raise ValueError(err_message)
    else:
        return M

def rbf(rbf_value, rbf_center, rbf_width_inv, x):
    dim = rbf_center.size
    K = 1.0 * det(rbf_width_inv) / power(2 * np.pi, dim / 2)
    v = (x - rbf_center)
    rbf_term = exp(-0.5 * v.transpose().dot(rbf_width_inv).dot(v))

    return K * rbf_value * rbf_term

def rbf_derivative(rbf_value, rbf_center, rbf_width_inv, x):
    v = np.matrix(x - rbf_center)
    rbf_vector = np.matrix(rbf(rbf_value, rbf_center, rbf_width_inv, x))

    return rbf_vector.T.dot(v).dot(rbf_width_inv)


class StateSpaceModel:
    """
    classe decrivant un modele suivant les equations :
        x_t+1 = sum_i(rho_i(x) h_i) + A x_t + B u_t + w_t, w_t zero mean with cov mat Q,rho_i RBF function
        y_t = C x_t + D u_t + v_t, v_t zero mean with cov mat R
    permet de faire du filtering et du smoothing
    """

    def __init__(self, isLinear=True, state_dim=None, input_dim=None, output_dim=None, Sigma_0=None, A=None, B=None, Q=None, C=None, D=None, R=None, rbf_parameters=None, rbf_coeffs=None):
        self.isLinear = isLinear

        if state_dim is None:
            print 'No state space imension given, default set to 1'
            self.state_dim = 1
        else:
            self.state_dim = state_dim
        if input_dim is None:
            print 'No input space dimension given, default set to 1'
            self.input_dim = 1
        else:
            self.input_dim = input_dim
        if output_dim is None:
            print 'No output space dimension given, default set to 1'
            self.output_dim = 1
        else:
            self.output_dim = output_dim

        self.Sigma_0 = check_matrix(Sigma_0, (self.state_dim, self.state_dim), 'matrix Sigma_0 shape must be equal to self.state_dim')
        self.A = check_matrix(A, (self.state_dim, self.state_dim), 'matrix A shape must equal to self.state_dim x self.state_dim')
        self.B = check_matrix(B, (self.state_dim, self.input_dim), 'matrix B shape must equal to self.state_dim x self.input_dim')
        self.Q = check_matrix(Q, (self.state_dim, self.state_dim), 'matrix Q shape must equal to self.state_dim x self.state_dim')
        self.C = check_matrix(C, (self.output_dim, self.state_dim), 'matrix C shape must equal to self.output_dim x self.state_dim')
        self.D = check_matrix(D, (self.output_dim, self.input_dim), 'matrix D shape must equal to self.output_dim x self.input_dim')
        self.R = check_matrix(R, (self.output_dim, self.output_dim), 'matrix R shape must equal to self.output_dim')
        self.output_sequence = None
        self.state_sequence = None
        self.input_sequence = None

        if not self.isLinear:
            if rbf_parameters is None:
                self.draw_sample(10 * DEFAULT_N_RBF)
                self.kalman_smoothing()
                self.rbf_parameters = {
                    'n_rbf': DEFAULT_N_RBF,
                    'centers': random.sample(self.smoothed_state_means, DEFAULT_N_RBF),
                    'width': random.sample(self.smoothed_state_covariance, DEFAULT_N_RBF)
                }
            else:
                self.rbf_parameters = rbf_parameters

            if rbf_coeffs is None:
                self.rbf_coeffs = []
                for i in range(0, self.rbf_parameters['n_rbf']):
                    self.rbf_coeffs.append(np.ones(self.state_dim))
            else:
                self.rbf_coeffs = rbf_coeffs

    def compute_f(self, x, u=None):
        if x.size != self.state_dim:
            raise ValueError('x vector must have state dimension')
        elif u is None:
            u = np.zeros(self.input_dim)
        elif u.size != self.input_dim:
            raise ValueError('u vector must have state dimension')

        f = self.A.dot(x) + self.B.dot(u)

        for i in range(0, self.rbf_parameters['n_rbf']):
            center = self.rbf_parameters['centers'][i]
            width = self.rbf_parameters['width'][i]
            value = self.rbf_coeffs[i]
            f += rbf(value, center, inv(width), x)

        return f

    def compute_f_derivative(self, x, u=None):
        if x.size != self.state_dim:
            raise ValueError('x vector must have state dimension')
        elif u is None:
            u = np.zeros(self.input_dim)
        elif u.size != self.input_dim:
            raise ValueError('u vector must have state dimension')

        df = self.A

        for i in range(0, self.rbf_parameters['n_rbf']):
            center = self.rbf_parameters['centers'][i]
            width = self.rbf_parameters['width'][i]
            value = self.rbf_coeffs[i]
            df += rbf_derivative(value, center, inv(width), x)

        return df

    def kalman_filtering(self, is_extended=False, input_sequence=None, output_sequence=None):
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

        if input_sequence is None and self.input_sequence is None:
            print 'WARNING: no input sequence'
            self.input_sequence = []
            for i in range(0,t):
                self.input_sequence.append(np.zeros(self.input_dim))

        # simplify notations
        A = self.A
        B = self.B
        Q = self.Q
        C = self.C
        D = self.D
        R = self.R
        AT = np.transpose(A)
        CT = np.transpose(C)

        self.filtered_state_means = []
        self.filtered_state_covariance = []

        for i in range(0, t):
            y = self.output_sequence[i]
            u = self.input_sequence[i]

            if i == 0:
                #initialization
                x_1_0 = np.zeros(self.state_dim)
                P_1_0 = self.Sigma_0
            else:
                x_1_0 = A.dot(self.filtered_state_means[i-1][1]) + B.dot(self.input_sequence[i-1])
                P_1_0 = A.dot(self.filtered_state_covariance[i-1][1]).dot(AT) + Q

            # kalma gain matrix
            K = P_1_0.dot(CT).dot(inv(C.dot(P_1_0).dot(CT) + R))
            x_1_1 = x_1_0 + K.dot(y - (C.dot(x_1_0) + D.dot(u)))
            P_1_1 = P_1_0 - K.dot(C).dot(P_1_0)

            self.filtered_state_means.append([x_1_0, x_1_1])
            self.filtered_state_covariance.append([P_1_0, P_1_1])

    def kalman_smoothing(self, is_extended=False, output_sequence=None):
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

                L = P_t_t.dot(AT).dot(inv(P_t_plus_1_t))
                LT = np.transpose(L)

                x_t_T = x_t_t + L.dot(x_t_plus_1_T - x_t_plus_1_t)
                P_t_T = P_t_t + L.dot(P_t_plus_1_T - P_t_plus_1_t).dot(LT)

            self.smoothed_state_means.insert(0, x_t_T)
            self.smoothed_state_covariance.insert(0, P_t_T)

    def draw_sample(self, T=1, input_sequence=None):
        if input_sequence is None and (self.input_sequence is None or len(self.input_sequence) < T):
            print 'No input sequence given, setting inputs to zero'
            self.input_sequence = []
            for i in range(0,T):
                self.input_sequence.append(np.zeros(self.input_dim))

        states = []
        outputs = []
        inputs = self.input_sequence


        x_1 = mv_norm(np.zeros(self.state_dim), self.Sigma_0)
        y_1 = mv_norm(self.C.dot(x_1) + self.D.dot(inputs[0]), self.R)

        states.append(x_1)
        outputs.append(y_1)

        for i in range(1,T):
            x_i = mv_norm(self.A.dot(states[i-1]) + self.B.dot(inputs[i-1]), self.Q)
            y_i = mv_norm(self.C.dot(x_i) + self.D.dot(inputs[i]), self.R)
            states.append(x_i)
            outputs.append(y_i)

        self.output_sequence = outputs
        self.state_sequence = states
