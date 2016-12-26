# coding: utf8
import numpy as np
from numpy import power, exp
from numpy.linalg import inv, det
from numpy.random import multivariate_normal as mv_norm
import random
import math

DEFAULT_N_RBF = 10

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
    dim = rbf_center.size
    K = 1.0 * det(rbf_width_inv) / power(2 * np.pi, dim / 2)
    v = (x - rbf_center)
    rbf_term = exp(-0.5 * v.dot(rbf_width_inv).dot(v))

    return K * rbf_value * rbf_term

def rbf_derivative(rbf_value, rbf_center, rbf_width_inv, x):
    rbf_vector = rbf(rbf_value, rbf_center, rbf_width_inv, x)[:, np.newaxis]
    v = (x - rbf_center)[np.newaxis, :]

    return rbf_vector.dot(v).dot(rbf_width_inv)


class StateSpaceModel:
    """
    classe decrivant un modele suivant les equations :
        x_t+1 = sum_i(rho_i(x) h_i) + A x_t + B u_t + b + w_t, w_t zero mean with cov mat Q,rho_i RBF function
        y_t = sum_j(rho_j(x) k_j) + C x_t + D u_t + d + v_t, v_t zero mean with cov mat R
    permet de faire du filtering et du smoothing
    """

    def __init__(self, is_f_linear=True, is_g_linear=True, state_dim=None, input_dim=None, output_dim=None, Sigma_0=None, A=None, B=None, b=None, Q=None, C=None, D=None, d=None, R=None, f_rbf_parameters=None, f_rbf_coeffs=None, g_rbf_parameters=None, g_rbf_coeffs=None):
        self.is_f_linear = is_f_linear
        self.is_g_linear = is_g_linear

        if state_dim is None:
            print 'No state space imension given, default set to 1'
            self.state_dim = 1
        else:
            self.state_dim = state_dim
        if input_dim is None:
            print 'No input space dimension given, default set to 0'
            self.input_dim = 0
        else:
            self.input_dim = input_dim
        if output_dim is None:
            print 'No output space dimension given, default set to 1'
            self.output_dim = 1
        else:
            self.output_dim = output_dim

        self.b = check_vector(b, self.state_dim, 'vector b size must be equal to state_dim')
        self.d = check_vector(d, self.output_dim, 'vector d size must de equal to output_dim')

        self.Sigma_0 = check_matrix(Sigma_0, (self.state_dim, self.state_dim), 'matrix Sigma_0 shape must be equal to self.state_dim')
        self.A = check_matrix(A, (self.state_dim, self.state_dim), 'matrix A shape must equal to state_dim x state_dim')
        self.B = check_matrix(B, (self.state_dim, self.input_dim), 'matrix B shape must equal to state_dim x input_dim') if (self.input_dim > 0) else None
        self.Q = check_matrix(Q, (self.state_dim, self.state_dim), 'matrix Q shape must equal to state_dim x state_dim')
        self.C = check_matrix(C, (self.output_dim, self.state_dim), 'matrix C shape must equal to output_dim x state_dim')
        self.D = check_matrix(D, (self.output_dim, self.input_dim), 'matrix D shape must equal to output_dim x input_dim') if (self.input_dim > 0) else None
        self.R = check_matrix(R, (self.output_dim, self.output_dim), 'matrix R shape must equal to self.output_dim')

        self.f_rbf_parameters = f_rbf_parameters
        self.g_rbf_parameters = g_rbf_parameters
        self.f_rbf_coeffs = f_rbf_coeffs
        self.g_rbf_coeffs = g_rbf_coeffs

        self.output_sequence = None
        self.state_sequence = None
        self.input_sequence = None

        if not self.is_f_linear and self.f_rbf_parameters is None:
            print 'No rbf parameters provided for f, initialize them'
            self.initialize_f_rbf_parameters()
        if not self.is_f_linear and self.f_rbf_coeffs is None:
            self.f_rbf_coeffs = [np.ones(self.state_dim) for _ in range(0, self.f_rbf_parameters['n_rbf'])]

        if not self.is_g_linear and self.g_rbf_parameters is None:
            print 'No rbf parameters provided for g, initialize them '
            self.initialize_g_rbf_parameters()
        if not self.is_g_linear and self.g_rbf_coeffs is None:
            self.g_rbf_coeffs = [np.ones(self.output_dim) for _ in range(0, self.g_rbf_parameters['n_rbf'])]

    def get_rbf_parameters_for_state(self):
        is_f_linear = self.is_f_linear
        is_g_linear = self.is_g_linear

        # make f and g linear
        self.is_f_linear = True
        self.is_g_linear = True

        self.draw_sample(10 * DEFAULT_N_RBF)
        self.kalman_smoothing()

        self.is_f_linear = is_f_linear
        self.is_g_linear = is_g_linear

        return {
            'n_rbf': DEFAULT_N_RBF,
            'centers': random.sample(self.smoothed_state_means, DEFAULT_N_RBF),#TODO : replace random selection by k-means
            'width': random.sample(self.smoothed_state_covariance, DEFAULT_N_RBF)
        }

    def initialize_f_rbf_parameters(self):
        if self.g_rbf_parameters is not None:
            self.f_rbf_parameters = self.g_rbf_parameters
        else:
            self.f_rbf_parameters = self.get_rbf_parameters_for_state()

    def initialize_g_rbf_parameters(self):
        if self.f_rbf_parameters is not None:
            self.g_rbf_parameters = self.f_rbf_parameters
        else:
            self.g_rbf_parameters = self.get_rbf_parameters_for_state()

    def compute_f(self, x, u=None):
        if x.size != self.state_dim:
            raise ValueError('x vector must have state dimension')
        elif u is None:
            u = np.zeros(self.input_dim)
        elif u.size != self.input_dim:
            raise ValueError('u vector must have state dimension')

        f = self.A.dot(x) + self.B.dot(u) if (self.input_dim > 0) else self.A.dot(x)

        if not self.is_f_linear:
            for i in range(0, self.f_rbf_parameters['n_rbf']):
                center = self.f_rbf_parameters['centers'][i]
                width = self.f_rbf_parameters['width'][i]
                value = self.f_rbf_coeffs[i]
                f += rbf(value, center, inv(width), x)

        return f

    def compute_g(self, x, u=None):
        if x.size != self.state_dim:
            raise ValueError('x vector must have state dimension')
        elif u is None:
            u = np.zeros(self.input_dim)
        elif u.size != self.input_dim:
            raise ValueError('u vector must have state dimension')

        g = self.C.dot(x) + self.D.dot(u) if (self.input_dim > 0) else self.C.dot(x)

        if not self.is_g_linear:
            for i in range(0, self.g_rbf_parameters['n_rbf']):
                center = self.g_rbf_parameters['centers'][i]
                width = self.g_rbf_parameters['width'][i]
                value = self.g_rbf_coeffs[i]
                g += rbf(value, center, inv(width), x)

        return g

    def compute_df_dx(self, x):  # derivative of f ne depend pas de (u_t)_1..T
        if x.size != self.state_dim:
            raise ValueError('x vector must have state dimension')

        df = self.A

        if not self.is_f_linear:
            for i in range(0, self.f_rbf_parameters['n_rbf']):
                center = self.f_rbf_parameters['centers'][i]
                width = self.f_rbf_parameters['width'][i]
                value = self.f_rbf_coeffs[i]
                df += rbf_derivative(value, center, inv(width), x)

        return df

    def compute_dg_dx(self, x):   # derivative of g ne depend pas de (u_t)_1..T
        if x.size != self.state_dim:
            raise ValueError('x vector must have state dimension')

        dg = self.C

        if not self.is_g_linear:
            for i in range(0, self.g_rbf_parameters['n_rbf']):
                center = self.g_rbf_parameters['centers'][i]
                width = self.g_rbf_parameters['width'][i]
                value = self.g_rbf_coeffs[i]
                dg += rbf_derivative(value, center, inv(width), x)

        return dg

    def kalman_filtering(self, is_extended=False, input_sequence=None, output_sequence=None):
        """
            etant donne une sequence [y_1, ..., y_t], calcule de façon dynamique
            les moyennes et covariances de probabilités gaussiennes
            p(x_k|y_1, ... , y_k) et p(x_k|y_1, ... , y_k-1) pour k=1...t
            stocke les resultats dans self.filtered_state_means et self.filtered_state_covariance
        """

        if is_extended and (self.is_f_linear and self.is_g_linear):
            raise ValueError('Can not do extended Kalman filter with linear state space model')

        if output_sequence is None and self.output_sequence is None:
            raise ValueError('Can not do filtering if output_sequence is None')
        elif output_sequence is not None:
            self.output_sequence = output_sequence

        T = len(self.output_sequence)

        if (self.input_dim > 0) and (input_sequence is None) and (self.input_sequence is None):
            print 'WARNING: no input sequence, setting it to zero'
            self.input_sequence = [np.zeros(self.input_dim) for _ in range(0, T)]

        # state evolution equation
        A = self.A
        AT = np.transpose(A)
        B = self.B
        b = self.b
        Q = self.Q

        # output equation
        C = self.C
        CT = np.transpose(C)
        D = self.D
        d = self.d
        R = self.R

        self.filtered_state_means = np.zeros((T, 2, self.state_dim))
        self.filtered_state_covariance = np.zeros((T, 2, self.state_dim, self.state_dim))
        self.filtered_state_correlation = np.zeros((T-1, self.state_dim, self.state_dim)) # stock P_{t,t+1 | t} : correlation between states /!\ length = T-1
        for t in range(0, T):
            y = self.output_sequence[t]
            u = self.input_sequence[t] if (self.input_dim > 0) else None
            # pour le extended kalman filter, on change les valeurs de A, b, C, d a chaque étape
            if is_extended:
                # calcul des points de linéarisation x_tilde, u_f et u_g
                u_g = u
                if t == 0:
                    x_tilde = np.zeros(self.state_dim)
                    u_f = None
                else:
                    x_tilde = self.filtered_state_means[t-1, 1]
                    u_f = self.input_sequence[t-1] if (self.input_dim > 0) else None

                if not self.is_f_linear:
                    A = self.A + self.compute_df_dx(x_tilde)
                    AT = np.transpose(A)
                    b = self.b + self.compute_f(x_tilde, u_f)
                if not self.is_g_linear:
                    C = self.C + self.compute_dg_dx(x_tilde)
                    CT = np.transpose(C)
                    d = self.d + self.compute_g(x_tilde, u_g)

            if t == 0:
                #initialization
                x_1_0 = np.zeros(self.state_dim)
                P_1_0 = self.Sigma_0
            else:
                Bu = B.dot(self.input_sequence[t-1]) if (self.input_dim > 0) else np.zeros(self.state_dim)
                x_1_0 = A.dot(self.filtered_state_means[t-1, 1]) + Bu + b
                P_1_0 = A.dot(self.filtered_state_covariance[t-1, 1]).dot(AT) + Q
                P_t_comma_t_plus_1_t = self.filtered_state_covariance[t-1, 1].dot(AT)  # voir notation pdf section KF
                self.filtered_state_correlation[t-1] = P_t_comma_t_plus_1_t

            Du = D.dot(u) if (self.input_dim > 0) else np.zeros(self.output_dim)
            # kalman gain matrix
            K = P_1_0.dot(CT).dot(inv(C.dot(P_1_0).dot(CT) + R))
            x_1_1 = x_1_0 + K.dot(y - (C.dot(x_1_0) + Du + d))
            P_1_1 = P_1_0 - K.dot(C).dot(P_1_0)

            self.filtered_state_means[t, 0] = x_1_0
            self.filtered_state_means[t, 1] = x_1_1
            self.filtered_state_covariance[t, 0] = P_1_0
            self.filtered_state_covariance[t, 1] = P_1_1



    def kalman_smoothing(self, is_extended=False, output_sequence=None):
        """
            etant donne une sequence [y_1, ..., y_T], calcule de façon dynamique
            les moyennes et covariances de probabilités gaussiennes
            p(x_t|y_1, ... , y_T) pour t=1...T
            stocke les resultats dans self.smoothed_state_means et self.smoothed_state_covariance
        """
        self.kalman_filtering(is_extended=is_extended, output_sequence=output_sequence)

        T = len(self.output_sequence)

        self.smoothed_state_means = np.zeros((T, self.state_dim))
        self.smoothed_state_covariance = np.zeros((T, self.state_dim, self.state_dim))
        self.smoothed_state_correlation = np.zeros((T-1, self.state_dim, self.state_dim)) # stock P_{t,t+1 | T}  /!\ length = T avec None en dernière position
        AT = np.transpose(self.A)

        for t in range(T-1, -1, -1):
            if is_extended:
                x_dot = self.filtered_state_means[t, 1]  # On linéarise autour de la moyenne renvoyé par Kalman Filter
                A = self.A + self.compute_df_dx(x_dot)
                AT = np.transpose(A)

            if t == T-1:  # initialisation en backward
                x_t_T = self.filtered_state_means[t, 1]
                P_t_T = self.filtered_state_covariance[t, 1]
            else:
                P_t_t = self.filtered_state_covariance[t, 1]
                P_t_plus_1_t = self.filtered_state_covariance[t+1, 0]
                P_t_plus_1_T = self.smoothed_state_covariance[t+1]
                x_t_t = self.filtered_state_means[t, 1]
                x_t_plus_1_t = self.filtered_state_means[t+1, 0]
                x_t_plus_1_T = self.smoothed_state_means[t+1]

                L = P_t_t.dot(AT).dot(inv(P_t_plus_1_t))
                LT = np.transpose(L)

                x_t_T = x_t_t + L.dot(x_t_plus_1_T - x_t_plus_1_t)
                P_t_T = P_t_t + L.dot(P_t_plus_1_T - P_t_plus_1_t).dot(LT)

                P_t_comma_t_plus_1_T = self.filtered_state_correlation[t] - (x_t_t - x_t_T)[:, np.newaxis].dot((x_t_plus_1_t - x_t_plus_1_T)[np.newaxis, :])
                self.smoothed_state_correlation[t] = P_t_comma_t_plus_1_T

            self.smoothed_state_means[t] = x_t_T
            self.smoothed_state_covariance[t] = P_t_T



    def draw_sample(self, T=1, input_sequence=None):
        if (self.input_dim > 0 ) and (input_sequence is None) and (self.input_sequence is None or len(self.input_sequence) < T):
            print 'No input sequence given, setting inputs to zero'
            self.input_sequence = [np.zeros(self.input_dim) for _ in range(0,T)]

        states = np.zeros((T, self.state_dim))
        outputs = np.zeros((T, self.output_dim))

        x = np.zeros(self.state_dim)
        x_1 = mv_norm(x, self.Sigma_0)
        y = self.compute_g(x_1, self.input_sequence[0]) if (self.input_dim > 0) else self.compute_g(x_1)
        y_1 = mv_norm(y, self.R)

        states[0] = x_1
        outputs[0] = y_1

        for t in range(1, T):
            x = self.compute_f(states[t-1], self.input_sequence[t-1])  if (self.input_dim > 0) else self.compute_f(states[t-1])
            x_t = mv_norm(x, self.Q)
            y = self.compute_g(x_t, self.input_sequence[t])  if (self.input_dim > 0) else self.compute_g(x_t)
            y_t = mv_norm(y, self.R)

            states[t] = x_t
            outputs[t] = y_t

        self.output_sequence = outputs
        self.state_sequence = states

    def compute_f_optimal_parameters(self):
        T = len(self.output_sequence)
        I = self.f_rbf_parameters['n_rbf'] if (not self.is_f_linear) else 0
        p = self.state_dim
        q = self.input_dim
        n_params = I+p+q+1

        if (n_params > T-1):
            raise Exception('More paramerers (' + str(n_params) + ') than values (' + str(T-1) +')')

        xPhiT = np.zeros((p, I+p+q+1))
        PhiPhiT = np.zeros((I+p+q+1, I+p+q+1))

        for t in range(0,T):
            PInv = inv(self.smoothed_state_covariance[t])
            x = self.smoothed_state_means[t]
            u = self.input_sequence[t] if (q > 0) else None

            # expectations involving only x
            PhiPhiT[I:I+p, I:I+p] = x[:, np.newaxis].dot(x[np.newaxis, :]) + self.smoothed_state_covariance[t]

            if (q > 0):
                xuT = x[:, np.newaxis].dot(u[np.newaxis, :])
                PhiPhiT[I:I+p, I+p:I+p+q] = xuT
                PhiPhiT[I+p:I+p+q, I:I+p] = xuT.transpose()

                PhiPhiT[I+p:I+p+q, I+p:I+p+q] = u[:, np.newaxis].dot(u[np.newaxis, :])

                PhiPhiT[I+p:I+p+q, I+p+q] = u
                PhiPhiT[I+p+q, I+p:I+p+q] = u

            PhiPhiT[I:I+p, I+p+q] = x
            PhiPhiT[I+p+q, I:I+p] = x


            PhiPhiT[I+p+q, I+p+q] = 1

            if (t < T-1):
                xPhiT[:,I:I+p] = self.smoothed_state_means[t+1][:, np.newaxis].dot(self.smoothed_state_means[t][np.newaxis, :]) + self.smoothed_state_correlation[t]
                xPhiT[:,I+p+q] = self.smoothed_state_means[t+1]
                if (q > 0):
                    xPhiT[:,I+p:I+p+q] = self.smoothed_state_means[t+1][:, np.newaxis].dot(u[np.newaxis, :])

            # expectations involving RBF
            for i in range(0,I):
                SInv = inv(self.f_rbf_parameters['width'][i])
                c = self.f_rbf_parameters['centers'][i]

                Sigma = inv(PInv + SInv)
                mu = Sigma.dot(PInv.dot(x) + SInv.dot(c))
                delta = c.dot(SInv).dot(c) + x.dot(PInv).dot(x) - mu.dot(Sigma).dot(mu)
                beta = power(det(Sigma) * det(SInv) * det(PInv), 0.5) * exp(-0.5 * delta) / power(2 * np.pi, 0.5 * p)

                PhiPhiT[I: I+p, i] += beta * mu
                PhiPhiT[i, I: I+p] += beta * mu

                if (q > 0):
                    PhiPhiT[I+p: I+p+q, i] += beta * u
                    PhiPhiT[i, I+p: I+p+q] += beta * u

                PhiPhiT[I+p+q, i] += beta
                PhiPhiT[i, I+p+q] += beta

                # expectations with mu^{i,j}_t and beta^{i,j}_t
                for j in range(i,I):
                    SjInv = inv(self.f_rbf_parameters['width'][j])
                    cj = self.f_rbf_parameters['centers'][j]

                    Sigma = inv(PInv + SInv + SjInv)
                    mu = Sigma.dot(PInv.dot(x) + SInv.dot(c) + SjInv.dot(cj))
                    delta = c.dot(SInv).dot(c) + cj.dot(SjInv).dot(c) + x.dot(PInv).dot(x) - mu.dot(Sigma).dot(mu)
                    beta = power(det(Sigma) * det(SInv) * det(SjInv) * det(PInv), 0.5) * exp(-0.5 * delta) / power(2 * np.pi, p)

                    PhiPhiT[i, j] += beta
                    PhiPhiT[j, i] += beta

                # expectations with mu^i_{t,t+1} and beta^i_{t,t+1}
                if (t < T-1):
                    P2Inv = np.zeros((2*p, 2*p))
                    P2Inv[0:p,0:p] = self.smoothed_state_covariance[t]
                    P2Inv[p:2*p,p:2*p] = self.smoothed_state_covariance[t+1]
                    P2Inv[0:p,p:2*p] = self.smoothed_state_correlation[t]
                    P2Inv[p:2*p,0:p] = self.smoothed_state_correlation[t]
                    S2Inv = np.zeros((2*p, 2*p))
                    S2Inv[0:p,0:p] = SInv

                    x2 = np.concatenate((self.smoothed_state_means[t], self.smoothed_state_means[t+1]))
                    Sigma = inv(P2Inv + S2Inv)
                    mu = Sigma.dot(P2Inv.dot(x2) + np.concatenate((SInv.dot(c), np.zeros(p))))
                    delta = c.dot(SInv).dot(c) + x.dot(PInv).dot(x) - mu.dot(Sigma).dot(mu)
                    beta = np.sqrt(det(Sigma) * det(SInv) * det(PInv) / power(2 * np.pi, p)) * exp(-0.5 * delta)

                    xPhiT[:, i] +=  beta * mu[p:2*p]

        theta_f = xPhiT.dot(inv(PhiPhiT))

        return theta_f

