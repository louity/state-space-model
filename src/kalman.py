# coding: utf8
import numpy as np
import matplotlib.pyplot as plt
from numpy import power, exp, sqrt, log
from numpy.linalg import inv, det
from numpy.random import multivariate_normal as mv_norm
import random
import math
import utils

DEFAULT_N_RBF = 10

class StateSpaceModel:
    """
    classe decrivant un modele suivant les equations :
        x_t+1 = sum_i(rho_i(x) h_i) + A x_t + B u_t + b + w_t, w_t zero mean with cov mat Q,rho_i RBF function
        y_t = sum_j(rho_j(x) k_j) + C x_t + D u_t + d + v_t, v_t zero mean with cov mat R
    permet de faire du filtering et du smoothing
    """

    def __init__(self, is_f_linear=True, is_g_linear=True, state_dim=None, input_dim=None, output_dim=None, Sigma_0=None, A=None, B=None, b=None, Q=None, C=None, D=None, d=None, R=None, f_rbf_parameters=None, f_rbf_coeffs=None, g_rbf_parameters=None, g_rbf_coeffs=None, f_analytical=None):
        '''
        Cette fonction donne les attributs a l'objet self et verifie leur coherence
        '''
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

        self.b = utils.check_vector(b, self.state_dim, 'vector b size must be equal to state_dim')
        self.d = utils.check_vector(d, self.output_dim, 'vector d size must de equal to output_dim')

        self.Sigma_0 = utils.check_matrix(Sigma_0, (self.state_dim, self.state_dim), 'matrix Sigma_0 shape must be equal to self.state_dim')
        self.A = utils.check_matrix(A, (self.state_dim, self.state_dim), 'matrix A shape must equal to state_dim x state_dim')
        self.B = utils.check_matrix(B, (self.state_dim, self.input_dim), 'matrix B shape must equal to state_dim x input_dim') if (self.input_dim > 0) else None
        self.Q = utils.check_matrix(Q, (self.state_dim, self.state_dim), 'matrix Q shape must equal to state_dim x state_dim')
        self.C = utils.check_matrix(C, (self.output_dim, self.state_dim), 'matrix C shape must equal to output_dim x state_dim')
        self.D = utils.check_matrix(D, (self.output_dim, self.input_dim), 'matrix D shape must equal to output_dim x input_dim') if (self.input_dim > 0) else None
        self.R = utils.check_matrix(R, (self.output_dim, self.output_dim), 'matrix R shape must equal to self.output_dim')

        self.f_rbf_parameters = f_rbf_parameters
        self.g_rbf_parameters = g_rbf_parameters
        self.f_rbf_coeffs = f_rbf_coeffs
        self.g_rbf_coeffs = g_rbf_coeffs
        
        self.f_analytical = f_analytical

        self.output_sequence = None
        self.state_sequence = None
        self.input_sequence = None

        if not self.is_f_linear and self.f_rbf_parameters is None:
            print 'No rbf parameters provided for f, initialize them'
            self.initialize_f_rbf_parameters()
        if not self.is_f_linear and self.f_rbf_coeffs is None:
            self.f_rbf_coeffs = np.zeros((self.f_rbf_parameters['n_rbf'], self.state_dim))
        if not self.is_g_linear and self.g_rbf_parameters is None:
            print 'No rbf parameters provided for g, initialize them '
            self.initialize_g_rbf_parameters()
        if not self.is_g_linear and self.g_rbf_coeffs is None:
            self.g_rbf_coeffs = np.zeros((self.g_rbf_parameters['n_rbf'], self.output_dim))

    def get_rbf_parameters_for_state(self):
        '''
        give default parameters to the RBF on the state space
        '''
        is_f_linear = self.is_f_linear
        is_g_linear = self.is_g_linear

        # make f and g linear
        self.is_f_linear = True
        self.is_g_linear = True

        self.draw_sample(10 * DEFAULT_N_RBF)
        self.kalman_smoothing()

        self.is_f_linear = is_f_linear
        self.is_g_linear = is_g_linear

        # if state dim is one, place rbf uniformly
        if (self.state_dim == 1):
            x_min = np.min(self.smoothed_state_means)
            x_max = np.max(self.smoothed_state_means)
            rbf_centers = np.zeros((DEFAULT_N_RBF, 1))
            rbf_width = np.zeros((DEFAULT_N_RBF, 1, 1))
            center_space = (x_max - x_min) / DEFAULT_N_RBF

            for i in range(0, DEFAULT_N_RBF):
                rbf_centers[i, 0] = x_min + (i + 0.5) * center_space
                rbf_width[i, 0, 0] = center_space**2

            return {
                'n_rbf': DEFAULT_N_RBF,
                'centers': rbf_centers,
                'width': rbf_width
            }
        else:
            return {
                'n_rbf': DEFAULT_N_RBF,
                'centers': random.sample(self.smoothed_state_means, DEFAULT_N_RBF),#TODO : replace random selection by k-means
                'width': random.sample(self.smoothed_state_covariance, DEFAULT_N_RBF)
            }

    def initialize_f_rbf_parameters(self):
        '''
        Idem on ne devrait pas l'utiliser puisque l'on ne veut pas adapter les parametres de la RBF
        '''
        if self.g_rbf_parameters is not None:
            self.f_rbf_parameters = self.g_rbf_parameters
        else:
            self.f_rbf_parameters = self.get_rbf_parameters_for_state()

    def initialize_g_rbf_parameters(self):
        '''
        Idem on ne devrait pas l'utiliser puisque l'on ne veut pas adapter les parametres de la RBF
        '''
        if self.f_rbf_parameters is not None:
            self.g_rbf_parameters = self.f_rbf_parameters
        else:
            self.g_rbf_parameters = self.get_rbf_parameters_for_state()

    def compute_f(self, x, u=None):
        if x.size != self.state_dim:
            raise ValueError('x vector must have state dimension')
        if (u is not None and u.size != self.input_dim):
            raise ValueError('u vector must have state dimension')
            
        if (self.f_analytical is not None and u is None):
            return self.f_analytical(x)

        f = self.A.dot(x) + self.b

        if (u is not None):
             f += self.B.dot(u)

        if not self.is_f_linear:
            I = self.f_rbf_parameters['n_rbf']
            for i in range(0, I):
                center = self.f_rbf_parameters['centers'][i]
                width = self.f_rbf_parameters['width'][i]
                value = self.f_rbf_coeffs[i]
                f += utils.rbf(value, center, inv(width), x)

        return f

    def compute_g(self, x, u=None):
        if x.size != self.state_dim:
            raise ValueError('x vector must have state dimension')
        elif (u is not None and u.size != self.input_dim):
            raise ValueError('u vector must have state dimension')

        g = self.C.dot(x) + self.d

        if (u is not None):
            g += self.D.dot(u)

        if not self.is_g_linear:
            J = self.g_rbf_parameters['n_rbf']
            for j in range(0, J):
                center = self.g_rbf_parameters['centers'][j]
                width = self.g_rbf_parameters['width'][j]
                value = self.g_rbf_coeffs[j]
                g += utils.rbf(value, center, inv(width), x)

        return g

    def compute_df_dx(self, x):  # derivative of f ne depend pas de (u_t)_1..T
        if x.size != self.state_dim:
            raise ValueError('x vector must have state dimension')
        p = self.state_dim
        df = np.zeros((p, p))
        df += self.A

        if not self.is_f_linear:
            for i in range(0, self.f_rbf_parameters['n_rbf']):
                center = self.f_rbf_parameters['centers'][i]
                width = self.f_rbf_parameters['width'][i]
                value = self.f_rbf_coeffs[i]
                df += utils.rbf_derivative(value, center, inv(width), x)

        return df

    def compute_dg_dx(self, x):   # derivative of g ne depend pas de (u_t)_1..T
        if x.size != self.state_dim:
            raise ValueError('x vector must have state dimension')
        p = self.state_dim
        n = self.output_dim

        dg = np.zeros((n, p))
        dg += self.C

        if not self.is_g_linear:
            for i in range(0, self.g_rbf_parameters['n_rbf']):
                center = self.g_rbf_parameters['centers'][i]
                width = self.g_rbf_parameters['width'][i]
                value = self.g_rbf_coeffs[i]
                dg += utils.rbf_derivative(value, center, inv(width), x)

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
            if (self.input_dim > 0):
                u_g = self.input_sequence[t]
                d = self.d + D.dot(u_g)

                if t > 0:
                    u_f = self.input_sequence[t-1]
                    b = self.b + B.dot(u_f)
            else:
                u_f = None
                u_g = None

            if is_extended: # calcul du points de linéarisation x_tilde et modification de A, C et b, d
                if t == 0:
                    x_tilde = np.zeros(self.state_dim)
                else:
                    x_tilde = self.filtered_state_means[t-1, 1]
                if not self.is_f_linear:
                    A = self.compute_df_dx(x_tilde)
                    AT = np.transpose(A)
                    b = self.compute_f(x_tilde, u_f) - A.dot(x_tilde)
                if not self.is_g_linear:
                    C = self.compute_dg_dx(x_tilde)
                    CT = np.transpose(C)
                    d = self.compute_g(x_tilde, u_g) - C.dot(x_tilde)

            # calcul des moyennes et covariances
            if t == 0: # initialization
                x_1_0 = np.zeros(self.state_dim)
                P_1_0 = self.Sigma_0
            else:
                x_1_0 = A.dot(self.filtered_state_means[t-1, 1]) + b
                P_1_0 = A.dot(self.filtered_state_covariance[t-1, 1]).dot(AT) + Q
                P_t_comma_t_plus_1_t = self.filtered_state_covariance[t-1, 1].dot(AT)  # voir notation pdf section KF
                self.filtered_state_correlation[t-1] = P_t_comma_t_plus_1_t

            # kalman gain matrix
            K = P_1_0.dot(CT).dot(inv(C.dot(P_1_0).dot(CT) + R))
            x_1_1 = x_1_0 + K.dot(y - (C.dot(x_1_0) + d))
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
                x_dot = self.filtered_state_means[t, 1]  # On linéarise autour de la moyenne renvoyée par le Kalman Filter
                AT = np.transpose(self.compute_df_dx(x_dot))

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
        '''
        genere une output_sequence et une state_sequence de tqille T avec les vrais
        le for n'est clairement pas optimal
        Comment fait-on si on veut un x_true et un x_learn: oblige de creer 2 objets?
        '''
        if (self.input_dim > 0 ) and (input_sequence is None) and (self.input_sequence is None or len(self.input_sequence) < T):
            print 'No input sequence given, setting inputs to zero'
            self.input_sequence = [np.zeros(self.input_dim) for _ in range(0, T)]

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
        
    
    def compute_f_optimal_parameters(self, use_smoothed_values=False):
        T = len(self.output_sequence)
        I = self.f_rbf_parameters['n_rbf'] if (not self.is_f_linear) else 0
        p = self.state_dim
        q = self.input_dim
        n_params = I+p+q+1

        if (n_params > T-1):
            raise Exception('More paramerers (' + str(n_params) + ') than values (' + str(T-1) +')')

        xPhiT = np.zeros((p, I+p+q+1))
        PhiPhiT = np.zeros((I+p+q+1, I+p+q+1))
        xxT = np.zeros((p, p))

        for t in range(0, T):
            # simplify notations
            u = self.input_sequence[t] if (q > 0) else None

            if (use_smoothed_values):
                P = self.smoothed_state_covariance[t]
                PInv = inv(P)
                x = self.smoothed_state_means[t]
                if (t < T-1):
                    x_plus = self.smoothed_state_means[t+1]
                    P_plus = self.smoothed_state_covariance[t+1]
                    PCor = self.smoothed_state_correlation[t]
            else:
                P = self.filtered_state_covariance[t, 1]
                PInv = inv(P)
                x = self.filtered_state_means[t, 1]
                if (t < T-1):
                    x_plus = self.filtered_state_means[t+1, 1]
                    P_plus = self.filtered_state_covariance[t+1, 1]
                    PCor = self.filtered_state_correlation[t]

            # expectations involving only x
            # PhiPhiT
            PhiPhiT[I:I+p, I:I+p] += x[:, np.newaxis].dot(x[np.newaxis, :]) + P
            PhiPhiT[I:I+p, I+p+q] += x
            PhiPhiT[I+p+q, I:I+p] += x
            PhiPhiT[I+p+q, I+p+q] += 1

            # if input space dimension > 1
            if (q > 0):
                xuT = x[:, np.newaxis].dot(u[np.newaxis, :])
                uuT = u[:, np.newaxis].dot(u[np.newaxis, :])

                PhiPhiT[I:I+p, I+p:I+p+q] += xuT
                PhiPhiT[I+p:I+p+q, I:I+p] += xuT.transpose()
                PhiPhiT[I+p:I+p+q, I+p:I+p+q] += uuT
                PhiPhiT[I+p:I+p+q, I+p+q] += u
                PhiPhiT[I+p+q, I+p:I+p+q] += u

            # xPhiT and xxT
            if (t < T-1):
                xPhiT[:,I:I+p] += x_plus[:, np.newaxis].dot(x[np.newaxis, :]) + PCor.transpose()
                xPhiT[:,I+p+q] += x_plus
                xxT += x_plus[:, np.newaxis].dot(x_plus[np.newaxis, :]) + P_plus
                if (q > 0):
                    xPhiT[:,I+p:I+p+q] += x_plus[:, np.newaxis].dot(u[np.newaxis, :])

            # expectations involving RBF
            for i in range(0, I):
                # simplify notatations
                SInv = inv(self.f_rbf_parameters['width'][i])
                c = self.f_rbf_parameters['centers'][i]

                SigmaInv = PInv + SInv
                Sigma = inv(SigmaInv)
                mu = Sigma.dot(PInv.dot(x) + SInv.dot(c))
                delta = c.dot(SInv).dot(c) + x.dot(PInv).dot(x) - mu.dot(SigmaInv).dot(mu)
                beta = power(2 * np.pi, -0.5 * p) * sqrt(det(Sigma) * det(SInv) * det(PInv)) * exp(-0.5 * delta)

                PhiPhiT[I: I+p, i] += beta * mu
                PhiPhiT[i, I: I+p] += beta * mu
                PhiPhiT[I+p+q, i] += beta
                PhiPhiT[i, I+p+q] += beta

                if (q > 0):
                    PhiPhiT[I+p: I+p+q, i] += beta * u
                    PhiPhiT[i, I+p: I+p+q] += beta * u

                # expectations with mu^{i,j}_t and beta^{i,j}_t
                for j in range(i, I):
                    SjInv = inv(self.f_rbf_parameters['width'][j])
                    cj = self.f_rbf_parameters['centers'][j]

                    SigmaInv = PInv + SInv + SjInv
                    Sigma = inv(SigmaInv)
                    mu = Sigma.dot(PInv.dot(x) + SInv.dot(c) + SjInv.dot(cj))
                    delta = c.dot(SInv).dot(c) + cj.dot(SjInv).dot(cj) + x.dot(PInv).dot(x) - mu.dot(SigmaInv).dot(mu)
                    beta = power(2 * np.pi, -p) * sqrt(det(Sigma) * det(SInv) * det(SjInv) * det(PInv)) * exp(-0.5 * delta)

                    if (i == j):
                        PhiPhiT[i, i] += beta
                    else:
                        PhiPhiT[i, j] += beta
                        PhiPhiT[j, i] += beta

                # expectations with mu^i_{t,t+1} and beta^i_{t,t+1}
                if (t < T-1):
                    P2 = np.zeros((2*p, 2*p))
                    P2[0:p,0:p] = P
                    P2[p:2*p,p:2*p] = P_plus
                    P2[0:p,p:2*p] = PCor
                    P2[p:2*p,0:p] = PCor.transpose()
                    P2Inv = inv(P2)

                    S2Inv = np.zeros((2*p, 2*p))
                    S2Inv[0:p,0:p] = SInv

                    x2 = np.concatenate((x, x_plus))
                    SigmaInv = P2Inv + S2Inv
                    Sigma = inv(SigmaInv)
                    mu = Sigma.dot(P2Inv.dot(x2) + np.concatenate((SInv.dot(c), np.zeros(p))))
                    delta = c.dot(SInv).dot(c) + x2.dot(P2Inv).dot(x2) - mu.dot(SigmaInv).dot(mu)
                    beta = power(2 * np.pi, -0.5 * p) * sqrt(det(Sigma) * det(SInv) * det(P2Inv)) * exp(-0.5 * delta)

                    xPhiT[:, i] +=  beta * mu[p:2*p]

        theta_f = xPhiT.dot(inv(PhiPhiT))
        self.Q = 1.0 / (T - 1) * (xxT - theta_f.dot(xPhiT.transpose()))

        self.A = theta_f[:, I:I+p]
        self.b = theta_f[:, I+p+q]
        for i in range(0, I):
            self.f_rbf_coeffs[i] = theta_f[:, i]
        if (q > 0):
            self.B = theta_f[:, I+p:I+p+q]

    def compute_g_optimal_parameters(self, use_smoothed_values=False):
        T = len(self.output_sequence)
        J = self.g_rbf_parameters['n_rbf'] if (not self.is_g_linear) else 0
        p = self.state_dim
        q = self.input_dim
        n = self.output_dim
        n_params = J+p+q+1

        if (n_params > T-1):
            raise Exception('More paramerers (' + str(n_params) + ') than values (' + str(T-1) +')')

        yPhiT = np.zeros((n, J+p+q+1))
        PhiPhiT = np.zeros((J+p+q+1, J+p+q+1))
        yyT = np.zeros((n, n))

        for t in range(0, T):
            # simplify notations
            u = self.input_sequence[t] if (q > 0) else None
            y = self.output_sequence[t]

            if (use_smoothed_values):
                P = self.smoothed_state_covariance[t]
                PInv = inv(P)
                x = self.smoothed_state_means[t]
            else:
                P = self.filtered_state_covariance[t, 1]
                PInv = inv(P)
                x = self.filtered_state_means[t, 1]

            # expectations involving only x
            # PhiPhiT
            PhiPhiT[J:J+p, J:J+p] += x[:, np.newaxis].dot(x[np.newaxis, :]) + P
            PhiPhiT[J:J+p, J+p+q] += x
            PhiPhiT[J+p+q, J:J+p] += x
            PhiPhiT[J+p+q, J+p+q] += 1

            # if input space dimension > 1
            if (q > 0):
                xuT = x[:, np.newaxis].dot(u[np.newaxis, :])
                uuT = u[:, np.newaxis].dot(u[np.newaxis, :])

                PhiPhiT[J:J+p, J+p:J+p+q] += xuT
                PhiPhiT[J+p:J+p+q, J:J+p] += xuT.transpose()
                PhiPhiT[J+p:J+p+q, J+p:J+p+q] += uuT
                PhiPhiT[J+p:J+p+q, J+p+q] += u
                PhiPhiT[J+p+q, J+p:J+p+q] += u

            # yPhiT and yyT
            yPhiT[:,J:J+p] += y[:, np.newaxis].dot(x[np.newaxis, :])
            yPhiT[:,J+p+q] += y
            yyT += y[:, np.newaxis].dot(y[np.newaxis, :])
            if (q > 0):
                yPhiT[:,J+p:J+p+q] += y[:, np.newaxis].dot(u[np.newaxis, :])

            # expectations involving RBF
            for j in range(0, J):
                # simplify notatations
                SInv = inv(self.g_rbf_parameters['width'][j])
                c = self.g_rbf_parameters['centers'][j]

                SigmaInv = PInv + SInv
                Sigma = inv(SigmaInv)
                mu = Sigma.dot(PInv.dot(x) + SInv.dot(c))
                delta = c.dot(SInv).dot(c) + x.dot(PInv).dot(x) - mu.dot(SigmaInv).dot(mu)
                beta = power(2 * np.pi, -0.5 * p) * sqrt(det(Sigma) * det(SInv) * det(PInv)) * exp(-0.5 * delta)

                PhiPhiT[J: J+p, j] += beta * mu
                PhiPhiT[j, J: J+p] += beta * mu
                PhiPhiT[J+p+q, j] += beta
                PhiPhiT[j, J+p+q] += beta

                if (q > 0):
                    PhiPhiT[J+p: J+p+q, j] += beta * u
                    PhiPhiT[j, J+p: J+p+q] += beta * u

                yPhiT[:, j] +=  beta * y

                # expectations with mu^{i,j}_t and beta^{i,j}_t
                for k in range(j, J):
                    SkInv = inv(self.g_rbf_parameters['width'][k])
                    ck = self.g_rbf_parameters['centers'][k]

                    SigmaInv = PInv + SInv + SkInv
                    Sigma = inv(SigmaInv)
                    mu = Sigma.dot(PInv.dot(x) + SInv.dot(c) + SkInv.dot(ck))
                    delta = c.dot(SInv).dot(c) + ck.dot(SkInv).dot(ck) + x.dot(PInv).dot(x) - mu.dot(SigmaInv).dot(mu)
                    beta = power(2 * np.pi, -p) * sqrt(det(Sigma) * det(SInv) * det(SkInv) * det(PInv)) * exp(-0.5 * delta)

                    if (j == k):
                        PhiPhiT[j, j] += beta
                    else:
                        PhiPhiT[j, k] += beta
                        PhiPhiT[k, j] += beta

        theta_g = yPhiT.dot(inv(PhiPhiT))
        self.R = 1.0 / T * (yyT - theta_g.dot(yPhiT.transpose()))

        self.C = theta_g[:, J:J+p]
        self.d = theta_g[:, J+p+q]
        for j in range(0, J):
            self.g_rbf_coeffs[j] = theta_g[:,j]
        if (q > 0):
            self.D = theta_g[:,J+p:J+p+q]



    def E_step_factor_Analysis(self):
        '''
        Ici on fait une etape du E-step a la "Nbr_iteartion" iteration.
        '''
        T = len(self.output_sequence)
        p = self.state_dim
        C = self.C
        CT = C.transpose()
        RInv = inv(self.R)
        p = self.state_dim

        E_x = np.zeros((T, p))
        E_xxT = np.zeros((T, p, p))

        for t in range(0, T):
            y_t = self.output_sequence[t]
            sigma_x_t = inv(np.identity(p) + CT.dot(RInv).dot(C))

            E_x[t] = sigma_x_t.dot(CT).dot(RInv).dot(y_t - self.d)
            E_xxT[t] = sigma_x_t + E_x[t][:, np.newaxis].dot(E_x[t][np.newaxis, :])

        return (E_x, E_xxT)

    def M_step_factor_Analysis(self, E_x, E_xxT):
        '''
        Ici on fait une etape du M-step a la "Nbr_iteartion" iteration.
        '''
        #on recupere la taille des donnees
        T = len(self.output_sequence)
        p = self.state_dim
        n = self.output_dim

        #variables intermediaires
        yxT = np.zeros((n, p))
        yyT = np.zeros((n, n))
        xxT = np.sum(E_xxT, axis=0)

        for t in range(0, T):
            y_t = self.output_sequence[t][:, np.newaxis]
            x_t = E_x[t][:, np.newaxis]

            yxT = yxT + y_t.dot(x_t.transpose())
            yyT = yyT + y_t.dot(y_t.transpose())


        xyT = yxT.transpose()
        #c'est le resultat du M-step
        self.C = yxT.dot(inv(xxT))
        self.R = np.diag(np.diag(yyT - self.C.dot(xyT)) / T)

    def compute_expected_complete_log_likelihood_factor_analysis(self):
        T = len(self.output_sequence)
        R = self.R
        C = self.C

        compute_expected_complete_likelihood = T/2*log(det(R))
        for t in range(T):
            x_t = self.estimated_state_sequence_with_FA[t]
            y_t = self.output_sequence[t]
            compute_expected_complete_likelihood += -0.5 * np.trace(inv(R).dot((y_t - C.dot(x_t)).dot((y_t-C.dot(x_t)).transpose()) ))
        return(compute_expected_complete_likelihood)

    def initialize_f_with_factor_analysis(self, n_EM_iterations):
        '''
        '''
        T = len(self.output_sequence)
        p = self.state_dim

        self.d = np.mean(self.output_sequence, axis=0)

        likelihood_evolution = np.zeros(n_EM_iterations)

        for i in range(n_EM_iterations):
            (E_x, E_xxT) = self.E_step_factor_Analysis()
            self.M_step_factor_Analysis(E_x, E_xxT)

            self.estimated_state_sequence_with_FA = E_x
            likelihood_evolution[i] = self.compute_expected_complete_log_likelihood_factor_analysis()


        return likelihood_evolution


    def learn_f_and_g_with_EM_algorithm(self, use_smoothed_values=None):
        n_EM_iterations = 20

        if (self.is_f_linear and self.is_g_linear):
            if (use_smoothed_values is None):
                use_smoothed_values = False
            is_extended=False
        elif (not self.is_f_linear and self.is_g_linear):
            if (use_smoothed_values is None):
                    use_smoothed_values = False
            is_extended=True
            #self.initialize_g_with_factor_analysis()
        elif (self.is_f_linear and not self.is_g_linear):
            raise Exception('EM not implemented for f linear and g non-linear')
        elif (not self.is_f_linear and not self.is_g_linear):
            raise Exception('EM not implemented for f non-linear and g non-linear')

        log_likelihood = np.zeros(n_EM_iterations)

        for EM_iteration in range(0, n_EM_iterations):
            # E-Step
            self.kalman_smoothing(is_extended=False)
            #M-Step
            self.compute_f_optimal_parameters(use_smoothed_values=use_smoothed_values)
            #self.compute_g_optimal_parameters(use_smoothed_values=use_smoothed_values)
            log_likelihood[EM_iteration] = self.compute_outputs_log_likelihood(use_smoothed_values=use_smoothed_values)

        return log_likelihood

    def plot_states_in_1D(self):
        if (self.state_dim != 1):
            raise Exception('state plot can be sonly in 1D')
        plt.clf()
        plt.figure(1)
        min = 0
        max = 1

        if (self.state_sequence is not None):
            states = self.state_sequence
            T = len(states)
            plt.scatter(states[0:T-1], states[1:])
            min = np.min(states)
            max = np.max(states)

        def f(x):
            return self.compute_f(np.array([x]))[0]

        X = np.linspace(min, max, 100)
        plt.plot(X, np.vectorize(f)(X),'r-')
        plt.plot(X, X, 'r--')
        if self.is_f_linear:
            plt.title('True states evolution')
        else:
            plt.title('true states evolution. f : x -> ' + str(self.A[0, 0]) + ' * x + ' + str(self.b[0]))
        plt.show()

    def plot_estimmated_states_in_1D(self, use_smoothed_values=False):
        if (self.state_dim != 1):
            raise Exception('state plot can be sonly in 1D')
        plt.clf()
        plt.figure(1)
        min = 0
        max = 1

        if (use_smoothed_values):
            states = self.smoothed_state_means
        else:
            states = self.filtered_state_means[:, 1]

        T = len(states)
        plt.scatter(states[0:T-1], states[1:])
        min = np.min(states)
        max = np.max(states)

        def f(x):
            return self.compute_f(np.array([x]))[0]

        X = np.linspace(min, max, 100)
        plt.plot(X, np.vectorize(f)(X), 'r-')
        plt.plot(X, X, 'r--')
        #ce n'était bon que pour la partir lineaire
        if self.is_f_linear:
            plt.title('inferred states evolution. f : x -> ' + str(self.A[0, 0]) + ' * x + ' + str(self.b[0]))
        else:
            plt.title('Inferred states evolution')
            
        plt.legend(['true function'])
        plt.show()

    def plot_states_outputs_in_1D(self):
        if (self.state_dim != 1 or self.output_dim != 1):
            raise Exception('state-output plots can be only in 1D')
        plt.clf()
        plt.figure(1)
        min = 0
        max = 1

        if (self.state_sequence is not None):
            states = self.state_sequence
            outputs = self.output_sequence
            T = len(states)
            plt.scatter(states, outputs)
            min = np.min(states)
            max = np.max(states)

        def g(x):
            return self.compute_g(np.array([x]))[0]

        X = np.linspace(min, max, 100)
        plt.plot(X, np.vectorize(g)(X),'r-')
        plt.plot(X, X, 'r--')
        plt.title('state-output. g : x -> ' + str(self.C[0, 0]) + ' * x + ' + str(self.d[0]))
        plt.show()

    def compute_outputs_log_likelihood(self, use_smoothed_values=False):
        outputs = self.output_sequence
        T = len(outputs)
        n = self.output_dim
        log_likelihood = 0
        R = self.R
        C = self.C
        CT = C.transpose()

        for t in range(0, T):
            y = outputs[t]

            if (use_smoothed_values):
                mu_x = self.smoothed_state_means[t]
                Sigma_x = self.smoothed_state_means[t]
            else:
                mu_x = self.filtered_state_means[t, 1]
                Sigma_x = self.filtered_state_covariance[t, 1]

            if (not self.is_g_linear):
                C = C + self.compute_dg_dx(mu_x)

            mu_y = self.compute_g(mu_x)
            Sigma_y = C.dot(Sigma_x).dot(CT) + R

            log_likelihood += -0.5 * n * log(2 * np.pi) - 0.5 * log(det(Sigma_y)) - 0.5 * (y - mu_y).dot(inv(Sigma_y)).dot(y - mu_y)

        return log_likelihood

