# coding: utf8

import unittest
import numpy as np
from state_space_model import StateSpaceModel
from numpy.random import rand, random
from numpy.linalg import inv
import matplotlib.pyplot as plt

import utils

class TestStateSpaceModelConstructorAndSample(unittest.TestCase):
    """
        Tests constructor and draw_sample method of class StateSpaceModel
    """

    def test_constructor_and_sample_method(self):
        input_dim = 0
        for (state_dim, output_dim) in  zip([1, 4, 2], [1, 2, 4]):
            state_dim = 1
            output_dim = 1

            A = random((state_dim, state_dim))
            C = random((output_dim, state_dim))
            Q = np.ones((state_dim, state_dim))
            R = np.ones((output_dim, output_dim))

            def f(x):
                return A.dot(x)
            def g(x):
                return C.dot(x)

            ssm = StateSpaceModel(
                input_dim=input_dim,
                output_dim=output_dim,
                state_dim=state_dim,
                Q=Q,
                R=R,
                f=f,
                g=g
            )

            n_sample = 50
            ssm.draw_sample(T=n_sample)

            for i in range(0, n_sample):
                x = ssm.state_sequence[i]
                y = ssm.output_sequence[i]
                self.assertEqual(x.size, ssm.state_dim)
                self.assertEqual(y.size, ssm.output_dim)

class TestStateSpaceModelInferenceMethods(unittest.TestCase):
    def test_extended_kalman_smoother(self):
        input_dim = 0
        for (state_dim, output_dim) in  zip([1, 4, 2], [1, 2, 4]):
            state_dim = 1
            output_dim = 1

            A = random((state_dim, state_dim))
            C = random((output_dim, state_dim))
            Q = np.ones((state_dim, state_dim))
            R = np.ones((output_dim, output_dim))

            def f(x):
                return A.dot(x)
            def g(x):
                return C.dot(x)
            def df_dx(x):
                return A
            def dg_dx(x):
                return C

            ssm = StateSpaceModel(
                input_dim=input_dim,
                output_dim=output_dim,
                state_dim=state_dim,
                Q=Q,
                R=R,
                f=f,
                g=g,
                df_dx=df_dx,
                dg_dx=dg_dx
            )

            T = 50
            ssm.draw_sample(T=T)

            ssm.extended_kalman_smoother()

            self.assertEqual(len(getattr(ssm, 'filtered_state_means')), T)
            self.assertEqual(len(getattr(ssm, 'filtered_state_covariance')), T)
            self.assertEqual(len(getattr(ssm, 'smoothed_state_means')), T)
            self.assertEqual(len(getattr(ssm, 'smoothed_state_covariance')), T)

            # verifier que les moyennes et covariances estimmées ont les bonnes dimensions
            for i in range(0,T):
                xFilter0 = ssm.filtered_state_means[i][0]
                xFilter1 = ssm.filtered_state_means[i][1]
                PFilter0 = ssm.filtered_state_covariance[i][0]
                PFilter1 = ssm.filtered_state_covariance[i][1]

                xSmooth = ssm.smoothed_state_means[i]
                PSmooth = ssm.smoothed_state_covariance[T - 1 - i]

                self.assertEqual(xFilter0.size, ssm.state_dim, 'mean vector must have state_dim dimension')
                self.assertEqual(xFilter1.size, ssm.state_dim, 'mean vector must have state_dim dimension')
                self.assertEqual(PFilter0.shape, (ssm.state_dim, ssm.state_dim), 'cov matrix must have state_dim dimension')
                self.assertEqual(PFilter1.shape, (ssm.state_dim, ssm.state_dim), 'cov matrix must have state_dim dimension')
                # check that matrices are definite positive
                self.assertTrue(utils.is_pos_def(PFilter0), 'filtered covariance matrix must be positive definite')
                self.assertTrue(utils.is_pos_def(PFilter1), 'filtered covariance matrix must be positive definite')

                self.assertEqual(xSmooth.size, ssm.state_dim, 'mean vector must have state_dim dimension')
                self.assertEqual(PSmooth.shape, (ssm.state_dim, ssm.state_dim), 'cov matrix must have state_dim dimension')
                self.assertTrue(utils.is_pos_def(PSmooth), 'Smoothed covariance P_{T-' + str(i) +'} matrix must be positive definite')


class TestStateSpaceModelLearningMethods(unittest.TestCase):
    """
        Tests learning methods of class StateSpaceModel
    """

    def test_parameter_learning_in_linear_case(self):
        input_dim = 0
        for (state_dim, output_dim) in zip([1], [1]):
            n_sample = 100
            ssm = StateSpaceModel(
                state_dim=state_dim,
                output_dim=output_dim,
                input_dim=input_dim
            )
            ssm.draw_sample(T=n_sample)
            ssm.extended_kalman_smoother()
            ssm.compute_f_optimal_parameters()
            ssm.compute_g_optimal_parameters()

    def test_EM_algorithm_in_1D_linear_case(self):
        n_sample = 100
        input_dim = 0
        state_dim = 1
        output_dim = 1
        A = np.ones((state_dim, state_dim))
        b = rand(state_dim)
        C = np.ones((output_dim, state_dim))
        d = rand(output_dim)
        Q = 0.1 * np.ones((state_dim, state_dim))
        R = 0.1 * np.ones((output_dim, output_dim))

        def f(x):
            return A.dot(x) + b
        def g(x):
            return C.dot(x) + d
        def dg_dx(x):
            return C

        ssm = StateSpaceModel(
            input_dim=input_dim,
            state_dim=state_dim,
            output_dim=output_dim,
            f=f,
            g=g,
            dg_dx=dg_dx,
            Q=Q,
            R=R
        )

        ssm.draw_sample(T=n_sample)
        ssm.plot_states_in_1D(plot_estimated_values=False)

        ssm.f = None
        ssm.initialize_f_linear_parametrization()

        ssm.A[0, 0] = 0.5
        ssm.b[0] = 0.5
        use_smoothed_values = False
        log_likelihood = ssm.learn_f_and_g_with_EM_algorithm(use_smoothed_values=use_smoothed_values)
        plt.figure(10)
        plt.title('f and g linear. log-likelihood evolution during EM algorithm')
        plt.plot(log_likelihood)
        plt.show()
        ssm.plot_states_in_1D(plot_estimated_values=True)

    def test_EM_algorithm_in_1D_non_linear_with_rbf_case(self):
        n_sample = 500
        input_dim = 0
        state_dim = 1
        output_dim = 1
        A = np.ones((state_dim, state_dim))
        b = rand(state_dim)
        C = np.ones((output_dim, state_dim))
        d = rand(output_dim)
        Q = 0.1 * np.ones((state_dim, state_dim))
        R = 0.1 * np.ones((output_dim, output_dim))
        f_rbf_parameters = {
            'n_rbf': 2,
            'centers': np.array([[-0.2], [0.2]]),
            'width': np.array([[[0.02]], [[0.02]]])
        }
        f_rbf_coeffs = np.array([[0.2], [-0.2]])

        def f_1(x):
            f = A.dot(x) + b
            for i in range(f_rbf_parameters['n_rbf']):
                center = f_rbf_parameters['centers'][i]
                width = f_rbf_parameters['width'][i]
                value = f_rbf_coeffs[i]
                f += utils.rbf(value, center, inv(width), x)
            return f

        def f_2(x):
            return np.cos(10 * x) * np.ones(state_dim)

        def g(x):
            return C.dot(x) + d

        def dg_dx(x):
            return C

        for f in [f_1]:
            ssm = StateSpaceModel(
                input_dim=input_dim,
                state_dim=state_dim,
                output_dim=output_dim,
                f=f,
                g=g,
                dg_dx=dg_dx,
                Q=Q,
                R=R
            )

            print "------------------------"
            print ssm.compute_f(np.ones(1))

            ssm.draw_sample(T=n_sample)
            ssm.plot_states_in_1D(plot_estimated_values=False)

            ssm.f = None
            ssm.initialize_f_linear_parametrization()
            ssm.linear_approximation_for_f = False
            ssm.rbf_parameters = f_rbf_parameters
            ssm.f_rbf_coeffs = np.zeros((2, 1))

            use_smoothed_values = False
            log_likelihood = ssm.learn_f_and_g_with_EM_algorithm(use_smoothed_values=use_smoothed_values)
            plt.figure(10)
            plt.title('f and g linear. log-likelihood evolution during EM algorithm')
            plt.plot(log_likelihood)
            plt.show()
            ssm.plot_states_in_1D(plot_estimated_values=True)

#   def test_initialization_with_factor_analysis(self):
#        '''
#        '''
#        n_sample = 100
#        Sigma_0 = np.ones((1, 1))
#        A = np.ones((1, 1))
#        Q = np.ones((1, 1))
#        b = np.zeros(1)
#        C = np.array([[1],[2]])
#        R = np.array([[0.1 ,0] ,[0 ,0.4]])
#        d = np.array([1, 3])


#        ssm = StateSpaceModel(
#            is_f_linear=True,
#            is_g_linear=True,
#            state_dim=1,
#            input_dim=0,
#            output_dim=2,
#            Sigma_0=Sigma_0,
#            A=A,
#            Q=Q,
#            b=b,
#            C=C,
#            R=R,
#            d=d
#        )

#        ssm.draw_sample(T=n_sample)

#        n_iterations_FA = 30
#        likelihood_evolution = ssm.initialize_f_with_factor_analysis(n_iterations_FA)

#        plt.figure(1)
#        plt.plot(likelihood_evolution)
#        plt.title('Expected-complete log-likelihood during EM algo')

#        plt.figure(2)
#        plt.plot(ssm.state_sequence[:, 0])
#        plt.plot(ssm.estimated_state_sequence_with_FA[:, 0])
#        plt.title('comparison between true state sequence and estimmated one.')
#        plt.legend(['true states', 'estimmated states'])
#        plt.show()

test_suite_1 = unittest.TestLoader().loadTestsFromTestCase(TestStateSpaceModelConstructorAndSample)
unittest.TextTestRunner(verbosity=2).run(test_suite_1)

test_suite_2 = unittest.TestLoader().loadTestsFromTestCase(TestStateSpaceModelInferenceMethods)
unittest.TextTestRunner(verbosity=2).run(test_suite_2)

test_suite_3 = unittest.TestLoader().loadTestsFromTestCase(TestStateSpaceModelLearningMethods)
unittest.TextTestRunner(verbosity=2).run(test_suite_3)
