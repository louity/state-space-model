# coding: utf8

import unittest
import numpy as np
from kalman import StateSpaceModel
from numpy.random import rand, random
import matplotlib.pyplot as plt

def is_pos_def(M):
    return np.all(np.linalg.eigvals(M) > 0)

STATE_SPACE_MODEL_MINIMAL_ATTRIBUTES = ['is_f_linear', 'state_dim', 'input_dim', 'output_dim', 'Sigma_0', 'A', 'Q', 'C', 'R']

class TestStateSpaceModel(unittest.TestCase):
    """
        Test case utilisé pour tester la classe StateSpaceModel"""

    def test_constructor(self):
        """
            teste la construction d'une instance sans paramètre
        """

        # tester le cas lineaire et non lineaire
        for i, is_f_linear in enumerate([True, False]):
            ssm = StateSpaceModel(is_f_linear=is_f_linear)

            # verifier que les attributs minimaux sont bien définis
            for j, attr in enumerate(STATE_SPACE_MODEL_MINIMAL_ATTRIBUTES):
                self.assertIsNotNone(getattr(ssm, attr), 'attribute ' + attr + ' should not be None')

    def test_sample_method(self):
        """
            teste la méthode sample dans plusieurs cas
        """
        # tester le cas lineaire et non lineaire
        for i, is_f_linear in enumerate([True, False]):
            # tester différentes dimensions pour les espaces
            for j, (state_dim, output_dim, input_dim) in enumerate(zip([1, 4, 3], [1, 3, 4], [0, 0, 0])):
                # tester différentes tailles
                for k, n_sample in enumerate([1, 10]):
                    ssm = StateSpaceModel(is_f_linear=is_f_linear, state_dim=state_dim, output_dim=output_dim, input_dim=input_dim)
                    ssm.draw_sample(T=n_sample)
                    self.assertEqual(len(getattr(ssm, 'state_sequence')), n_sample)

                    # verifier que les vecteurs tirés ont les bonnes dimensions
                    for i in range(0, n_sample):
                        x = ssm.state_sequence[i]
                        y = ssm.output_sequence[i]
                        self.assertEqual(x.size, ssm.state_dim)
                        self.assertEqual(y.size, ssm.output_dim)

    def test_linear_kalman_methods(self):
        """
            teste les méthodes kalman_filtering et kalman_smoothing dans le cas linéaire
        """
        for i, (state_dim, output_dim, input_dim) in enumerate(zip([1, 4, 3], [1, 3, 4], [0, 0, 0])):
            # tester différentes tailles
            for j, T in enumerate([1, 10]):
                default_message = 'Parameters : T = ' + str(T) + '. state_dim = ' + str(state_dim) + '.  output_dim = ' + str(output_dim) + '. input_dim = ' + str(input_dim)
                ssm = StateSpaceModel(is_f_linear=True, is_g_linear=True, state_dim=state_dim, output_dim=output_dim, input_dim=input_dim)
                ssm.draw_sample(T=T)
                ssm.kalman_smoothing(is_extended=False)
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
                    self.assertTrue(is_pos_def(PFilter0), 'filtered covariance matrix must be positive definite')
                    self.assertTrue(is_pos_def(PFilter1), 'filtered covariance matrix must be positive definite')

                    self.assertEqual(xSmooth.size, ssm.state_dim, 'mean vector must have state_dim dimension')
                    self.assertEqual(PSmooth.shape, (ssm.state_dim, ssm.state_dim), 'cov matrix must have state_dim dimension')
                    self.assertTrue(is_pos_def(PSmooth), default_message + '. Smoothed covariance P_{T-' + str(i) +'} matrix must be positive definite')

    def test_extended_kalman_filter(self):
        """
            teste la méthodes kalman_filtering dans le cas ou f ou g sont non linéaires
        """
        for i, (is_f_linear, is_g_linear) in enumerate(zip([False, True, False], [True, False, False])):
            # tester différentes dimensions pour les espaces
            for j, (state_dim, output_dim, input_dim) in enumerate(zip([1, 4, 3], [1, 3, 4], [0, 0, 0])):
                # tester différentes tailles
                for k, T in enumerate([1, 10]):
                    ssm = StateSpaceModel(is_f_linear=is_f_linear, is_g_linear=is_g_linear, state_dim=state_dim, output_dim=output_dim, input_dim=input_dim)
                    ssm.draw_sample(T=T)
                    ssm.kalman_filtering(is_extended=True)
                    self.assertEqual(len(getattr(ssm, 'filtered_state_means')), T)
                    self.assertEqual(len(getattr(ssm, 'filtered_state_covariance')), T)

                    # verifier que les moyennes et covariances estimmées ont les bonnes dimensions
                    for i in range(0,T):
                        xFilter0 = ssm.filtered_state_means[i][0]
                        xFilter1 = ssm.filtered_state_means[i][1]
                        PFilter0 = ssm.filtered_state_covariance[i][0]
                        PFilter1 = ssm.filtered_state_covariance[i][1]

                        self.assertEqual(xFilter0.size, ssm.state_dim, 'mean vector must have state_dim dimension')
                        self.assertEqual(xFilter1.size, ssm.state_dim, 'mean vector must have state_dim dimension')
                        self.assertEqual(PFilter0.shape, (ssm.state_dim, ssm.state_dim), 'cov matrix must have state_dim dimension')
                        self.assertEqual(PFilter1.shape, (ssm.state_dim, ssm.state_dim), 'cov matrix must have state_dim dimension')
                        # check that matrices are definite positive
                        self.assertTrue(is_pos_def(PFilter0), 'filtered covariance matrix must be positive definite')
                        self.assertTrue(is_pos_def(PFilter1), 'filtered covariance matrix must be positive definite')

    def test_extended_kalman_smoother(self):
        """
            teste les méthodes kalman_filtering et kalman_smoothing dans le cas ou f ou g sont non linéaires
        """
        for i, (is_f_linear, is_g_linear) in enumerate(zip([False, True, False], [True, False, False])):
            # tester différentes dimensions pour les espaces
            for j, (state_dim, output_dim, input_dim) in enumerate(zip([1, 4, 3], [1, 3, 4], [0, 0, 0])):
                # tester différentes tailles
                for k, T in enumerate([1, 10]):
                    ssm = StateSpaceModel(is_f_linear=is_f_linear, is_g_linear=is_g_linear, state_dim=state_dim, output_dim=output_dim, input_dim=input_dim)
                    ssm.draw_sample(T=T)
                    ssm.kalman_smoothing(is_extended=True)
                    self.assertEqual(len(getattr(ssm, 'filtered_state_means')), T)
                    self.assertEqual(len(getattr(ssm, 'filtered_state_covariance')), T)
                    self.assertEqual(len(getattr(ssm, 'smoothed_state_means')), T)
                    self.assertEqual(len(getattr(ssm, 'smoothed_state_covariance')), T)

                    # verifier que les moyennes et covariances estimmées ont les bonnes dimensions
                    for i in range(0, T):
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
                        self.assertTrue(is_pos_def(PFilter0), 'filtered covariance matrix must be positive definite')
                        self.assertTrue(is_pos_def(PFilter1), 'filtered covariance matrix must be positive definite')

                        self.assertEqual(xSmooth.size, ssm.state_dim, 'mean vector must have state_dim dimension')
                        self.assertEqual(PSmooth.shape, (ssm.state_dim, ssm.state_dim), 'cov matrix must have state_dim dimension')
                        self.assertTrue(is_pos_def(PSmooth), 'is_f_linear' + str(is_f_linear) + '.  is_g_linear' + str(is_g_linear) + '.  state_dim' + str(state_dim) + '.  output_dim' + str(output_dim) + '. Smoothed covariance P_{T-' + str(i) +'} matrix must be positive definite')

    def test_parameter_learning_in_linear_case(self):
        """
            teste la méthode compute_f_optimal_parametes
        """
        is_f_linear = True
        is_g_linear = True
        for j, (state_dim, output_dim) in enumerate(zip([1], [1])):
            # tester différentes tailles
            for k, n_sample in enumerate([100]):
                ssm = StateSpaceModel(is_f_linear=is_f_linear, is_g_linear=is_g_linear, state_dim=state_dim, output_dim=output_dim)
                ssm.draw_sample(T=n_sample)
                is_extended = False
                ssm.kalman_smoothing(is_extended=is_extended)
                ssm.compute_f_optimal_parameters()
                ssm.compute_g_optimal_parameters()

    def test_EM_algorithm_in_linear_case(self):
        n_sample = 50
        is_f_linear = True
        is_g_linear = True
        state_dim = 1
        output_dim = 1
        A = np.ones((1, 1))
        b = rand(1)
        C = np.ones((1, 1))
        d = rand(1)
        Q = np.array([[0.1]])
        R = np.array([[0.1]])
        ssm = StateSpaceModel(
            is_f_linear=is_f_linear,
            is_g_linear=is_g_linear,
            state_dim=state_dim,
            output_dim=output_dim,
            A=A,
            b=b,
            C=C,
            d=d,
            Q=Q,
            R=R
        )
        ssm.draw_sample(T=n_sample)
        ssm.plot_states_in_1D()

        ssm.A[0, 0] += 0.1 * random()
        ssm.b[0] += 0.1 * random()
        ssm.C[0, 0] += 0.1 * random()
        ssm.d[0] += 0.1 * random()
        log_likelihood = ssm.learn_f_and_g_with_EM_algorithm(use_smoothed_values=False)
        plt.figure(1)
        plt.title('f and g linear. log-likelihood evolution during EM algorithm')
        plt.plot(log_likelihood)
        plt.show()
        ssm.plot_states_in_1D()

    def test_EM_algorithm_in_non_linear_case(self):
        n_sample = 50
        is_f_linear = False
        is_g_linear = True
        state_dim = 1
        output_dim = 1
        A = 0.5 * np.ones((1, 1))
        b = 0.25 * np.ones(1)
        C = np.ones((1, 1))
        d = np.zeros(1)
        Q = np.array([[0.04]])
        R = np.array([[0.04]])
        f_rbf_parameters = {
            'n_rbf': 2,
            'centers': np.array([[-0.2], [0.2]]),
            'width': np.array([[[0.02]], [[0.02]]])
        }
        f_rbf_coeffs = np.array([[0.2], [-0.2]])

        ssm = StateSpaceModel(
            is_f_linear=is_f_linear,
            is_g_linear=is_g_linear,
            f_rbf_parameters=f_rbf_parameters,
            f_rbf_coeffs=f_rbf_coeffs,
            state_dim=state_dim,
            output_dim=output_dim,
            A=A,
            b=b,
            C=C,
            d=d,
            Q=Q,
            R=R
        )
        ssm.draw_sample(T=n_sample)
        ssm.plot_states_in_1D()

        #ssm.A[0, 0] += 0.1 * random()
        #ssm.b[0] += 0.1 * random()
        #ssm.C[0, 0] += 0.1 * random()
        #ssm.d[0] += 0.1 * random()
        log_likelihood = ssm.learn_f_and_g_with_EM_algorithm(use_smoothed_values=False)
        plt.figure(1)
        plt.title('f non-linear, g linear. log-likelihood evolution during EM algorithm')
        plt.plot(log_likelihood)
        plt.show()
        ssm.plot_states_in_1D()


# lance les tests
suite = unittest.TestLoader().loadTestsFromTestCase(TestStateSpaceModel)
unittest.TextTestRunner(verbosity=2).run(suite)
