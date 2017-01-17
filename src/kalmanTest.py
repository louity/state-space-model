# coding: utf8

import unittest
import numpy as np
from kalman import StateSpaceModel
from numpy.random import rand, random
import matplotlib.pyplot as plt

import utils

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
            for j, attr in enumerate(['is_f_linear', 'state_dim', 'input_dim', 'output_dim', 'Sigma_0', 'A', 'Q', 'C', 'R']):
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
                    self.assertTrue(utils.is_pos_def(PFilter0), 'filtered covariance matrix must be positive definite')
                    self.assertTrue(utils.is_pos_def(PFilter1), 'filtered covariance matrix must be positive definite')

                    self.assertEqual(xSmooth.size, ssm.state_dim, 'mean vector must have state_dim dimension')
                    self.assertEqual(PSmooth.shape, (ssm.state_dim, ssm.state_dim), 'cov matrix must have state_dim dimension')
                    self.assertTrue(utils.is_pos_def(PSmooth), default_message + '. Smoothed covariance P_{T-' + str(i) +'} matrix must be positive definite')

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
                        self.assertTrue(utils.is_pos_def(PFilter0), 'filtered covariance matrix must be positive definite')
                        self.assertTrue(utils.is_pos_def(PFilter1), 'filtered covariance matrix must be positive definite')

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
                        self.assertTrue(utils.is_pos_def(PFilter0), 'filtered covariance matrix must be positive definite')
                        self.assertTrue(utils.is_pos_def(PFilter1), 'filtered covariance matrix must be positive definite')

                        self.assertEqual(xSmooth.size, ssm.state_dim, 'mean vector must have state_dim dimension')
                        self.assertEqual(PSmooth.shape, (ssm.state_dim, ssm.state_dim), 'cov matrix must have state_dim dimension')
                        self.assertTrue(utils.is_pos_def(PSmooth), 'is_f_linear' + str(is_f_linear) + '.  is_g_linear' + str(is_g_linear) + '.  state_dim' + str(state_dim) + '.  output_dim' + str(output_dim) + '. Smoothed covariance P_{T-' + str(i) +'} matrix must be positive definite')

#    def test_parameter_learning_in_linear_case(self):
#        """
#            teste la méthode compute_f_optimal_parametes
#        """
#        is_f_linear = True
#        is_g_linear = True
#        for j, (state_dim, output_dim) in enumerate(zip([1], [1])):
#            # tester différentes tailles
#            for k, n_sample in enumerate([100]):
#                ssm = StateSpaceModel(is_f_linear=is_f_linear, is_g_linear=is_g_linear, state_dim=state_dim, output_dim=output_dim)
#                ssm.draw_sample(T=n_sample)
#                is_extended = False
#                ssm.kalman_smoothing(is_extended=is_extended)
#                ssm.compute_f_optimal_parameters()
#                ssm.compute_g_optimal_parameters()
#
#    def test_EM_algorithm_in_linear_case(self):
#        n_sample = 100
#        is_f_linear = True
#        is_g_linear = True
#        state_dim = 1
#        output_dim = 1
#        A = np.ones((1, 1))
#        b = rand(1)
#        C = np.ones((1, 1))
#        d = rand(1)
#        Q = np.array([[0.1]])
#        R = np.array([[0.1]])
#        ssm = StateSpaceModel(
#            is_f_linear=is_f_linear,
#            is_g_linear=is_g_linear,
#            state_dim=state_dim,
#            output_dim=output_dim,
#            A=A,
#            b=b,
#            C=C,
#            d=d,
#            Q=Q,
#            R=R
#        )
#        ssm.draw_sample(T=n_sample)
#        ssm.plot_states_in_1D()
#
#        ssm.A[0, 0] = 0.5
#        ssm.b[0] = 0.5
#        ssm.C[0, 0] = 0.5
#        ssm.d[0] = 0.5
#        use_smoothed_values = False
#        log_likelihood = ssm.learn_f_and_g_with_EM_algorithm(use_smoothed_values=use_smoothed_values)
#        plt.figure(10)
#        plt.title('f and g linear. log-likelihood evolution during EM algorithm')
#        plt.plot(log_likelihood)
#        plt.show()
#        ssm.plot_estimmated_states_in_1D(use_smoothed_values=use_smoothed_values)

    def test_EM_algorithm_in_non_linear_case(self):
        n_sample = 5000
        is_f_linear = False
        is_g_linear = True
        state_dim = 1
        output_dim = 1
        A = 0.5 * np.ones((1, 1))
        b = 0.25 * np.ones(1)
        C = np.ones((1, 1))
        d = np.zeros(1)
        Q = np.array([[0.01]])
        R = np.array([[0.01]])
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
        
        #on va definir une f_analytical quelconque
        #ssm.f_analytical=np.cos
        
        def f_Not_RBF_scalar(x):
            value=0
            value=np.cos(10*x)
            return(value)
        
        f_Not_RBF=np.vectorize(f_Not_RBF_scalar)
        ssm.f_analytical=f_Not_RBF
        
        ssm.draw_sample(T=n_sample)
        ssm.plot_states_in_1D()

        ssm.A[0, 0] += 0.1 * random()
        ssm.b[0] += 0.1 * random()
        ssm.C[0, 0] += 0.1 * random()
        ssm.d[0] += 0.1 * random()
        
        #et si on commence a apprendre avec plus de I qu'il n'y en a réellement
        f_rbf_parameters_Bis = {
            'n_rbf': 4,
            'centers': np.array([[-0.2], [0.2], [0.3], [0.7]]),
            'width': np.array([[[0.02]], [[0.02]],[[0.02]],[[0.02]]])
        }
        ssm.f_rbf_parameters=f_rbf_parameters_Bis
        ssm.f_rbf_coeffs= np.array([[-0.2], [0.2], [0.3], [0.7]])
        #ssm.f_rbf_coeffs = np.array([[0.], [0.]])
        use_smoothed_values = False
        log_likelihood = ssm.learn_f_and_g_with_EM_algorithm(use_smoothed_values=use_smoothed_values)
        plt.figure(1)
        plt.title('f non-linear, g linear. log-likelihood evolution during EM algorithm')
        plt.plot(log_likelihood)
        plt.xlabel('Number of Iteration')
        plt.ylabel('likelihood')
        plt.show()
        #pour que compute_f me renvoit ce qui a ete appris
        ssm.f_analytical=None
        ssm.plot_estimmated_states_in_1D(use_smoothed_values=use_smoothed_values)
        print(ssm.f_rbf_coeffs)
        


#    def test_initialization_with_factor_analysis(self):
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
#
#
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
#
#        ssm.draw_sample(T=n_sample)
#
#        n_iterations_FA = 30
#        likelihood_evolution = ssm.initialize_f_with_factor_analysis(n_iterations_FA)
#
#        plt.figure(1)
#        plt.plot(likelihood_evolution)
#        plt.title('Expected-complete log-likelihood during EM algo')
#
#        plt.figure(2)
#        plt.plot(ssm.state_sequence[:, 0])
#        plt.plot(ssm.estimated_state_sequence_with_FA[:, 0])
#        plt.title('comparison between true state sequence and estimmated one.')
#        plt.legend(['true states', 'estimmated states'])
#        plt.show()

# lance les tests
suite = unittest.TestLoader().loadTestsFromTestCase(TestStateSpaceModel)
unittest.TextTestRunner(verbosity=2).run(suite)
