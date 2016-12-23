# coding: utf8

import unittest
import numpy as np
from kalman import StateSpaceModel

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

    def test_kalman_methods(self):
        """
            teste les méthodes kalman_filtering et kalman_smoothing dans plusieurs cas
        """
        for i, (is_f_linear, is_g_linear) in enumerate(zip([True, False, True, False], [True, True, False, False])):
            # tester différentes dimensions pour les espaces
            for j, (state_dim, output_dim, input_dim) in enumerate(zip([1, 4, 3], [1, 3, 4], [0, 0, 0])):
                # tester différentes tailles
                for k, T in enumerate([1, 10]):
                    ssm = StateSpaceModel(is_f_linear=is_f_linear, is_g_linear=is_g_linear, state_dim=state_dim, output_dim=output_dim, input_dim=input_dim)
                    ssm.draw_sample(T=T)
                    is_extended = not is_f_linear or not is_g_linear
                    ssm.kalman_smoothing(is_extended=is_extended)# la methode kalman_smoothing appelle la méthode_kalman filtering
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
                        self.assertTrue(is_pos_def(PSmooth), 'smoothed covariance P_{T-' + str(i) +'} matrix must be positive definite')


    def test_parameter_learning(self):
        """
            teste la méthode compute_f_optimal_parametes
        """
        print '>>>>>>>>> Test parameter learning <<<<<<<<<<<<<<<<'
        for i, (is_f_linear, is_g_linear) in enumerate(zip([True, False, True, False], [True, True, False, False])):
            # tester différentes dimensions pour les espaces
            for j, (state_dim, output_dim) in enumerate(zip([1], [1])):
                # tester différentes tailles
                for k, n_sample in enumerate([15]):
                    print 'f linear :', is_f_linear, '. g linear :', is_g_linear, ' state_dim :', state_dim, ' output_dim :', output_dim, ' n_sample :', n_sample
                    ssm = StateSpaceModel(is_f_linear=is_f_linear, is_g_linear=is_g_linear, state_dim=state_dim, output_dim=output_dim)
                    ssm.draw_sample(T=n_sample)
                    is_extended = not is_f_linear or not is_g_linear
                    ssm.kalman_smoothing(is_extended=is_extended)
                    ssm.compute_f_optimal_parameters()

# lance les tests
unittest.main()
