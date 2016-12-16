# coding: utf8
from kalman import StateSpaceModel
import unittest

STATE_SPACE_MODEL_MINIMAL_ATTRIBUTES = ['isLinear', 'state_dim', 'input_dim', 'output_dim', 'Sigma_0', 'A', 'B', 'Q', 'C', 'D', 'R']

class TestStateSpaceModel(unittest.TestCase):
    """
        Test case utilisé pour tester la classe StateSpaceModel"""

    def test_constructor(self):
        """
            teste la construction d'une instance sans paramètre
        """

        # tester le cas lineaire et non lineaire
        for i, isLinear in enumerate([True, False]):
            ssm = StateSpaceModel(isLinear=isLinear)

            # verifier que les attributs minimaux sont bien définis
            for i, attr in enumerate(STATE_SPACE_MODEL_MINIMAL_ATTRIBUTES):
                self.assertIsNotNone(getattr(ssm, attr), 'attribute ' + attr + ' should not be None')

    def test_sample_method(self):
        """
            teste la méthode sample dans plusieurs cas
        """
        # tester le cas lineaire et non lineaire
        for i, isLinear in enumerate([True, False]):
            # tester différentes dimensions pour les espaces
            for j, (state_dim, output_dim) in enumerate(zip([1, 4, 3], [1, 3, 4])):
                # tester différentes tailles
                for k, n_sample in enumerate([1, 10]):
                    ssm = StateSpaceModel(isLinear=isLinear, state_dim=state_dim, output_dim=output_dim)
                    ssm.draw_sample(T=n_sample)
                    self.assertEqual(len(getattr(ssm, 'state_sequence')), n_sample)

                    # verifier que les vecteurs tirés ont les bonnes dimensions
                    for i in range(0,n_sample):
                        x = ssm.state_sequence[i]
                        y = ssm.output_sequence[i]
                        self.assertEqual(x.size, ssm.state_dim)
                        self.assertEqual(y.size, ssm.output_dim)

    def test_kalman_methods(self):
        """
            teste les méthodes kalman_filtering et kalman_smoothing dans plusieurs cas
        """
        ssm = StateSpaceModel()
        ssm.draw_sample()
        ssm.kalman_filtering()
        self.assertEqual(len(getattr(ssm, 'filtered_state_means')), 1)
        self.assertEqual(len(getattr(ssm, 'filtered_state_covariance')), 1)

        n_sample = 10
        state_dim = 4
        output_dim = 3
        ssm = StateSpaceModel(state_dim=state_dim, output_dim=output_dim)
        ssm.draw_sample(T=n_sample)
        ssm.kalman_smoothing()# la methode kalman_smoothing appelle la méthode_kalman filtering
        self.assertEqual(len(getattr(ssm, 'filtered_state_means')), n_sample)
        self.assertEqual(len(getattr(ssm, 'filtered_state_covariance')), n_sample)
        self.assertEqual(len(getattr(ssm, 'smoothed_state_means')), n_sample)
        self.assertEqual(len(getattr(ssm, 'smoothed_state_covariance')), n_sample)

        # verifier que les moyennes et covariances estimmées ont les bonnes dimensions
        for i in range(0,n_sample):
            xFilter0 = ssm.filtered_state_means[i][0]
            xFilter1 = ssm.filtered_state_means[i][1]
            PFilter0 = ssm.filtered_state_covariance[i][0]
            PFilter1 = ssm.filtered_state_covariance[i][1]

            xSmooth = ssm.smoothed_state_means[i]
            PSmooth = ssm.smoothed_state_covariance[i]

            self.assertEqual(xFilter0.size, ssm.state_dim, 'mean vector must have state_dim dimension')
            self.assertEqual(xFilter1.size, ssm.state_dim, 'mean vector must have state_dim dimension')
            self.assertEqual(PFilter0.shape, (ssm.state_dim, ssm.state_dim), 'cov matrix must have state_dim dimension')
            self.assertEqual(PFilter1.shape, (ssm.state_dim, ssm.state_dim), 'cov matrix must have state_dim dimension')

            self.assertEqual(xSmooth.size, ssm.state_dim, 'mean vector must have state_dim dimension')
            self.assertEqual(PSmooth.shape, (ssm.state_dim, ssm.state_dim), 'cov matrix must have state_dim dimension')

# lance les tests
unittest.main()
