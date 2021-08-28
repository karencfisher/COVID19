import unittest
import numpy as np
import os
from tensorflow.keras import backend as K


import util


class smokeTest(unittest.TestCase):
    def test_weighted_loss(self):
        y_true = K.constant(np.array([[1, 1, 1],
                                     [1, 1, 0],
                                     [0, 1, 0],
                                     [1, 0, 1]]))

        y_pred_1 = K.constant(0.7*np.ones(y_true.shape))
        y_pred_2 = K.constant(0.3*np.ones(y_true.shape))
        L = util.Weighted_Loss(y_true, epsilon=1)

        L1 = L(y_true, y_pred_1).numpy()
        L2 = L(y_true, y_pred_2).numpy()

        self.assertEqual(L1, L2)
        self.assertAlmostEqual(round(L1, 4), -0.4956)

    def test_weighted_loss2(self):
        y_true = K.constant(np.array([[1], [1], [1], [0], [0], [0]]))

        y_pred_1 = K.constant(0.7*np.ones(y_true.shape))
        y_pred_2 = K.constant(0.3*np.ones(y_true.shape))
        L = util.Weighted_Loss(y_true, epsilon=1)

        L1 = L(y_true, y_pred_1).numpy()
        L2 = L(y_true, y_pred_2).numpy()

        self.assertEqual(L1, L2)

    def test_model_metrics(self):
        y_true = np.array([1, 1, 1, 0, 0, 0, 1, 1, 1, 1])
        y_pred = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1])
        labels = np.array(['sick'])

        scores = util.model_metrics(y_true, y_pred, labels)['sick']
        self.assertAlmostEqual(round(scores['accuracy'], 4), .7)
        self.assertAlmostEqual(round(scores['sensitivity'], 4), .7143)
        self.assertAlmostEqual(round(scores['specificity'], 4), .6667)
        self.assertAlmostEqual(round(scores['ppv'], 4), .8333)

    def test_Model_metrics2(self):
        y_true = np.array([[1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
                           [1, 1, 1, 0, 0, 0, 1, 1, 1, 1]])
        y_pred = np.array([[1, 0, 1, 1, 0, 0, 1, 0, 1, 1],
                           [1, 0, 0, 1, 0, 0, 1, 0, 1, 1]])
        labels = np.array(['sick', 'well'])

        scores = util.model_metrics(y_true, y_pred, labels)
        scores1 = scores['sick']
        self.assertAlmostEqual(round(scores1['ppv'], 4), .8333)
        scores2 = scores['well']
        self.assertAlmostEqual(round(scores2['ppv'], 4), .8000)

    
if __name__ == '__main__':
    print('testing')
    unittest.main()  