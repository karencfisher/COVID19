import unittest
import numpy as np
import os
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array


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
        y_true = np.expand_dims(y_true, axis=1)
        y_pred = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1])
        y_pred = np.expand_dims(y_pred, axis=1)
        labels = np.array(['sick'])

        df, _ = util.model_metrics(y_true, y_pred, labels)
        scores = df.loc['sick']

        self.assertAlmostEqual(round(scores['Accuracy'], 4), .7)
        self.assertAlmostEqual(round(scores['Sensitivity'], 4), .7143)
        self.assertAlmostEqual(round(scores['Specificity'], 4), .6667)
        self.assertAlmostEqual(round(scores['PPV'], 4), .8333)

    def test_Model_metrics2(self):
        y_true = np.array([[1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
                           [1, 1, 1, 0, 0, 0, 1, 1, 1, 1]]).T
        y_pred = np.array([[1, 0, 1, 1, 0, 0, 1, 0, 1, 1],
                           [1, 0, 0, 1, 0, 0, 1, 0, 1, 1]]).T
        labels = np.array(['sick', 'well'])

        df, _ = util.model_metrics(y_true, y_pred, labels)
        scores1 = df.loc['sick']
        self.assertAlmostEqual(round(scores1['PPV'], 4), .8333)
        scores2 = df.loc['well']
        self.assertAlmostEqual(round(scores2['PPV'], 4), .8000)

    def test_cam(self):
        cwd = os.getcwd()
        model_path = os.path.join(cwd, 'tools', 'test_model.h5')
        model = load_model(model_path, compile=False)

        img_path = os.path.join(cwd, 'tools', 'test_img.png')
        img = load_img(img_path)
        img = img_to_array(img)
        img = img / 255.
        img = np.expand_dims(img, axis=0)
        cam = util.grad_cam(model, img, 0, 'conv2d_4', test=True)
        self.assertEqual(cam.shape, img[0].shape)


if __name__ == '__main__':
    print('testing')
    unittest.main()  