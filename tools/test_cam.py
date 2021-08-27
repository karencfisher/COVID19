from logging import disable
import unittest
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.compat.v1 import disable_eager_execution

import util


class TestCam(unittest.TestCase):
    def test_cam(self):
        model = load_model('test_model.h5', compile=False)
        img = load_img('test_img.png')
        img = img_to_array(img)
        img = img / 255.
        img = np.expand_dims(img, axis=0)

        _ = model.predict(img)
        disable_eager_execution()
        cam = util.grad_cam(model, img, 0, 'conv2d_4')

        self.assertEqual(cam.shape, img.shape)