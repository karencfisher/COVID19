import os
import unittest

import numpy as np

from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.compat.v1 import disable_eager_execution

import util


class TestCam(unittest.TestCase):
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