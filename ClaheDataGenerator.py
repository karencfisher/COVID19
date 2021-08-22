import os
import pandas as pd
import numpy as np
import cv2

from tensorflow.keras.utils import Sequence
from imblearn.over_sampling import RandomOverSampler
from imblearn.tensorflow import balanced_batch_generator
from sklearn.utils import shuffle


class ClahedDataGenerator(Sequence):
    """
    ImageDataGenerator + RandomOversampling from directory

    Given a directory structure comtaining subdirectories for images divided into 
    classes, e.g.

    DATA/TRAIN
         ├───ClassA
         |   |---- A1.png
         |   |---- A2.png
         |   ...
         └───ClassB
             |---- B1.png
             |---- B2.png
             ...

    Parameters:

    data_path: root of the data directories (e.g., DATA/TRAIN)
    classes: list of classes (e.g., ['COVID', 'normal'])
    target_size: H/W target dimensions of images. Tuple. (e.g., (299, 299))
    batch_size: batch size
    class_mode: class mode for Keras' flow_from_dataframe method. E.g.,
                'binary', 'category', etc. See Keras documentation.

   
    """
    def __init__(self, data_path, classes, target_size, batch_size=32, 
                 random_state=42):
        self.data_path = data_path
        self.classes = classes
        self.target_size = target_size

        # We will generate parallel arrays for filenames and labels
        self.X, self.y = self.__inventory__()
        print(f'found {len(self.X)} images in {len(self.classes)} classes')
        self.batch_size = min(batch_size, len(self.X))

        # shuffle them
        self.X, self.y = shuffle(self.X, self.y, random_state=random_state)
        
    def __len__(self):
        return self.batch_size

    def __getitem__(self, idx):
        batch_x = self.X[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        y_labels = [batch_y.index(label) for label in batch_y]

        images = []
        for img_file in batch_x:
            img_path = os.path.join(self.data_path, img_file)
            img = cv2.imread(img_path)
            gs_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            eq_img = clahe.apply(gs_img)
            #eq_img = cv2.cvtColor(eq_img, cv2.COLOR_GRAY2RGB)
            eq_img = eq_img.astype(np.float32) / 255.
            images.append(eq_img)

        return np.stack(images, axis=0), np.array(y_labels)

    def __inventory__(self):
        '''
        Help function to create X and y
        '''
        X = []
        y = []

        for label in self.classes:
            class_directory = os.path.join(self.data_path, label)
            class_files = os.listdir(class_directory)
            for image in class_files:
                class_file = os.path.join(label, image)
                X.append(class_file)
            y += [label] * len(class_files)

        return X, y
