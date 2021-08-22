import os
import pandas as pd
import numpy as np

from tensorflow.keras.utils import Sequence
from imblearn.over_sampling import RandomOverSampler
from imblearn.tensorflow import balanced_batch_generator

class BalancedDataGenerator(Sequence):
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

    Generates batches with minority classes oversampled.

    Parameters:

    data_path: root of the data directories (e.g., DATA/TRAIN)
    classes: list of classes (e.g., ['ClassA', 'ClassB'])
    datagen: instance of Keras ImageDataGenerator. Can include data
             augmentation options.
    target_size: H/W target dimensions of images. Tuple. (e.g., (299, 299))
    batch_size: batch size
    class_mode: class mode for Keras' flow_from_dataframe method. E.g.,
                'binary', 'category', etc. See Keras documentation.

    Reference:
    https://medium.com/analytics-vidhya/how-to-apply-data-augmentation-to-deal-with-unbalanced-datasets-in-20-lines-of-code-ada8521320c9
    
    """
    def __init__(self, data_path, classes, datagen, target_size, batch_size=32, class_mode='categorical'):
        self.data_path = data_path
        self.classes = classes
        self.datagen = datagen
        self.target_size = target_size
        self.class_mode = class_mode

        # We will generate parallel arrays for filenames and labels
        X, y = self.__inventory__()

        self.batch_size = min(batch_size, X.shape[0])
        self.gen, self.steps_per_epoch = balanced_batch_generator(X.reshape(X.shape[0], -1), y, sampler=RandomOverSampler(), batch_size=self.batch_size, keep_sparse=True)
        self._shape = (self.steps_per_epoch * batch_size, *X.shape[1:])
        
    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, idx):
        x_batch, y_batch = self.gen.__next__()
        x_batch = x_batch.reshape(-1)

        # Generate a pandas dataframe with x_batch and y_batch as columns
        df = pd.DataFrame()
        df['image'] = x_batch
        df['label'] = y_batch
       
        # With that, load the batch using the flow_from_dataframe method 
        return self.datagen.flow_from_dataframe(df, 
                                                directory=self.data_path,
                                                x_col='image',
                                                y_col='label',
                                                target_size=self.target_size,
                                                classes=self.classes,
                                                class_mode=self.class_mode,
                                                batch_size=self.batch_size).next()

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

        return np.array(X), np.array(y)
