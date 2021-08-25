import os
import pandas as pd
import numpy as np

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


def getGenerator(data_path, classes, datagen, target_size, strategy=None,
                        batch_size=32, class_mode='categorical', shuffle=True, 
                        random_state=42):
    '''
    Sets up a data generator allows Keras image data generation to resample image 
    files and/or create one-hot encoding of class labels. 

    E.g., Given a directory structure comtaining subdirectories for images divided into 
    classes

    DATA/TRAIN
         ├───ClassA
         |   |---- A1.png
         |   |---- A2.png
         |   ...
         └───ClassB
             |---- B1.png
             |---- B2.png
             ...

    1) balance classes by undersampling non-majority classes or oversampling
       non-minority classes, and/or
    2) Generate labels as one-hot encoding (replacing list of classes with a
       sparse matrix)

    It is designed to integrate with the Keras ImageDataGenerator class so to 
    take advantage of it's other functionality (e.g., data augmentation). Outwardly,
    it appears to be flowinw from a directory, while internally it uses the 
    flow_from_dataframe method of ImageDataGenerator.

    Example implementation:
        datagen = ImageDataGenerator(rescale=1./255.)
        train_gen = getdGenerator(train_data, classes, datagen, (200, 200),
                                  strategy='under')

    to draw images from train_data, target image size of (200, 200), and undersampling
    non-minority classes

    Parameters:

    data_path: root of the data directories (e.g., DATA/TRAIN)
    classes: list of classes (e.g., ['ClassA', 'ClassB'])
    datagen: instance of Keras ImageDataGenerator. Can include data
             augmentation options.
    target_size: H/W target dimensions of images. Tuple. (e.g., (299, 299))
    strategy: If None, don't resample; "under" to undersample non-minority 
              classes or "over" to oversample the non-majority classes. 
              Default is None.
    batch_size: batch size. Default 32
    class_mode: class mode for Keras' flow_from_dataframe method. E.g.,
                'binary', 'categorical', etc. See Keras documentation.
                default is 'categorical'
    random_state: random seed. Default is 42. None if you don't care.

    Returns: the generator to be used in training a model.
    '''
    # List the images in each class into parallel arrays
    X = []
    y = []
    for label in classes:
        class_directory = os.path.join(data_path, label)
        class_files = os.listdir(class_directory)
        for image in class_files:
            class_file = os.path.join(label, image)
            X.append(class_file)
        y += [label] * len(class_files)

    # If resampling is indicated
    if strategy is not None:
        if strategy == 'over':
            resample = RandomOverSampler(random_state=random_state)
        else:
            resample = RandomUnderSampler(random_state=random_state)
        X = np.array(X).reshape(-1, 1)
        y = np.array(y).reshape(-1, 1)
        X_res, y_res = resample.fit_resample(X, y)
        X = X_res.reshape(-1)
        y = y_res.reshape(-1)

    # make into a dataframe
    df = pd.DataFrame(list(zip(X, y)), columns=['image', 'label'])

    # if categorical, translate labels into on-hot encoding
    if class_mode == 'categorical':
        df = pd.get_dummies(df, columns=['label'])
        
    y_columns = list(df.columns[1:])

    # instantiate generator. 
    gen = datagen.flow_from_dataframe(df, 
                                      directory=data_path,
                                      x_col='image',
                                      y_col=y_columns,
                                      target_size=target_size,
                                      classes=classes,
                                      class_mode=class_mode,
                                      batch_size=batch_size,
                                      shuffle=shuffle,
                                      random_state=random_state)
    return gen
