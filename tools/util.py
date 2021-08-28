''' Model utilities '''
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import backend as K
from tensorflow import GradientTape
from tensorflow.keras.models import Model
from tensorflow import cast, reduce_mean
from sklearn.metrics import roc_auc_score, roc_curve


def Weighted_Loss(y_true, epsilon=1e-7):
    '''
    Weighted_Loss function -- genrates a loss function

    parameters:
    y_true: Ground truth for all classes, as an Numpy Array. Either one or two 
            dimensional array.
    epsilon: tiny value to prevent division by zer exceptions. Default 1e-7.

    returns:
    Loss function

    effects:
    None
    '''
    if len(y_true.shape) < 2:
        y_true = np.expand_dims(y_true, axis=0)

    neg_w = np.sum(y_true, axis=0) / y_true.shape[0]
    pos_w = 1 - neg_w

    def weighted_loss(y_true, y_pred):
        '''
        Weighted Binary Crossentropy Loss (for multiple classes)

        parameters:
        y_true: Ground truth for all classes
        y_pred: Predicted classes

        Both are Numpy array, one or two dimensions, as one-hot encoded labels.

        returns:
        weighted_loss (as scalar value)
        '''
        print(y_true.shape, y_pred.shape)
        assert y_true.shape == y_pred.shape, "mismatched shapes y_true and y_pred"
        if len(y_true.shape) < 2:
            y_true = np.expand_dims(y_true, axis=0)
        if len(y_pred.shape) < 2:
            y_pred = np.expand_dims(y_pred, axis=0)

        loss = 0
        for i in range(len(pos_w)):
            pos_loss = -1 * K.mean(pos_w[i] * y_true[:,i] * K.log(y_pred[:,i] + epsilon))
            neg_loss = -1 * K.mean(neg_w[i] * (1 - y_true[:,i]) * 
                                        K.log(1 - y_pred[:,i] + epsilon))
            loss += pos_loss + neg_loss
        return loss

    return weighted_loss


def model_metrics(y_true, y_pred, labels):
    '''
    Calculate metrics (accuracy, sensitivity, specificity, ppv, roc curves and
    auc scores) for each class.

    Paremeters:
    y_true: Ground truth for all classes
    y_pred: Predicted classes
            Both are Numpy array, one or two dimensions, as one-hot encoded labels.
    labels: array of the labels (as strings)

    returns: nested dictionary
               {classA: {'accuracy': accuracy,
                         'sensitivity': sensitivity,
                         'specificity': specificity,
                         'ppv': ppv,
                         'auc_score': auc_score,
                         'roc_curve': {'tpr': tpr, 'fpr': fpr},
                classB: ...}

    effects:
    None  
    '''
    assert y_true.shape == y_pred.shape, "mismatched shapes y_true and y_pred"
    if len(y_true.shape) < 2:
        y_true = np.expand_dims(y_true, axis=0)
    if len(y_pred.shape) < 2:
        y_pred = np.expand_dims(y_pred, axis=0)

    metrics = {}
    for i in range(y_true.shape[0]):
        # tp, fp, tn, fn
        tp = np.sum((y_true[i] == 1) & (y_pred[i] == 1))
        fp = np.sum((y_true[i] == 0) & (y_pred[i] == 1))
        tn = np.sum((y_true[i] == 0) & (y_pred[i] == 0))
        fn = np.sum((y_true[i] == 1) & (y_pred[i] == 0))

        # sensitivity, specificity
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
            
        # Calculate PPV according to Bayes Theorem
        prev = np.sum(y_true[i]) / len(y_true[i])
        numerator = sensitivity * prev
        denominator = sensitivity * prev + (1 - specificity) * (1 - prev)
        ppv = numerator / denominator

        #claculate ROC and AUC
        tpr, fpr, _ = roc_curve(y_true[i], y_pred[i])
        auc_score = roc_auc_score(y_true[i], y_pred[i])

        metrics[labels[i]] = {'accuracy': accuracy,
                              'sensitivity': sensitivity,
                              'specificity': specificity,
                              'ppv': ppv,
                              'auc_score': auc_score,
                              'roc_curve': {'tpr': tpr, 'fpr': fpr}}
    return metrics


def grad_cam(model, image, cls, layer_name, test=False):
    '''
    GradCAM method for visualizing input saliency.

    parameters:
    model: the model in use
    image: a chosen image as array (w, h, 3)
    cls: index of the labels
    layer_name: layer of the model to obtain feature maps

    returns:
    class activation map for image, as image array

    effects:
    None

    Thank you to https://gist.github.com/RaphaelMeudec/e9a805fa82880876f8d89766f0690b54
    '''
    grad_model = Model([model.inputs], 
                       [model.get_layer(layer_name).output, model.output])

    with GradientTape() as tape:
        conv_outputs, predictions = grad_model([image])
        loss = predictions[:, cls]

    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]

    guided_grads = cast(output > 0, 'float32') * cast(grads > 0, 'float32') * grads
    weights = reduce_mean(guided_grads, axis=(0, 1))

    cam = np.ones(output.shape[0: 2], dtype = np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    cam = cv2.resize(cam.numpy(), (image.shape[1], image.shape[2]))
    cam = np.maximum(cam, 0)
    cam = cv2.applyColorMap(np.uint8(cam), cv2.COLORMAP_JET)
    return cam
