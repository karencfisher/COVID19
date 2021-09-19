from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization, Input
from tensorflow.keras.layers import Conv2DTranspose, Concatenate, Activation, Layer
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import smart_resize
import tensorflow as tf


''' Model building functions '''

def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding='same')(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(num_filters, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def encoder_block(input, num_filters):
    x = conv_block(input, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p

def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding='same')(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def build_unet(input_shape, num_layers, min_num_filters=64):
    inputs = Input(input_shape)
    skip_features = []

    s, x = encoder_block(inputs, min_num_filters)
    skip_features.append(s)

    for i in range(1, num_layers):
        num_filters = min_num_filters * 2 ** i
        s, x = encoder_block(x, num_filters)
        skip_features.append(s)

    num_filters = min_num_filters * 2 ** num_layers
    x = conv_block(x, num_filters)

    for i in range(num_layers):
        num_filters /= 2
        s = skip_features.pop()
        x = decoder_block(x, s, num_filters)

    x = Conv2D(1, 1, padding='same')(x)
    outputs = Activation('sigmoid')(x)

    model = Model(inputs, outputs, name='U-Net')
    return model


''' model evaluation functions'''

def dice_coeff(y_true, y_pred, epsilon=1e-7):
    dice_numerator = 2 * K.sum(y_true * y_pred, axis=(1, 2)) + epsilon
    dice_denominator = (K.sum(y_true, axis=(1, 2)) + 
                        K.sum(y_pred, axis=(1, 2)) + epsilon)
    coeff = K.mean(dice_numerator / dice_denominator)
    return coeff

def get_dice_loss(epsilon=1e-7):
    '''
    Get dice loss function

    parameter:
    epsilon: smoothing, default 1e-7

    returns:
    dice loss function
    '''
    def dice_loss(y_true, y_pred):
        loss = 1 - dice_coeff(y_true, y_pred, epsilon=epsilon)
        return loss
    return dice_loss

'''Threshold masks 

   So pixels[i,j] ε {0, 1}
'''

def thresh(x, threshold=0.5):
    greater = K.greater_equal(x, threshold) #will return boolean values
    greater = K.cast(greater, dtype=K.floatx()) #will convert bool to 0 and 1    
    return greater 


''' Layer to merge images/masks and zoom'''

class MergeZoom(Layer):
    def __init__(self, threshold=0.5, **kwargs):
        self.threshold = threshold
        super(MergeZoom, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({'threshold': self.threshold})
        return config

    def build(self, input_shape):
        super(MergeZoom, self).build(input_shape)

    def call(self, inputs):
        image_shape = inputs[1].shape[1:]
        cropped = tf.map_fn(self.crop_images, 
                            inputs,
                            fn_output_signature=tf.TensorSpec(image_shape))
        return cropped

    def crop_images(self, inputs):
        mask, image = inputs
        mask = K.greater_equal(mask, self.threshold)
        mask = K.cast(mask, 'float32')

        if tf.reduce_all(mask == 0):
            return image

        x = K.sum(mask, axis=0)
        y = K.sum(mask, axis=1)

        xl, xr = self.find_edges(x)
        yl, yr = self.find_edges(x)

        cropped = mask[yl:yr, xl:xr, :] * image[yl:yr, xl:xr, :]
        cropped = tf.image.resize(cropped, 
                                  (image.shape[0], image.shape[1]),
                                  preserve_aspect_ratio=False)
        return cropped

    def find_edges(self, x):
        xl = 0
        xr = 0
        i = 0
        while K.equal(xl, 0):
            xl = tf.cond(x[i,0] > 0, lambda: i - 3, lambda: xl)
            i += 1
    
        i = len(x) - 1
        while K.equal(xr, 0):
            xr = tf.cond(x[i,0] > 0, lambda: i + 3, lambda: xr)
            i -= 1

        xl = K.maximum(xl, 0)
        xr = K.minimum(xr, len(x))
        return xl, xr
