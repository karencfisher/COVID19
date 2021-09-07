from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization, Input
from tensorflow.keras.layers import Conv2DTranspose, Concatenate, Activation, Layer
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import smart_resize


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


''' Layer to merge images/masks and zoom'''

class MergeZoom(Layer):
    def __init__(self, batch_size=8, threshold=0.5, **kwargs):
        self.threshold = threshold
        self.batch_size = batch_size
        super(MergeZoom, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MergeZoom, self).build(input_shape)

    def call(self, inputs):
        masks, images = inputs
        masks = K.greater_equal(masks, self.threshold)
        masks = K.cast(masks, 'float32')
        zooms = []

        for j in range(self.batch_size):
            x = K.sum(masks[j], axis=1)
            y = K.sum(masks[j], axis=0)
            xl = 0
            xr = 0
            i = 0
            while K.equal(xl, 0):
                if x[i] > 0:
                    xl = i - 1
                i += 1
        
            i = len(x) - 1
            while K.equal(xr, 0):
                if x[i] > 0:
                    xr = i + 1
                i -= 1
        
            yl = 0
            yr = 0
            i = 0
            while K.equal(yl, 0):
                if y[i] > 0:
                    yl = i - 1
                i += 1

            i = len(y) - 1
            while K.equal(yr, 0):
                if y[i] > 0:
                    yr = i + 1
                i -= 1

            cropped = masks[j][xl:xr, yl:yr] * images[j][xl:xr, yl:yr]
            cropped = smart_resize(cropped, (images.shape[1], images.shape[2]))
            zooms.append(cropped)
        
        self.result = K.stack(zooms)
        return self.result

    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)