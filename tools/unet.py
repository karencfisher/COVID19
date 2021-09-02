from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization, Input
from tensorflow.keras.layers import Conv2DTranspose, Concatenate, Activation
from tensorflow.keras import backend as K


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

