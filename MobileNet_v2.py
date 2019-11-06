"""
https://arxiv.org/abs/1801.04381
"""

from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import (Conv2D, BatchNormalization, ReLU, Dense, Add,
                                            DepthwiseConv2D, GlobalAveragePooling2D, Reshape)


def convolution_block(inputs, filters, kernel_size, stride):
    """
    # INPUT
        input: Tensor
    # OUTPUT
        Tensor
    """

    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=stride, padding='same', use_bias=False)(inputs)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
    return ReLU(6.)(x)


def bottleneck_block(inputs, t, filters, stride, adding=False):
    """
    # INPUT
        input: Tensor
        t: Expansion factor
        adding: inverted is YES
    # OUTPUT
        Tensor
    """

    expansion = int(t * inputs.shape[-1])

    #  input: h x w x k
    # output: h x w x (tk)
    # Not use first residuals
    x = convolution_block(inputs, expansion, 1, 1) if filters != 16 else inputs

    #  input: h x w x (tk)
    # output: h/s x w/s x (tk)
    x = DepthwiseConv2D(kernel_size=3, strides=stride, padding='same', use_bias=False)(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
    x = ReLU(6.)(x)

    #  input: h/s x w/s x (tk)
    # output: h/s x w/s x filters
    x = Conv2D(filters=filters, kernel_size=1, strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999)(x)

    if adding:
        x = Add()([inputs, x])

    # h/s x w/s x k'
    return x


def inverted_residuals_block(inputs, t, filters, n, stride):
    x = bottleneck_block(inputs, t, filters, stride, False)

    for _ in range(1, n):
        x = bottleneck_block(x, t, filters, 1, True)

    return x


def MobileNet_v2(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = convolution_block(inputs, 32, 3, 2)

    # input: input, t, filters, n, stride
    x = inverted_residuals_block(x, 1, 16, 1, 1)
    x = inverted_residuals_block(x, 6, 24, 2, 2)
    x = inverted_residuals_block(x, 6, 32, 3, 2)
    x = inverted_residuals_block(x, 6, 64, 4, 2)
    x = inverted_residuals_block(x, 6, 96, 3, 1)
    x = inverted_residuals_block(x, 6, 160, 3, 2)
    x = inverted_residuals_block(x, 6, 320, 1, 1)

    x = convolution_block(x, 1280, 1, 1)
    x = GlobalAveragePooling2D()(x)
    x = Reshape((1, 1, 1280))(x)

    # x = keras.layers.Dropout(rate=0.3)(x)
    # x = keras.layers.Conv2D(filters=num_classes, kernel_size=1, padding='same')(x)
    # x = keras.layers.Softmax()(x)

    x = Dense(num_classes, activation='softmax', use_bias=True)(x)
    output = Reshape((num_classes, ))(x)

    # Create model
    model = Model(inputs, output)

    return model


if __name__ == '__main__':
    input_shape = (48, 48, 1)
    num_classes = 7
    model = MobileNet_v2(input_shape, num_classes)
    model.summary()
