"""
https://arxiv.org/abs/1801.04381
"""

from tensorflow.python.keras import Input, Model, layers


def convolution_block(inputs, filters, kernel_size, stride):
    """
    # INPUT
        input: Tensor
    # OUTPUT
        Tensor
    """

    x = layers.Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=stride, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
    return layers.ReLU(max_value=6.0)(x)


def bottleneck_block(inputs, expansion, filters, stride):
    """
    # INPUT
        input: Tensor
        t: Expansion factor
        adding: inverted is YES
    # OUTPUT
        Tensor
    """

    #  input: h x w x k
    # output: h x w x (tk)
    # Not use first residuals
    x = convolution_block(inputs, int(expansion * inputs.shape[-1]), 1, 1) if filters != 16 else inputs

    #  input: h x w x (tk)
    # output: h/s x w/s x (tk)
    if stride == 2:
        x = layers.ZeroPadding2D(padding=(1, 1))(x)
    x = layers.DepthwiseConv2D(kernel_size=3, strides=stride,
                               padding='same' if stride == 1 else 'valid', use_bias=False)(x)
    x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
    x = layers.ReLU(max_value=6.0)(x)

    #  input: h/s x w/s x (tk)
    # output: h/s x w/s x filters
    x = layers.Conv2D(filters=filters, kernel_size=1, strides=1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999)(x)

    if inputs.shape[-1] == x.shape[-1] and stride == 1:
        x = layers.Add()([inputs, x])

    # h/s x w/s x k'
    return x


def inverted_residuals_block(inputs, t, filters, n, stride):
    x = bottleneck_block(inputs, t, filters, stride)

    for _ in range(1, n):
        x = bottleneck_block(x, t, filters, 1)

    return x


def MobileNet_v2(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = convolution_block(inputs, 32, 3, 2)

    # input: input, expansion(t), filters(c), repeat(n), stride(s)
    x = inverted_residuals_block(x, 1, 16, 1, 1)
    x = inverted_residuals_block(x, 6, 24, 2, 2)
    x = inverted_residuals_block(x, 6, 32, 3, 2)
    x = inverted_residuals_block(x, 6, 64, 4, 2)
    x = inverted_residuals_block(x, 6, 96, 3, 1)
    x = inverted_residuals_block(x, 6, 160, 3, 2)
    x = inverted_residuals_block(x, 6, 320, 1, 1)

    x = convolution_block(x, 1280, 1, 1)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Reshape((1, 1, 1280))(x)

    # x = layers.Dropout(rate=0.3)(x)
    # x = layers.Conv2D(filters=num_classes, kernel_size=1, padding='same')(x)
    # x = layers.Softmax()(x)

    x = layers.Dense(num_classes, activation='softmax', use_bias=True)(x)
    output = layers.Reshape((num_classes, ))(x)

    # Create model
    model = Model(inputs, output)

    return model


if __name__ == '__main__':
    input_shape = (224, 224, 3)
    num_classes = 1000
    model = MobileNet_v2(input_shape, num_classes)
    model.summary()
