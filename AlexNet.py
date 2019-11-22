"""
ImageNet Classification with Deep Convolutional Neural Networks
Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton
"""

from tensorflow.python.keras import Input, Model, layers


def convolution_layer(inputs, filters, kernel_size, strides, padding):
    x = layers.Conv2D(filters=filters,
                      kernel_size=kernel_size,
                      strides=strides,
                      padding=padding)(inputs)
    return layers.ReLU()(x)


def dropout_layer(inputs, training):
    """ P.6
        At test time, we use all the neurons but multiply their outputs by 0.5,
        which is a reasonable approximation to taking the geometric mean of the predictive distribution
        produced by the exponentially-many dropout networks.

        We use dropout in the first two fully-connected layers.
        Without dropout, our network exhibits substantial overfitting.
        Dropout roughly doubles the number of iterations required to coverage.
    """

    # (?, 4096)
    assert len(inputs.shape) == 1, 'only valid for 1D tensor.'
    if training:
        x = layers.Dropout(rate=0.5)(inputs)
    else:
        x = layers.Lambda(lambda t: t * 0.5)(inputs)

    return x


def AlexNet(input_shape, num_classes, training):
    inputs = Input(shape=input_shape)

    # AlexNet consists of 5 Convolutions layers and 3 Fully Connected layers
    conv_1 = convolution_layer(inputs, 96, (11, 11), 4, 'valid')
    conv_1 = layers.MaxPooling2D(pool_size=(3, 3), strides=2)(conv_1)

    conv_2 = convolution_layer(conv_1, 256, (5, 5), 1, 'same')
    conv_2 = layers.MaxPooling2D(pool_size=(3, 3), strides=2)(conv_2)

    conv_3 = convolution_layer(conv_2, 384, (3, 3), 1, 'same')

    conv_4 = convolution_layer(conv_3, 384, (3, 3), 1, 'same')

    conv_5 = convolution_layer(conv_4, 256, (3, 3), 1, 'same')
    conv_5 = layers.MaxPooling2D(pool_size=(3, 3), strides=2)(conv_5)
    conv_5 = layers.Flatten()(conv_5)

    fc_1 = layers.Dense(units=4096)(conv_5)
    fc_1 = dropout_layer(fc_1, training)

    fc_2 = layers.Dense(units=4096)(fc_1)
    fc_2 = dropout_layer(fc_2, training)

    outputs = layers.Dense(units=num_classes, activation='softmax')(fc_2)

    model = Model(inputs, outputs)

    return model


if __name__ == '__main__':
    input_shape = (227, 227, 3)
    num_classes = 1000
    training = True
    model = AlexNet(input_shape, num_classes, True)
    model.summary()
