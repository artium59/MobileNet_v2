# -*- coding: utf-8 -*-
"""
Going Deeper with Convolutions (2014)
Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich

https://arxiv.org/pdf/1409.4842v1.pdf
https://ai.google/research/pubs/pub43022
"""

from tensorflow.python.keras import Input, Model, layers


def inception_layer(inputs, one, three_reduce, three, five_reduce, five, pool, name):
    """
    # Arguments
        inputs: Tensor
        one: Integer, #1×1 projection layer filters number
        three_reduce: Integer, #3×3 reduce layer filters number
        three: Integer, #3×3 conv layer filters number
        five_reduce: Integer, #5×5 reduce layer filters number
        five: Integer, #5×5 conv layer filters number
        pool: Integer, pool projection filter number
        name: String, Distinguish the Inception layer
    # Returns
        Inception Tensor
    """

    conv_1 = layers.Conv2D(filters=one, kernel_size=(1, 1),
                           strides=1, padding='same',
                           activation='relu',
                           name='{}_proj'.format(name))(inputs)

    conv_2 = layers.Conv2D(filters=three_reduce, kernel_size=(1, 1),
                           strides=1, padding='same',
                           activation='relu',
                           name='{}_conv_3x3reduce'.format(name))(inputs)
    conv_2 = layers.Conv2D(filters=three, kernel_size=(3, 3),
                           strides=1, padding='same',
                           activation='relu',
                           name='{}_conv_3x3'.format(name))(conv_2)

    conv_3 = layers.Conv2D(filters=five_reduce, kernel_size=(1, 1),
                           strides=1, padding='same',
                           activation='relu',
                           name='{}_conv_5x5reduce'.format(name))(inputs)
    conv_3 = layers.Conv2D(filters=five, kernel_size=(5, 5),
                           strides=1, padding='same',
                           activation='relu',
                           name='{}_conv_5x5'.format(name))(conv_3)

    conv_4 = layers.MaxPooling2D(pool_size=(3, 3),
                                 strides=1, padding='same',
                                 name='{}_maxpool'.format(name))(inputs)
    conv_4 = layers.Conv2D(filters=pool, kernel_size=(1, 1),
                           strides=1, padding='same',
                           activation='relu',
                           name='{}_maxpool_proj'.format(name))(conv_4)

    return layers.Concatenate(name='{}'.format(name))([conv_1, conv_2, conv_3, conv_4])


def GoogLeNet(input_shape=(224, 224, 3), num_classes=1000, name='Inception_v1'):
    """ P.6
    All the convolutions, including those inside the Inception modules, use rectified linear activation.
    The size of the receptive field in our network is 224×224 taking RGB color channels with mean sub-traction.
    “#3×3reduce” and “#5×5reduce” stands for the number of 1×1 filters in the reduction layer used before the 3×3 and 5×5 convolutions.
    One can see the number of 1×1 filters in the projection layer after the built-in max-pooling in the pool projection column.
    All these reduction/projection layers use rectified linear activation as well.
    """

    inputs = Input(shape=input_shape)

    # 원래 lRN 해야 하는데, BN으로 대체
    x = layers.Conv2D(filters=64, kernel_size=(7, 7),
                      strides=2, padding='same',
                      activation='relu',
                      name='conv_7x7_1')(inputs)
    x = layers.MaxPooling2D(pool_size=(3, 3),
                            strides=2, padding='same',
                            name='maxpool_3x3_1')(x)
    x = layers.BatchNormalization(name='bn_1')(x)

    x = layers.Conv2D(filters=64, kernel_size=(1, 1),
                      strides=1, padding='valid',
                      activation='relu',
                      name='reduce_3x3_1')(x)
    x = layers.Conv2D(filters=192, kernel_size=(3, 3),
                      strides=1, padding='same',
                      activation='relu',
                      name='conv_3x3_1')(x)
    x = layers.BatchNormalization(name='bn_2')(x)
    x = layers.MaxPooling2D(pool_size=(3, 3),
                            strides=2, padding='same',
                            name='maxpool_3x3_2')(x)

    # Inception 3X
    x = inception_layer(x, 64, 96, 128, 16, 32, 32, 'inception3a')
    x = inception_layer(x, 128, 128, 192, 32, 96, 64, 'inception3b')
    x = layers.MaxPooling2D(pool_size=(3, 3),
                            strides=2, padding='same',
                            name='maxpool_3x3_3')(x)

    # Inception 4X
    x = inception_layer(x, 192,  96, 208, 16,  48,  64, 'inception4a')
    x = inception_layer(x, 160, 112, 224, 24,  64,  64, 'inception4b') # softmax0
    x = inception_layer(x, 128, 128, 256, 24,  64,  64, 'inception4c')
    x = inception_layer(x, 112, 144, 288, 32,  64,  64, 'inception4d')
    x = inception_layer(x, 256, 160, 320, 32, 128, 128, 'inception4e') # softmax1
    x = layers.MaxPooling2D(pool_size=(3, 3),
                            strides=2, padding='same',
                            name='maxpool_3x3_4')(x)

    # Inception 5X
    x = inception_layer(x, 256, 160, 320, 32, 128, 128, 'inception5a')
    x = inception_layer(x, 384, 192, 384, 48, 128, 128, 'inception5b')

    x = layers.AveragePooling2D(pool_size=(7, 7),
                                strides=1, padding='valid',
                                name='avgpool_7x7')(x)
    x = layers.Conv2D(filters=num_classes, kernel_size=(1, 1),
                      activation='softmax', name='fc_softmax')(x)

    model = Model(inputs, x, name=name)

    return model


if __name__ == '__main__':
    input_shape = (224, 224, 3)
    num_classes = 1000

    model = GoogLeNet(input_shape, num_classes)
    model.summary()
