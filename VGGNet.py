# -*- coding: utf-8 -*-
"""
Very Deep Convolutional Networks for Large-Scale Image Recognition
Karen Simonyan, Andrew Zisserman, 2015

https://arxiv.org/pdf/1409.1556.pdf
http://www.robots.ox.ac.uk/~vgg/research/very_deep/
"""

import numpy as np
from tensorflow.python.keras import Input, Model, layers, utils
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.applications.imagenet_utils import preprocess_input, decode_predictions


def convolution_layer(inputs, filters, kernel_size, is_padding, number):
    """
    # Arguments
        inputs: Tensor, input tensor of convolution layer.
        filters: Integer, number of channels
        kernel_size: Integer or Tuple, 3x3, 1x1
            3×3 : which is the smallest size to capture the notion of left/right, up/down, center.
            1×1 : which can be seen as a linear transformation of the input channels (followed by non-linearity).
                  (Only ConvNet configuration C uses)
        is_padding: Bool, True if kernel_size == 3x3
            Spatial padding of convolution layer input is such that the spatial resolution is preserved after convolution
        number: Integer, count the convolution layer
    # Returns
        Output tensor.
    """
    x = layers.Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=1, padding='same' if is_padding else 'valid',
                      name='conv{}-{}_{}'.format(kernel_size, filters, number))(inputs)
    return layers.ReLU()(x)


def VGGNet(input_shape, num_classes, name, is_vgg16):
    """
    This is ConvNet configuration D.
    The convolution layer parameters are denoted as "conv{receptive field size}-{number of channels}_{number}"

    # Arguments
        input_shape: Tuple, ImageNet dataset size
        num_classes: Integer, ImageNet classes
        name: String, model's name
    # Returns
        VGG16 model
    """

    inputs = Input(shape=input_shape)

    # Block 1
    x = convolution_layer(inputs, 64, 3, True, 1)
    x = convolution_layer(x, 64, 3, True, 2)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=2, name='block1_maxpool')(x)

    # Block 2
    x = convolution_layer(x, 128, 3, True, 1)
    x = convolution_layer(x, 128, 3, True, 2)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=2, name='block2_maxpool')(x)

    # Block 3
    x = convolution_layer(x, 256, 3, True, 1)
    x = convolution_layer(x, 256, 3, True, 2)
    x = convolution_layer(x, 256, 3, True, 3)
    if not is_vgg16:
        x = convolution_layer(x, 256, 3, True, 4)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=2, name='block3_maxpool')(x)

    # Block 4
    x = convolution_layer(x, 512, 3, True, 1)
    x = convolution_layer(x, 512, 3, True, 2)
    x = convolution_layer(x, 512, 3, True, 3)
    if not is_vgg16:
        x = convolution_layer(x, 512, 3, True, 4)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=2, name='block4_maxpool')(x)

    # Block 5
    x = convolution_layer(x, 512, 3, True, 5)
    x = convolution_layer(x, 512, 3, True, 6)
    x = convolution_layer(x, 512, 3, True, 7)
    if not is_vgg16:
        x = convolution_layer(x, 512, 3, True, 8)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=2, name='block5_maxpool')(x)

    x = layers.Flatten()(x)
    x = layers.Dense(units=4096, activation='relu', name='FC-4096_1')(x)
    x = layers.Dense(units=4096, activation='relu', name='FC-4096_2')(x)
    x = layers.Dense(units=num_classes, activation='softmax', name='FC-{}'.format(num_classes))(x)

    # Create model
    model = Model(inputs, x, name=name)

    # Load weights
    weights = 16 if is_vgg16 else 19
    WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases' \
                   '/download/v0.1/vgg{}_weights_tf_dim_ordering_tf_kernels.h5'.format(weights)

    weights_path = utils.get_file('vgg{}_weights_tf_dim_ordering_tf_kernels.h5'.format(weights),
                                  WEIGHTS_PATH,
                                  cache_dir='models')
    model.load_weights(weights_path)

    return model


if __name__ == '__main__':
    input_shape = (224, 224, 3)
    num_classes = 1000
    is_vgg16 = True
    name = 'VGG16' if is_vgg16 else 'VGG19'

    # Number of Parameters
    # VGG16: 138M, VGG19: 144M
    model = VGGNet(input_shape, num_classes, name=name, is_vgg16=is_vgg16)
    model.summary()

    img_path = 'cat.jpg'
    img = image.load_img(img_path, target_size=input_shape)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    print('Input image tensor: {}'.format(img_array.shape))

    assert len(img_array.shape) == 4, 'x.shape is (1, 224, 224, 3)'
    pred = model.predict(img_array)
    pred = np.array(decode_predictions(pred)).reshape(-1, 3)
    print(pred)
