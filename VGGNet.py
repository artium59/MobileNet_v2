# -*- coding: utf-8 -*-

"""
Very Deep Convolutional Networks for Large-Scale Image Recognition
Karen Simonyan, Andrew Zisserman, 2015

https://arxiv.org/pdf/1409.1556.pdf
http://www.robots.ox.ac.uk/~vgg/research/very_deep/
"""

from tensorflow.python.keras import Input, Model, layers, utils

input_shape = (224, 224, 3)
num_classes = 1000
WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases' \
               '/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5 '


def preprocess_input(inputs):
    pass


def convolution_layer(inputs, filters, kernel_size, is_padding, number):
    """
    # Arguments
        inputs: Tensor, input tensor of convolution layer.
        filters: Integer, number of channels
        kernel_size: Integer or Tuple, 3x3, 1x1
            3×3 : which is the smallest size to capture the notion of left/right, up/down, center.
            1×1 : which can be seen as a linear transformation of the input channels (followed by non-linearity).
                  (Only ConvNet configuration C uses)
        is_padding: Bool, true if kernel_size == 3x3
            Spatial padding of convolution layer input is such that the spatial resolution is preserved after convolution
        number: Integer, count the convolution layer
    # Returns
        Output tensor.
    """
    x = layers.Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=1, padding='same' if is_padding else 'valid',
                      name='conv{}-{}_{}'.format(kernel_size, filters, number))(inputs)
    return layers.ReLU()(x)


def VGGNet_16_layers(input_shape, num_classes, name='VGG16'):
    """
    This is ConvNet configuration D
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
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=2, name='block3_maxpool')(x)

    # Block 4
    x = convolution_layer(x, 512, 3, True, 1)
    x = convolution_layer(x, 512, 3, True, 2)
    x = convolution_layer(x, 512, 3, True, 3)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=2, name='block4_maxpool')(x)

    # Block 5
    x = convolution_layer(x, 512, 3, True, 4)
    x = convolution_layer(x, 512, 3, True, 5)
    x = convolution_layer(x, 512, 3, True, 6)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=2, name='block5_maxpool')(x)

    x = layers.Flatten()(x)
    x = layers.Dense(units=4096, activation='relu', name='FC-4096_1')(x)
    x = layers.Dense(units=4096, activation='relu', name='FC-4096_2')(x)
    x = layers.Dense(units=num_classes, activation='softmax', name='FC-1000')(x)

    # Create model
    model = Model(inputs, x, name=name)

    # Load weights
    weights_path = utils.get_file('vgg16_weights_tf_dim_ordering_tf_kernels.h5',
                                  WEIGHTS_PATH,
                                  cache_dir='models')
    model.load_weights(weights_path)

    return model


def VGGNet_19_layers(input_shape, num_classes, weights_file_path):
    inputs = Input(shape=input_shape)
    pass


def train(vgg_model):
    pass


if __name__ == '__main__':
    model = VGGNet_16_layers(input_shape, num_classes)
    model.summary()
