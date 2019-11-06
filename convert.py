import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_keras_model_file('save/path/model.hdf5')
tf_lite_model = converter.convert()

open('tflite_mobilenetv2.tflite', 'wb').write(tf_lite_model)
