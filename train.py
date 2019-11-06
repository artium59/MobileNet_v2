import csv
import numpy as np

from tensorflow import keras
from MobileNet_v2 import *


batch_size = 64
num_epochs = 500
# num_epochs = 10
input_shape = (48, 48, 1)
# EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
num_classes = 7
verbose = 1
patience = 30

trained_models_path = 'train/model/path/'


def preprocess_input(pixels):
    picture = []

    for i in pixels.split():
        picture.append(int(i))

    return np.array(picture).reshape(48, 48, 1)


def load_fer2013():
    train_faces = []
    val_faces = []
    test_faces = []
    train_emotions = []
    val_emotions = []
    test_emotions = []

    with open('data/fer2013.csv') as csvDataFile:
        csv_reader = csv.DictReader(csvDataFile)
        for row in csv_reader:
            if row['Usage'] == 'Training':
                train_emotions.append(int(row['emotion']))
                train_faces.append(preprocess_input(row['pixels']))
            elif row['Usage'] == 'PublicTest':
                val_emotions.append(int(row['emotion']))
                val_faces.append(preprocess_input(row['pixels']))
            else:
                test_emotions.append(int(row['emotion']))
                test_faces.append(preprocess_input(row['pixels']))

    return np.array(train_faces), np.array(val_faces), np.array(test_faces), train_emotions, val_emotions, test_emotions


# data generator
data_generator = keras.preprocessing.image.ImageDataGenerator(
                            featurewise_center=True,
                            # featurewise_std_normalization=True,
                            rotation_range=10,
                            width_shift_range=0.1,
                            height_shift_range=0.1,
                            # rescale=1./255,
                            zoom_range=.1,
                            horizontal_flip=True,
                            fill_mode='nearest')

model = MobileNet_v2(input_shape, num_classes)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
early_stop = keras.callbacks.EarlyStopping('val_loss', patience=30)
reduce_lr = keras.callbacks.ReduceLROnPlateau('val_loss', factor=0.1, patience=10, verbose=1)

model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'

model_checkpoint = keras.callbacks.ModelCheckpoint(model_names, 'val_loss', verbose=1, save_best_only=True)
# callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr, logs]
# callbacks = [csv_logger, early_stop, reduce_lr, logs]
callbacks = [early_stop, reduce_lr]

# faces = preprocess_input(faces)
# faces = preprocess_input_0(faces)
# num_samples, num_classes = emotions.shape
x_train, x_val, x_test, y_train, y_val, y_test = load_fer2013()

y_train = np.array(keras.utils.to_categorical(y_train, num_classes))
y_val = np.array(keras.utils.to_categorical(y_val, num_classes))
y_test = np.array(keras.utils.to_categorical(y_test, num_classes))

model.fit_generator(data_generator.flow(x_train, y_train, batch_size),
                    steps_per_epoch=int(len(x_train) / batch_size),
                    epochs=num_epochs, verbose=1, callbacks=callbacks,
                    validation_data=(x_val, y_val))

test_score = model.evaluate(x_test, y_test)
print('Test Score is {}'.format(test_score[0]))
print('Test Accuracy is {}'.format(test_score[1]))

Model_names = trained_models_path + 'mobilenetv2-' + '{0:.4f}'.format(test_score[1])+'.hdf5'
model.save(Model_names)
# keras.models.save_model(model, trained_models_path)
