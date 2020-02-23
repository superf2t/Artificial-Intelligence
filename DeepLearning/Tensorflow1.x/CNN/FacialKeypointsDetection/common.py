import tensorflow as tf
import functools
import numpy as np

WeightsFile = 'Data/weights.hdf5'
IdLookupTableFile = 'Data/IdLookupTable.csv'
TestFile = 'Data/test.csv'
TrainingFile = 'Data/training.csv'

def process_img(X, y=None):
    imgs = [np.array(i.split(' '), dtype=np.float32).reshape(96, 96, 1) for i in X]
    imgs = [img / 255.0 for img in imgs]
    return np.array(imgs), y

# define model
Activation = 'elu'
Input = tf.keras.layers.Input
Conv2d = functools.partial(
    tf.keras.layers.Conv2D,
    activation=Activation,
    padding='same'
)
BatchNormalization = tf.keras.layers.BatchNormalization
AveragePooling2D = tf.keras.layers.AveragePooling2D
MaxPooling2D = tf.keras.layers.MaxPool2D
Dense = functools.partial(
    tf.keras.layers.Dense,
    activation=Activation
)
Flatten = tf.keras.layers.Flatten


def prepare_model():
    input = Input(shape=(96, 96, 1,))
    conv_1 = Conv2d(16, (2, 2))(input)
    batch_norm_1 = BatchNormalization()(conv_1)

    conv_2 = Conv2d(32, (3, 3))(batch_norm_1)
    batch_norm_2 = BatchNormalization()(conv_2)

    conv_3 = Conv2d(64, (4, 4))(batch_norm_2)
    avg_pool_1 = AveragePooling2D((2, 2))(conv_3)
    batch_norm_3 = BatchNormalization()(avg_pool_1)

    conv_128 = Conv2d(128, (4, 4))(batch_norm_2)
    avg_pool_128 = AveragePooling2D((2, 2))(conv_3)
    batch_norm_128 = BatchNormalization()(avg_pool_1)

    conv_4 = Conv2d(64, (7, 7))(batch_norm_128)
    avg_pool_1 = AveragePooling2D((2, 2))(conv_128)
    batch_norm_4 = BatchNormalization()(avg_pool_128)

    conv_5 = Conv2d(32, (7, 7))(batch_norm_4)
    flat_1 = Flatten()(conv_5)

    dense_1 = Dense(30)(flat_1)
    outputs = Dense(30)(dense_1)

    model = tf.keras.Model(input, dense_1)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    return model