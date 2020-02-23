import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import functools
import os
import datetime
from sklearn.model_selection import train_test_split
import common

train_df = pd.read_csv(common.TrainingFile)
print('num of cols {}', format(len(train_df.columns)))

# show first image and result
sample_img = np.array(train_df['Image'][0].split(' '), dtype=np.float32).reshape(96, 96)
print(sample_img)
y = train_df.drop('Image', axis=1)
t = y.iloc[0].values
plt.imshow(sample_img, cmap='gray')
# t[0::2] means：from 0 to end， interval 2
plt.scatter(t[0::2], t[1::2], c='red', marker='x')
plt.show()

# preprocess raw data
rownull = train_df.isnull().any(axis=0)
print('Before processing: \n', rownull.value_counts())
train_df.fillna(method='ffill', inplace=True)
rownull = train_df.isnull().any(axis=0)
print('After processing: \n', rownull.value_counts())

# training data and labels
training_data = train_df['Image'].values
labels = train_df.drop('Image', axis=1)
FEATURES = list(labels.columns)
print(FEATURES)
X_train, X_test, y_train, y_test = train_test_split(training_data, labels, random_state=42, test_size=0.1)


# data pipeline
BATCH_SIZE = 256
def data_pipeline(X, y, shuffle_size):
    dataset = (
        tf.data.Dataset.from_tensor_slices((X, y))
            .shuffle(shuffle_size)
            .batch(BATCH_SIZE)
            .prefetch(1)
            .repeat()
    )
    #     print('Dataset element spec {}'.format(dataset.element_spec))
    iterator = dataset.make_one_shot_iterator()
    return iterator


train_shuffle_size = len(X_train)
test_shuffle_size = len(X_test)
X_train, y_train = common.process_img(X_train, y_train.values)
X_test, y_test = common.process_img(X_test, y_test.values)

train_iterator = data_pipeline(X_train, y_train, train_shuffle_size)
validation_iterator = data_pipeline(X_test, y_test, test_shuffle_size)

model = common.prepare_model()
model.summary()
logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(filepath='Data/weights.hdf5', verbose=1, save_best_only=True),
    tensorboard_callback,
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
]

model.fit(train_iterator, steps_per_epoch=24, epochs=50, validation_data=validation_iterator, validation_steps=6,
          callbacks=callbacks)

