# Import the required libraries
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import activations
from tensorflow.keras.activations import softmax
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.regularizers import l2
import random
import sklearn.metrics
import numpy as np
import pandas as pd
import os
import librosa
import math
import datetime
from pathlib import Path
from tensorflow.keras.layers import (
    Activation,
    Add,
    AveragePooling2D,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    MaxPool2D,
    UpSampling2D,
    Flatten
)


# Setting seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

df = pd.read_csv('./data_infant_cry.csv')
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # For shuffling

# Set the parameters for audio processing
class conf:
    sampling_rate = 16000
    duration = 1  # All files in the infant cry database have a duration of 1 sec
    samples = int(sampling_rate * duration)
    n_mfcc = 36

# Take random samples from the audio or pad if smaller than given number of samples
def read_audio(conf, pathname, trim_long_data):
    y, sr = librosa.load(pathname, sr=conf.sampling_rate)
    # If audio length is greater than given number of samples, trim it
    if len(y) > conf.samples:
        if trim_long_data:
            ind = np.random.randint(0, len(y)-conf.samples+1)
            y = y[ind:ind+conf.samples]
    # If audio length is smaller, pad it
    else:
        padding = conf.samples - len(y)
        offset = padding // 2
        # Add padding on both sides
        y = np.pad(y, (math.ceil(offset), math.ceil(
            conf.samples - len(y) - offset)), 'constant')
    return y

# Convert the raw audio into MFCCs
def convert_audio(conf, audio):
    mfcc = librosa.feature.mfcc(
        audio, sr=conf.sampling_rate, n_mfcc=conf.n_mfcc)
    mfcc = mfcc.astype(np.float32)
    return mfcc

# This implements the previous two functions for one file
def read_file(conf, pathname, trim_long_data, debug_display=False):
    x = read_audio(conf, pathname, trim_long_data)
    mfcc = convert_audio(conf, x)
    return mfcc

# This generalizes for all the audio files
def convert_wav_to_mfcc(df, source, folders):
    X_train, X_valid, X_test, Y_train, Y_valid, Y_test = [], [], [], [], [], []
    for i in df.index:
        if(df.at[i, 'folder'] == folders[0]):
            x = read_file(conf, source[0] + '/' +
                          str(df.at[i, 'files']), trim_long_data=True)
            X_train.append(x.transpose())
            Y_train.append(df.at[i, 'classes'])
        elif(df.at[i, 'folder'] == folders[1]):
            x = read_file(conf, source[1] + '/' +
                          str(df.at[i, 'files']), trim_long_data=True)
            X_valid.append(x.transpose())
            Y_valid.append(df.at[i, 'classes'])
        elif(df.at[i, 'folder'] == folders[2]):
            x = read_file(conf, source[2] + '/' +
                          str(df.at[i, 'files']), trim_long_data=True)
            X_test.append(x.transpose())
            Y_test.append(df.at[i, 'classes'])
    return X_train, X_valid, X_test, Y_train, Y_valid, Y_test

train_path = './infant_cry/train'
valid_path = './infant_cry/valid'
test_path = './infant_cry/test'

folders = ['train', 'valid', 'test']
source = [train_path, valid_path, test_path]

# We get all our MFCCs and the labels
X_train, X_valid, X_test, Y_train, Y_valid, Y_test = convert_wav_to_mfcc(df, source, folders)

# Convert into numpy arrays
X_train = np.array(X_train)
X_valid = np.array(X_valid)
X_test = np.array(X_test)
Y_train = np.array(Y_train)
Y_valid = np.array(Y_valid)
Y_test = np.array(Y_test)

# Adding the channel axis to the data
X_train = tf.expand_dims(X_train, axis=-1)
X_valid = tf.expand_dims(X_valid, axis=-1)
X_test = tf.expand_dims(X_test, axis=-1)


# Source code for model - https://towardsdatascience.com/understand-and-implement-resnet-50-with-tensorflow-2-0-1190b9b52691

def res_identity(x, filters):
    # resnet block where dimension does not change.
    # The skip connection is just simple identity connection
    # we will have 3 blocks and then input will be added

    x_skip = x  # this will be used for addition with the residual block
    f1, f2 = filters

    # first block
    x = Conv2D(f1, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)

    # second block # bottleneck (but size kept same with padding)
    x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)

    # third block activation used after adding the input
    x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)

    # add the input
    x = Add()([x, x_skip])
    x = Activation(activations.relu)(x)

    return x


def res_conv(x, s, filters):
    '''
    here the input size changes'''
    x_skip = x
    f1, f2 = filters

    # first block
    x = Conv2D(f1, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=l2(0.001))(x)
    # when s = 2 then it is like downsizing the feature map
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)

    # second block
    x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)

    # third block
    x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)

    # shortcut
    x_skip = Conv2D(f2, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=l2(0.001))(x_skip)
    x_skip = BatchNormalization()(x_skip)

    # add
    x = Add()([x, x_skip])
    x = Activation(activations.relu)(x)

    return x


def resnet50():

    input_audio = Input(shape=(32, conf.n_mfcc, 1))

    # 1st stage
    # here we perform maxpooling

    x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2))(input_audio)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)
    x = MaxPool2D((3, 3), strides=(2, 2))(x)

    # 2nd stage
    # frm here on only conv block and identity block, no pooling

    x = res_conv(x, s=1, filters=(64, 256))
    x = res_identity(x, filters=(64, 256))
    x = res_identity(x, filters=(64, 256))

    # 3rd stage

    x = res_conv(x, s=2, filters=(128, 512))
    x = res_identity(x, filters=(128, 512))
    x = res_identity(x, filters=(128, 512))
    x = res_identity(x, filters=(128, 512))

    # 4th stage

    x = res_conv(x, s=2, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))

    # 5th stage

    x = res_conv(x, s=2, filters=(512, 2048))
    x = res_identity(x, filters=(512, 2048))
    x = res_identity(x, filters=(512, 2048))

    # ends with average pooling and dense connection

    x = AveragePooling2D((2, 2), padding='same')(x)

    x = Flatten()(x)
    x = Dense(1, activation='sigmoid', kernel_initializer='he_normal')(x)

    # define the model

    model = models.Model(inputs=input_audio, outputs=x, name='Resnet50')

    return model


# Defining the hyperparameters
EPOCHS = 500
learning_rate = 1e-5

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

log_dir = "./infant_cry/logs/" + datetime.datetime.now().strftime("%d%m%Y-%H%M%S")
# Saving the logs for the trained model
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Early stopping based on validation loss with a patience of 5
early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5,
                                                       verbose=1, mode='auto', baseline=None, restore_best_weights=True)

checkpoint_path = './infant_cry/saved_models/' + datetime.datetime.now().strftime("%d%m%Y-%H%M%S")

# Saving the model with the least validation loss
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss',
                                                               verbose=1, save_best_only=True, mode='min', save_weights_only=True)

model = resnet50()

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', 'AUC', 'Precision', 'Recall'])

history = model.fit(X_train, Y_train, batch_size=32, epochs=EPOCHS, validation_data=(X_valid, Y_valid),
                    callbacks=[early_stop_callback, tensorboard_callback, model_checkpoint_callback])

# Load the weights from the checkpoint path for the best model and find the performance for the test data
model.load_weights(checkpoint_path)
loss, acc, auc, precision, recall = model.evaluate(X_test, Y_test, batch_size=32, verbose=1)
