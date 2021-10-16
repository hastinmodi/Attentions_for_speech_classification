# File for testing model on different length of audio


# Import the required libraries
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.activations import softmax
from tensorflow.keras.activations import sigmoid
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
)


# Setting seeds for reproducibility
random.seed(42)
tf.random.set_seed(42)

df = pd.read_csv('./data_infant_cry.csv')
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # For shuffling

# Set the parameters for audio processing


class conf:
    sampling_rate = 16000
    duration = 1  # All files in the infant cry database have a duration of 1 sec
    samples = int(sampling_rate * duration)
    n_mfcc = 36
    # The length of audio to be kept
    del_frames = [0.5, 0.6, 0.7, 0.8, 0.9]


# Take random samples from the audio or pad if smaller than given number of samples
def read_audio(conf, pathname, trim_long_data):
    y, sr = librosa.load(pathname, sr=conf.sampling_rate)
    # If audio length is greater than given number of samples, trim it
    if len(y) > conf.samples:
        if trim_long_data:
            ind = np.random.randint(0, len(y) - conf.samples + 1)
            y = y[ind:ind+conf.samples]
    # If audio length is smaller, pad it
    else:
        padding = conf.samples - len(y)
        offset = padding // 2
        # Add padding on both sides
        y = np.pad(y, (math.ceil(offset), math.ceil(
            conf.samples - len(y) - offset)), 'constant')
    return y

# We convert our raw audio into MFCCs


def convert_audio(conf, audio):
    mfcc = librosa.feature.mfcc(
        audio, sr=conf.sampling_rate, n_mfcc=conf.n_mfcc)
    mfcc = mfcc.astype(np.float32)
    return mfcc

# This implements the previous two functions for one file


def read_file(conf, pathname, trim_long_data, debug_display=False):
    x = read_audio(conf, pathname, trim_long_data)
    mfcc = convert_audio(conf, x)
    n_rows, n_cols = mfcc.shape
    ind = int(conf.del_frames[4]*n_cols)
    # Cutting frames from the audio
    mfcc = np.delete(mfcc, np.s_[ind:n_cols], axis=1)
    return mfcc

# This generalizes for all the audio files


def convert_wav_to_mfcc(df, source, folders):
    X_test, Y_test = [], []
    for i in df.index:
        if(df.at[i, 'folder'] == folders[0]):
            x = read_file(conf, source[0] + '/' +
                          str(df.at[i, 'files']), trim_long_data=True)
            X_test.append(x.transpose())
            Y_test.append(df.at[i, 'classes'])
    return X_test, Y_test


test_path = './infant_cry/test'

folders = ['test']
source = [test_path]

# We get all our MFCCs and labels
X_test, Y_test = convert_wav_to_mfcc(df, source, folders)

# Convert into numpy arrays
X_test = np.array(X_test)
Y_test = np.array(Y_test)

# Adding the channel axis to the data
X_test = tf.expand_dims(X_test, axis=-1)

# Adding padding to the input data since the number of frames are reduced
# 3 frames are removed in 10% of audio
paddings = tf.constant([[0, 0], [2, 1], [0, 0], [0, 0]])
X_test = tf.pad(X_test, paddings, "SYMMETRIC")

# Start building the model


class GroupedConv2D(object):
    """Groupped convolution.
    https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mixnet/custom_py
    Currently tf.keras and tf.layers don't support group convolution, so here we
    use split/concat to implement this op. It reuses kernel_size for group
    definition, where len(kernel_size) is number of groups. Notably, it allows
    different group has different kernel size.
    """

    def __init__(self, filters, kernel_size, use_keras=True, **kwargs):
        """Initialize the layer.
        Args:
        filters: Integer, the dimensionality of the output space.
        kernel_size: An integer or a list. If it is a single integer, then it is
            same as the original Conv2D. If it is a list, then we split the channels
            and perform different kernel for each group.
        use_keras: An boolean value, whether to use keras layer.
        **kwargs: other parameters passed to the original conv2d layer.
        """
        self._groups = len(kernel_size)
        self._channel_axis = -1

        self._convs = []
        splits = self._split_channels(filters, self._groups)
        for i in range(self._groups):
            self._convs.append(self._get_conv2d(
                splits[i], kernel_size[i], use_keras, **kwargs))

    def _get_conv2d(self, filters, kernel_size, use_keras, **kwargs):
        """A helper function to create Conv2D layer."""
        if use_keras:
            return Conv2D(filters=filters, kernel_size=kernel_size, **kwargs)
        else:
            return Conv2D(filters=filters, kernel_size=kernel_size, **kwargs)

    def _split_channels(self, total_filters, num_groups):
        split = [total_filters // num_groups for _ in range(num_groups)]
        split[0] += total_filters - sum(split)
        return split

    def __call__(self, inputs):
        if len(self._convs) == 1:
            return self._convs[0](inputs)

        if tf.__version__ < "2.0.0":
            filters = inputs.shape[self._channel_axis].value
        else:
            filters = inputs.shape[self._channel_axis]
        splits = self._split_channels(filters, len(self._convs))
        x_splits = tf.split(inputs, splits, self._channel_axis)
        x_outputs = [c(x) for x, c in zip(x_splits, self._convs)]
        x = tf.concat(x_outputs, self._channel_axis)
        return x


class ResNest:
    def __init__(self, verbose=True, input_shape=(32, 36, 1), active="relu", n_classes=2,
                 dropout_rate=0.2, fc_activation=None, blocks_set=[3, 4, 6, 3], radix=2, groups=1,
                 bottleneck_width=64, deep_stem=True, stem_width=32, block_expansion=4, avg_down=True,
                 avd=True, avd_first=False, preact=False, using_basic_block=False, using_cb=False):
        self.channel_axis = -1  # not for change
        self.verbose = verbose
        self.active = active  # default relu
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.dropout_rate = dropout_rate
        self.fc_activation = fc_activation

        self.blocks_set = blocks_set
        self.radix = radix
        self.cardinality = groups
        self.bottleneck_width = bottleneck_width

        self.deep_stem = deep_stem
        self.stem_width = stem_width
        self.block_expansion = block_expansion
        self.avg_down = avg_down
        self.avd = avd
        self.avd_first = avd_first

        self.dilation = 1
        self.preact = preact
        self.using_basic_block = using_basic_block
        self.using_cb = using_cb

    def _make_stem(self, input_tensor, stem_width=64, deep_stem=False):
        x = input_tensor
        if deep_stem:
            x = Conv2D(stem_width, kernel_size=3, strides=2, padding="same", kernel_initializer="he_normal",
                       use_bias=False, data_format="channels_last")(x)

            x = BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
            x = Activation(self.active)(x)

            x = Conv2D(stem_width, kernel_size=3, strides=1, padding="same",
                       kernel_initializer="he_normal", use_bias=False, data_format="channels_last")(x)

            x = BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
            x = Activation(self.active)(x)

            x = Conv2D(stem_width * 2, kernel_size=3, strides=1, padding="same", kernel_initializer="he_normal",
                       use_bias=False, data_format="channels_last")(x)

        else:
            x = Conv2D(stem_width, kernel_size=7, strides=2, padding="same", kernel_initializer="he_normal",
                       use_bias=False, data_format="channels_last")(x)
        return x

    def _rsoftmax(self, input_tensor, filters, radix, groups):
        x = input_tensor
        batch = x.shape[0]
        if radix > 1:
            x = tf.reshape(x, [-1, groups, radix, filters // groups])
            x = tf.transpose(x, [0, 2, 1, 3])
            x = tf.keras.activations.softmax(x, axis=1)
            x = tf.reshape(x, [-1, 1, 1, radix * filters])
        else:
            x = Activation("sigmoid")(x)
        return x

    def _SplAtConv2d(self, input_tensor, filters=64, kernel_size=3, stride=1, dilation=1, groups=1, radix=0):
        x = input_tensor
        in_channels = input_tensor.shape[-1]

        x = GroupedConv2D(filters=filters * radix, kernel_size=[kernel_size for i in range(groups * radix)],
                          use_keras=True, padding="same", kernel_initializer="he_normal", use_bias=False,
                          data_format="channels_last", dilation_rate=dilation)(x)

        x = BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
        x = Activation(self.active)(x)

        batch, rchannel = x.shape[0], x.shape[-1]
        if radix > 1:
            splited = tf.split(x, radix, axis=-1)
            gap = sum(splited)
        else:
            gap = x

        gap = GlobalAveragePooling2D(data_format="channels_last")(gap)
        gap = tf.reshape(gap, [-1, 1, 1, filters])

        reduction_factor = 4
        inter_channels = max(in_channels * radix // reduction_factor, 32)

        x = Conv2D(inter_channels, kernel_size=1)(gap)

        x = BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
        x = Activation(self.active)(x)
        x = Conv2D(filters * radix, kernel_size=1)(x)

        atten = self._rsoftmax(x, filters, radix, groups)

        if radix > 1:
            logits = tf.split(atten, radix, axis=-1)
            out = sum([a * b for a, b in zip(splited, logits)])
        else:
            out = atten * x
        return out

    def _make_block(
        self, input_tensor, first_block=True, filters=64, stride=2, radix=1, avd=False, avd_first=False, is_first=False
    ):
        x = input_tensor
        inplanes = input_tensor.shape[-1]
        if stride != 1 or inplanes != filters * self.block_expansion:
            short_cut = input_tensor
            if self.avg_down:
                if self.dilation == 1:
                    short_cut = AveragePooling2D(pool_size=stride, strides=stride, padding="same", data_format="channels_last")(
                        short_cut
                    )
                else:
                    short_cut = AveragePooling2D(
                        pool_size=1, strides=1, padding="same", data_format="channels_last")(short_cut)
                short_cut = Conv2D(filters * self.block_expansion, kernel_size=1, strides=1, padding="same",
                                   kernel_initializer="he_normal", use_bias=False, data_format="channels_last")(short_cut)
            else:
                short_cut = Conv2D(filters * self.block_expansion, kernel_size=1, strides=stride, padding="same",
                                   kernel_initializer="he_normal", use_bias=False, data_format="channels_last")(short_cut)

            short_cut = BatchNormalization(
                axis=self.channel_axis, epsilon=1.001e-5)(short_cut)
        else:
            short_cut = input_tensor

        group_width = int(
            filters * (self.bottleneck_width / 64.0)) * self.cardinality
        x = Conv2D(group_width, kernel_size=1, strides=1, padding="same", kernel_initializer="he_normal", use_bias=False,
                   data_format="channels_last")(x)
        x = BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
        x = Activation(self.active)(x)

        avd = avd and (stride > 1 or is_first)
        avd_first = avd_first

        if avd:
            avd_layer = AveragePooling2D(
                pool_size=3, strides=stride, padding="same", data_format="channels_last")
            stride = 1

        if avd and avd_first:
            x = avd_layer(x)

        if radix >= 1:
            x = self._SplAtConv2d(x, filters=group_width, kernel_size=3, stride=stride, dilation=self.dilation,
                                  groups=self.cardinality, radix=radix)
        else:
            x = Conv2D(group_width, kernel_size=3, strides=stride, padding="same", kernel_initializer="he_normal",
                       dilation_rate=self.dilation, use_bias=False, data_format="channels_last")(x)
            x = BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
            x = Activation(self.active)(x)

        if avd and not avd_first:
            x = avd_layer(x)

        x = Conv2D(filters * self.block_expansion, kernel_size=1, strides=1, padding="same", kernel_initializer="he_normal",
                   dilation_rate=self.dilation, use_bias=False, data_format="channels_last")(x)
        x = BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)

        m2 = Add()([x, short_cut])
        m2 = Activation(self.active)(m2)
        return m2

    def _make_block_basic(
        self, input_tensor, first_block=True, filters=64, stride=2, radix=1, avd=False, avd_first=False, is_first=False
    ):
        """Conv2d_BN_Relu->Bn_Relu_Conv2d
        """
        x = input_tensor
        x = BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
        x = Activation(self.active)(x)

        short_cut = x
        inplanes = input_tensor.shape[-1]
        if stride != 1 or inplanes != filters * self.block_expansion:
            if self.avg_down:
                if self.dilation == 1:
                    short_cut = AveragePooling2D(pool_size=stride, strides=stride, padding="same", data_format="channels_last")(
                        short_cut
                    )
                else:
                    short_cut = AveragePooling2D(
                        pool_size=1, strides=1, padding="same", data_format="channels_last")(short_cut)
                short_cut = Conv2D(filters, kernel_size=1, strides=1, padding="same", kernel_initializer="he_normal",
                                   use_bias=False, data_format="channels_last")(short_cut)
            else:
                short_cut = Conv2D(filters, kernel_size=1, strides=stride, padding="same", kernel_initializer="he_normal",
                                   use_bias=False, data_format="channels_last")(short_cut)

        group_width = int(
            filters * (self.bottleneck_width / 64.0)) * self.cardinality
        avd = avd and (stride > 1 or is_first)
        avd_first = avd_first

        if avd:
            avd_layer = AveragePooling2D(
                pool_size=3, strides=stride, padding="same", data_format="channels_last")
            stride = 1

        if avd and avd_first:
            x = avd_layer(x)

        if radix >= 1:
            x = self._SplAtConv2d(x, filters=group_width, kernel_size=3, stride=stride, dilation=self.dilation,
                                  groups=self.cardinality, radix=radix)
        else:
            x = Conv2D(filters, kernel_size=3, strides=stride, padding="same", kernel_initializer="he_normal",
                       dilation_rate=self.dilation, use_bias=False, data_format="channels_last")(x)

        if avd and not avd_first:
            x = avd_layer(x)

        x = BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
        x = Activation(self.active)(x)
        x = Conv2D(filters, kernel_size=3, strides=1, padding="same", kernel_initializer="he_normal",
                   dilation_rate=self.dilation, use_bias=False, data_format="channels_last")(x)
        m2 = Add()([x, short_cut])
        return m2

    def _make_layer(self, input_tensor, blocks=4, filters=64, stride=2, is_first=True):
        x = input_tensor
        if self.using_basic_block is True:
            x = self._make_block_basic(x, first_block=True, filters=filters, stride=stride, radix=self.radix,
                                       avd=self.avd, avd_first=self.avd_first, is_first=is_first)

            for i in range(1, blocks):
                x = self._make_block_basic(
                    x, first_block=False, filters=filters, stride=1, radix=self.radix, avd=self.avd, avd_first=self.avd_first
                )

        elif self.using_basic_block is False:
            x = self._make_block(x, first_block=True, filters=filters, stride=stride, radix=self.radix, avd=self.avd,
                                 avd_first=self.avd_first, is_first=is_first)

            for i in range(1, blocks):
                x = self._make_block(
                    x, first_block=False, filters=filters, stride=1, radix=self.radix, avd=self.avd, avd_first=self.avd_first
                )
        return x

    def _make_Composite_layer(self, input_tensor, filters=256, kernel_size=1, stride=1, upsample=True):
        x = input_tensor
        x = Conv2D(filters, kernel_size, strides=stride, use_bias=False)(x)
        x = BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
        if upsample:
            x = UpSampling2D(size=2)(x)
        return x

    def build(self):
        input_sig = Input(shape=self.input_shape)
        x = self._make_stem(
            input_sig, stem_width=self.stem_width, deep_stem=self.deep_stem)

        if self.preact is False:
            x = BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
            x = Activation(self.active)(x)
        if self.verbose:
            print("stem_out", x.shape)

        x = MaxPool2D(pool_size=3, strides=2, padding="same",
                      data_format="channels_last")(x)
        if self.verbose:
            print("MaxPool2D out", x.shape)

        if self.preact is True:
            x = BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
            x = Activation(self.active)(x)

        if self.using_cb:
            second_x = x
            second_x = self._make_layer(
                x, blocks=self.blocks_set[0], filters=64, stride=1, is_first=False)
            second_x_tmp = self._make_Composite_layer(
                second_x, filters=x.shape[-1], upsample=False)
            if self.verbose:
                print('layer 0 db_com', second_x_tmp.shape)
            x = Add()([second_x_tmp, x])
        x = self._make_layer(
            x, blocks=self.blocks_set[0], filters=64, stride=1, is_first=False)
        if self.verbose:
            print("-" * 5, "layer 0 out", x.shape, "-" * 5)

        b1_b3_filters = [64, 128, 256, 512]
        for i in range(3):
            idx = i+1
            if self.using_cb:
                second_x = self._make_layer(
                    x, blocks=self.blocks_set[idx], filters=b1_b3_filters[idx], stride=2)
                second_x_tmp = self._make_Composite_layer(
                    second_x, filters=x.shape[-1])
                if self.verbose:
                    print('layer {} db_com out {}'.format(
                        idx, second_x_tmp.shape))
                x = Add()([second_x_tmp, x])
            x = self._make_layer(
                x, blocks=self.blocks_set[idx], filters=b1_b3_filters[idx], stride=2)
            if self.verbose:
                print('----- layer {} out {} -----'.format(idx, x.shape))

        x = GlobalAveragePooling2D(name='avg_pool')(x)
        if self.verbose:
            print("pool_out:", x.shape)  # remove the concats var

        if self.dropout_rate > 0:
            x = Dropout(self.dropout_rate, noise_shape=None)(x)

        # Output layer will have a size of 1 since it is binary classification and we are using sigmoid activation function
        fc_out = Dense(1, kernel_initializer="he_normal", use_bias=False,
                       activation=self.fc_activation, name="fc_NObias")(x)  # replace concats to x
        if self.verbose:
            print("fc_out:", fc_out.shape)

        model = models.Model(inputs=input_sig, outputs=fc_out)

        if self.verbose:
            print("Resnest builded with input {}, output{}".format(
                input_sig.shape, fc_out.shape))
            print("-------------------------------------------")
            print("")

        return model


def get_model(model_name='ResNest50', input_shape=(32, 36, 1), n_classes=2,
              verbose=True, dropout_rate=0.2, fc_activation=None, blocks_set=None, stem_width=None, **kwargs):
    '''get_model
    input_shape: (h,w,c)
    fc_activation: sigmoid,softmax
    '''
    model = ResNest(verbose=verbose, input_shape=input_shape,
                    n_classes=n_classes, dropout_rate=dropout_rate, fc_activation=fc_activation,
                    blocks_set=blocks_set, radix=2, groups=1, bottleneck_width=64, deep_stem=True,
                    stem_width=stem_width, avg_down=True, avd=True, avd_first=False, **kwargs).build()

    return model


# Defining the hyperparameters
model_name = 'ResNest50'
input_shape = [32, 36, 1]
n_classes = 2
fc_activation = 'sigmoid'
active = 'relu'
blocks_set = [3, 4, 6, 3]
stem_width = 32
dropout_rate = 0.2
learning_rate = 1e-3

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

checkpoint_path = './infant_cry/saved_models/19092021-213644'

model = get_model(model_name=model_name, input_shape=input_shape, n_classes=n_classes,
                  fc_activation=fc_activation, active=active, verbose=True, blocks_set=blocks_set,
                  stem_width=stem_width, dropout_rate=dropout_rate)

model.compile(optimizer=optimizer, loss='binary_crossentropy',
              metrics=['accuracy', 'AUC', 'Precision', 'Recall'])

# Load the weights from the checkpoint path for the best model and find the performance for the test data
model.load_weights(checkpoint_path)
loss, acc, auc, precision, recall = model.evaluate(
    X_test, Y_test, batch_size=32, verbose=1)
