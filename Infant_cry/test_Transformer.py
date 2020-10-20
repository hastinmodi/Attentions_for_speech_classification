# File for testing model on 85 and 80% of frames

import random
import sklearn.metrics
import numpy as np
import pandas as pd 
import os
import librosa
import matplotlib.pyplot as plt
import math
import tensorflow as tf
import datetime
from pathlib import Path


# Setting seeds for reproducibility
random.seed(42)
tf.random.set_seed(42)

df = pd.read_csv('./data_infant_cry.csv')
df = df.sample(frac=1, random_state=42).reset_index(drop=True) # For shuffling

# Set the parameters for audio processing
class conf:
    sampling_rate = 16000
    duration = 1 # All files in the infant cry database have a duration of 1 sec
    samples = int(sampling_rate * duration)
    n_mfcc = 36
    del_frames = [0.0, 0.15, 0.2] # The amount of frames to be removed


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
        y = np.pad(y, (math.ceil(offset), math.ceil(conf.samples - len(y) - offset)), 'constant')
    return y

# We convert our raw audio into MFCCs
def convert_audio(conf, audio):
    mfcc = librosa.feature.mfcc(audio, sr=conf.sampling_rate, n_mfcc=conf.n_mfcc)
    mfcc = mfcc.astype(np.float32)
    return mfcc

# This implements the previous two functions for one file
def read_file(conf, pathname, trim_long_data, debug_display=False):
    x = read_audio(conf, pathname, trim_long_data)
    mfcc = convert_audio(conf, x)
    n_rows, n_cols = mfcc.shape
    # Cutting 20% frames from the audio
    ind = np.random.randint(0, n_cols - int(conf.del_frames[2]*n_cols) + 1)
    mfcc = np.delete(mfcc, np.s_[ind:ind + int(conf.del_frames[2]*n_cols)], axis=1)
    return mfcc

# This generalizes for all the audio files
def convert_wav_to_mfcc(df, source, folders):
  X_test, Y_test = [], []
  for i in df.index:
    if(df.at[i, 'folder'] == folders[0]):
      x = read_file(conf, source[0] + '/' + str(df.at[i, 'files']), trim_long_data=True)
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


# Start building the model
# This allows to the transformer to know where there is real data and where it is padded
def create_padding_mask(seq):
  seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

  # add extra dimensions to add the padding
  # to the attention logits.
  return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def scaled_dot_product_attention(query, key, value, mask):
  """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead) 
  but it must be broadcastable for addition.
  
  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable 
          to (..., seq_len_q, seq_len_k). Defaults to None.
    
  Returns:
    output, attention_weights
  """

  matmul_qk = tf.matmul(query, key, transpose_b=True) # (..., seq_len_q, seq_len_k)

  # scale matmul_qk
  depth = tf.cast(tf.shape(key)[-1], tf.float32)
  logits = matmul_qk / tf.math.sqrt(depth)

  # add the mask zero out padding tokens.
  if mask is not None:
    logits += (mask * -1e9)

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  attention_weights = tf.nn.softmax(logits, axis=-1) # (..., seq_len_q, seq_len_k)

  return tf.matmul(attention_weights, value) # (..., seq_len_q, depth_v)

class MultiHeadAttention(tf.keras.layers.Layer):

  def __init__(self, d_model, num_heads, name="multi_head_attention"):
    super(MultiHeadAttention, self).__init__(name=name)
    self.num_heads = num_heads
    self.d_model = d_model

    assert d_model % self.num_heads == 0

    self.depth = d_model // self.num_heads

    self.query_dense = tf.keras.layers.Dense(units=d_model)
    self.key_dense = tf.keras.layers.Dense(units=d_model)
    self.value_dense = tf.keras.layers.Dense(units=d_model)

    self.dense = tf.keras.layers.Dense(units=d_model)

  def split_heads(self, inputs, batch_size):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    inputs = tf.reshape(inputs, shape=(batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(inputs, perm=[0, 2, 1, 3])

  def call(self, inputs):
    query, key, value, mask = inputs['query'], inputs['key'], inputs['value'], inputs['mask']
    batch_size = tf.shape(query)[0]

    # linear layers
    query = self.query_dense(query) # (batch_size, seq_len, d_model)
    key = self.key_dense(key) # (batch_size, seq_len, d_model)
    value = self.value_dense(value) # (batch_size, seq_len, d_model)

    query = self.split_heads(query, batch_size) # (batch_size, num_heads, seq_len_q, depth)
    key = self.split_heads(key, batch_size) # (batch_size, num_heads, seq_len_q, depth)
    value = self.split_heads(value, batch_size) # (batch_size, num_heads, seq_len_q, depth)

    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)

    scaled_attention = scaled_dot_product_attention(query, key, value, mask)

    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3]) # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model)) # (batch_size, seq_len_q, d_model)

    outputs = self.dense(concat_attention) # (batch_size, seq_len_q, d_model)

    return outputs

class PositionalEncoding(tf.keras.layers.Layer):

  def __init__(self, position, d_model):
    super(PositionalEncoding, self).__init__()
    self.pos_encoding = self.positional_encoding(position, d_model)

  def get_angles(self, position, i, d_model):
    angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
    return position * angles

  def positional_encoding(self, position, d_model):
    angle_rads = self.get_angles(position=tf.range(position, dtype=tf.float32)[:, tf.newaxis], i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :], d_model=d_model)
    # apply sin to even index in the array
    sines = tf.math.sin(angle_rads[:, 0::2])
    # apply cos to odd index in the array
    cosines = tf.math.cos(angle_rads[:, 1::2])

    pos_encoding = tf.concat([sines, cosines], axis=-1)
    pos_encoding = pos_encoding[tf.newaxis, ...]
    return tf.cast(pos_encoding, tf.float32)

  def call(self, inputs):
    return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

def encoder_layer(units, d_model, num_heads, dropout,name="encoder_layer"):

  inputs = tf.keras.Input(shape=(None,d_model ), name="inputs")
  padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

  attention = MultiHeadAttention(d_model, num_heads, name="attention")({'query': inputs, 'key': inputs, 'value': inputs, 'mask': padding_mask})

  attention = tf.keras.layers.Dropout(rate=dropout)(attention)
  attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attention)

  outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention)
  outputs = tf.keras.layers.Dense(units=d_model)(outputs)
  outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
  outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention + outputs)

  return tf.keras.Model(inputs=[inputs, padding_mask], outputs=outputs, name=name)

def encoder(time_steps, num_layers, units, d_model, num_heads, dropout, projection, name="encoder"):

  inputs = tf.keras.Input(shape=(None,d_model), name="inputs")
  padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")
  
  if projection=='linear':
    # We implement a linear projection based on Very Deep Self-Attention Networks for End-to-End Speech Recognition (https://arxiv.org/abs/1904.13377)
    projection=tf.keras.layers.Dense( d_model,use_bias=True, activation='linear')(inputs)
  
  else:
    projection=tf.identity(inputs)
   
  projection *= tf.math.sqrt(tf.cast(d_model, tf.float32))
  projection = PositionalEncoding(time_steps, d_model)(projection)

  outputs = tf.keras.layers.Dropout(rate=dropout)(projection)

  for i in range(num_layers):
    outputs = encoder_layer(units=units, d_model=d_model, num_heads=num_heads, dropout=dropout, name="encoder_layer_{}".format(i))([outputs, padding_mask])
 
  return tf.keras.Model(inputs=[inputs, padding_mask], outputs=outputs, name=name) # (batch_size, input_seq_len, d_model)

def transformer(time_steps, num_layers, units, d_model, num_heads, dropout, output_size, projection, name="transformer"):

  inputs = tf.keras.Input(shape=(None,d_model), name="inputs")
  
  enc_padding_mask = tf.keras.layers.Lambda(create_padding_mask, output_shape=(1, 1, None), name='enc_padding_mask')(tf.dtypes.cast(
          
    # Our input has a dimension of length X d_model but the masking is applied to a vector
    # We get the sum for each row and result is a vector. So, result 0 indicates that position was masked      
    tf.math.reduce_sum(inputs, axis=2, keepdims=False, name=None), tf.int32))
  
  enc_outputs = encoder(time_steps=time_steps, num_layers=num_layers, units=units, d_model=d_model, num_heads=num_heads, dropout=dropout, projection=projection, name='encoder')(inputs=[inputs, enc_padding_mask]) # (batch_size, inp_seq_len, d_model)

  # We reshape for feeding our FC in the next step
  outputs = tf.reshape(enc_outputs,(-1,time_steps*d_model))
  
  # Padding since the number of frames have decreased 
  # (for 5 sec, 157 * 36 = 5652)
  # (for 3 sec, 94 * 36 = 3384)
  # (for 1 sec, 32 * 36 = 1152)
  if(time_steps*d_model!=1152): # Since 1 sec audio is taken
      paddings = tf.constant([[0, 0], [int((1152-time_steps*d_model)/2), int((1152-time_steps*d_model)/2)]])
      outputs = tf.pad(outputs, paddings, "SYMMETRIC")

  # We predict the class with the maximum probability using sigmoid activation
  outputs = tf.keras.layers.Dense(units=output_size,use_bias=True,activation='sigmoid', name="outputs")(outputs)

  return tf.keras.Model(inputs=[inputs], outputs=outputs, name='audio_class')


OUTPUT_SIZE = 1 # Output size of our model will be 1 since it is a binary classification and we are using sigmoid activation function
D_MODEL = X_test.shape[2] # No. of mfccs
TIME_STEPS = X_test.shape[1] # No. of samples

# Defining the hyperparameters
projection = ['linear','none']
NUM_LAYERS = 6
NUM_HEADS = 6
UNITS = 1023
DROPOUT = 0.1
learning_rate = 5e-5

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

checkpoint_path = './infant_cry/saved_models/18102020-151728' 

model = transformer(time_steps=TIME_STEPS, num_layers=NUM_LAYERS, units=UNITS, d_model=D_MODEL, 
                    num_heads=NUM_HEADS, dropout=DROPOUT, output_size=OUTPUT_SIZE, projection=projection[0])

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', 'AUC', 'Precision', 'Recall'])  

# Load the weights from the checkpoint path for the best model and find the performance for the test data
model.load_weights(checkpoint_path)
loss, acc, auc, precision, recall = model.evaluate(X_test, Y_test, batch_size=32, verbose=1)
