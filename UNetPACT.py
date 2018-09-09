# coding: utf-8
'''
TODO:
    UNet implementation for the PACT de-artifact reconstruction
    induced by the limited-view acquistion
Dependencies:
    Keras 2.0.8
    Tensorflow 1.3.0
    UNetConfig

Usage:
from UNetPACT import UNet_PA, dice_coef_loss, dice_coef
from keras.optimizers import Adam

model = UNet_PA()
model.load_weights(weights_filename) # optional
optim = Adam() # optimizer
loss = dice_coef_loss # loss function
metrics = [dice_coef]
model.compile(optimizer=optim, loss=dice_coef_loss, metrics=[dice_coef]) # configuration

model.fit(...)
'''
__author__ = 'ACM'
import tensorflow as tf
from tensorflow.contrib.keras import models
from tensorflow.contrib.keras import layers
from tensorflow.contrib.keras import backend as K

import UNetConfig as uc

'''
Hyper-parameters
'''
# input data
INPUT_SIZE = uc.INPUT_SIZE
INPUT_CHANNEL = uc.INPUT_CHANNEL   # 1-grayscale, 3-RGB scale
OUTPUT_MASK_CHANNEL = uc.OUTPUT_MASK_CHANNEL
# network structure
FILTER_NUM = uc.FILTER_NUM # number of basic filters for the first layer
FILTER_SIZE = uc.FILTER_SIZE # size of the convolutional filter
DOWN_SAMP_SIZE = uc.DOWN_SAMP_SIZE # size of pooling filters
UP_SAMP_SIZE = uc.UP_SAMP_SIZE # size of upsampling filters

'''
Definitions of loss and evaluation metrices
'''

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)


def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)


def jacard_coef_loss(y_true, y_pred):
    return -jacard_coef(y_true, y_pred)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def double_conv_layer(x, filter_size, size, dropout, batch_norm):
    '''
    construction of a double convolutional layer using
    SAME padding
    RELU nonlinear activation function
    :param x: input
    :param filter_size: size of convolutional filter
    :param size: number of filters
    :param dropout: FLAG & RATE of dropout.
            if < 0 dropout cancelled, if > 0 set as the rate
    :param batch_norm: flag of if batch_norm used,
            if True batch normalization
    :return: output of a double convolutional layer
    '''
    axis = 3
    conv = layers.Conv2D(size, (filter_size, filter_size), padding='same')(x)
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=axis)(conv)
    conv = layers.Activation('relu')(conv)
    conv = layers.Conv2D(size, (filter_size, filter_size), padding='same')(conv)
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=axis)(conv)
    conv = layers.Activation('relu')(conv)
    if dropout > 0:
        conv = layers.Dropout(dropout)(conv)
    return conv

def UNet_PA(dropout_rate=0.0, batch_norm=True):
    '''
    UNet construction
    convolution: 3*3 SAME padding
    pooling: 2*2 VALID padding
    upsampling: 3*3 VALID padding
    final convolution: 1*1
    :param dropout_rate: FLAG & RATE of dropout.
            if < 0 dropout cancelled, if > 0 set as the rate
    :param batch_norm: flag of if batch_norm used,
            if True batch normalization
    :return: UNet model for PACT recons
    '''
    # input data
    # dimension of the image depth
    inputs = layers.Input((INPUT_SIZE, INPUT_SIZE, INPUT_CHANNEL))
    axis = 3

    # Subsampling layers
    # double layer 1, convolution + pooling
    conv_128 = double_conv_layer(inputs, FILTER_SIZE, INPUT_SIZE, dropout_rate, batch_norm)
    pool_64 = layers.MaxPooling2D(pool_size=(2,2))(conv_128)
    # double layer 2
    conv_64 = double_conv_layer(pool_64, 2*FILTER_SIZE, INPUT_SIZE, dropout_rate, batch_norm)
    pool_32 = layers.MaxPooling2D(pool_size=(2,2))(conv_64)
    # double layer 3
    conv_32 = double_conv_layer(pool_32, 4*FILTER_SIZE, INPUT_SIZE, dropout_rate, batch_norm)
    pool_16 = layers.MaxPooling2D(pool_size=(2,2))(conv_32)
    # double layer 4
    conv_16 = double_conv_layer(pool_16, 8*FILTER_SIZE, INPUT_SIZE, dropout_rate, batch_norm)
    pool_8 = layers.MaxPooling2D(pool_size=(2,2))(conv_16)
    # double layer 5, convolution only
    conv_8 = double_conv_layer(pool_8, 16*FILTER_SIZE, INPUT_SIZE, dropout_rate, batch_norm)

    # Upsampling layers
    # double layer 6, upsampling + concatenation + convolution
    up_16 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(conv_8)
    up_16 = layers.concatenate([up_16, conv_16], axis=axis)
    up_conv_16 = double_conv_layer(up_16, 8*FILTER_SIZE, INPUT_SIZE, dropout_rate, batch_norm)
    # double layer 7
    up_32 = layers.concatenate([layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_16), conv_32], axis=axis)
    up_conv_32 = double_conv_layer(up_32, 4*FILTER_SIZE, INPUT_SIZE, dropout_rate, batch_norm)
    # double layer 8
    up_64 = layers.concatenate([layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32), conv_64], axis=axis)
    up_conv_64 = double_conv_layer(up_64, 2*FILTER_SIZE, INPUT_SIZE, dropout_rate, batch_norm)
    # double layer 9
    up_128 = layers.concatenate([layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_64), conv_128], axis=axis)
    up_conv_128 = double_conv_layer(up_128, FILTER_SIZE, INPUT_SIZE, dropout_rate, batch_norm)

    # 1*1 convolutional layers
    # valid padding
    # batch normalization
    # sigmoid nonlinear activation
    conv_final = layers.Conv2D(OUTPUT_MASK_CHANNEL, kernel_size=(1,1))(up_conv_128)
    conv_final = layers.BatchNormalization(axis=axis)(conv_final)
    conv_final = layers.Activation('sigmoid')(conv_final)

    # Model integration
    model = models.Model(inputs, conv_final, name="UNet")
    return model




