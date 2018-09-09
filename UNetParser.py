# coding: utf-8
"""
TODO:
    Data import & training & test for the UNet
Dependencies:
    Keras 2.0.8
    Tensorflow
    UNetConfig, UNetPACT
"""

__author__ = 'ACM'

import argparse

from UNetPACT import UNet_PA
from tensorflow.contrib.keras import optimizers, callbacks
import UNetConfig as uc
import MatDataIO as mio
import matplotlib.pyplot as plt

if uc.TRAIN_FLAG:
    # data import
    x_train, y_train = mio.loadBatchData(
        uc.TRAIN_DATA_PATH, uc.TRAINING_SIZE, start_num=uc.TRAINING_START)

    # model construction
    model = UNet_PA(dropout_rate=uc.DROPOUT_RATE, batch_norm=uc.BATCH_NORM_FLAG)
    if uc.MODEL_LOAD_FLAG:
        # model.load_weights(uc.MODEL_LOAD_PATH)
        model.load_weights(uc.MODEL_LOAD_PATH)
    # training setup
    optimizer = optimizers.Adam() # training optimizer
    loss = ['mean_squared_error'] # training loss function
    metrics = ['mae'] # training evaluation metrics

    # model configuration
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # Tensorboard visualization
    if uc.TENSORBOARD_FLAG:
        tb = callbacks.TensorBoard(log_dir=uc.LOG_PATH,
                                  histogram_freq=0,
                                  batch_size=uc.BATCH_SIZE,
                                  write_graph=False,
                                  write_images=True)
    else:
        tb = None


    # model training
    history = model.fit(x=x_train,
                        y=y_train,
                        batch_size=uc.BATCH_SIZE,
                        epochs=uc.EPOCH,
                        verbose=1,
                        callbacks=[tb],
                        validation_split=uc.VALIDATION_SPLIT,
                        validation_data=None,
                        shuffle=True,
                        class_weight=None,
                        sample_weight=None,
                        initial_epoch=0)

    # visualization
    if uc.TRAINING_VISUAL_FLAG:
        print(history.history.keys())

        fig = plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='lower left')
        plt.show()

        # results storing
        # fig.savefig('performance.png')
    if uc.MODEL_SAVE_FLAG:
        model.save(uc.MODEL_SAVE_PATH)


'''
Test
'''
if uc.TEST_FLAG:
    import numpy as np
    import scipy.io as scio
    model = UNet_PA(dropout_rate=uc.DROPOUT_RATE, batch_norm=uc.BATCH_NORM_FLAG)
    model.load_weights(uc.MODEL_LOAD_PATH)
    for img_num in range(uc.TEST_START, uc.TEST_START+uc.TEST_SIZE):
    # test_img = mio.loadSingleData(uc.TEST_DATA_PATH + 'homo_disc_line.mat)
        test_img = mio.loadSingleData(uc.TRAIN_DATA_PATH +'homo_2D_high_random_disc_'+str(img_num)+'.mat')
        img_input = test_img['p0_recons'][np.newaxis,:,:,np.newaxis]
        img_label = test_img['p0_true'][np.newaxis,:,:,np.newaxis]
        img_pred = model.predict(img_input, batch_size=1)
        img_pred = img_pred[0,:,:,0]
        # plt.imshow(img_pred)
        # plt.show()

        scio.savemat(uc.PRED_DATA_PATH+str(img_num)+'_pred.mat',
                     {'p0_pred':img_pred, 'p0_true':test_img['p0_true'], 'p0_recons':test_img['p0_recons']})
    if False:
        for img_num in range(uc.TEST_START, uc.TEST_START+uc.TEST_SIZE):
    # test_img = mio.loadSingleData(uc.TEST_DATA_PATH + 'homo_disc_line.mat')
            for ii in range(3):
                test_img = mio.loadSingleData(uc.TRAIN_DATA_PATH +'real_2D_disc_'+str(img_num)+'_'+str(ii+1)+'.mat')
                img_input = test_img['p0_recons'][np.newaxis,:,:,np.newaxis]
                img_label = test_img['p0_true'][np.newaxis,:,:,np.newaxis]
                img_pred = model.predict(img_input, batch_size=1)
                img_pred = img_pred[0,:,:,0]
                # plt.imshow(img_pred)
                # plt.show()

                scio.savemat(uc.PRED_DATA_PATH+str(img_num)+'_'+str(ii+1)+'_pred.mat',
                             {'p0_pred':img_pred, 'p0_true':test_img['p0_true'], 'p0_recons':test_img['p0_recons']})


