"""
MatDataIO.py
TODO: IO communications with Matlab data
Dependencies:
    Numpy
    Scipy
    UNetConfig
"""
__author__ = 'ACM'

import scipy.io as scio
import numpy as np
import UNetConfig as uc

def loadSingleData(data_path):
    """
    load single sample
    :param data_path: file path of the single sample
    :return: numpy array of the single .mat data
    """
    return scio.loadmat(data_path)

def loadBatchData(data_path, batch_size, start_num=1):
    """
    load batch samples
    :param data_path: file path of the batch samples
    :param batch_size: number of samples in the batch
    :return: numpy array of the multiple .mat data
    """
    data_path_single = data_path + 'homo_2D_high_random_disc_'+str(start_num)+'.mat'
    # print(data_path_single)
    data = scio.loadmat(data_path_single)
    x_result = np.zeros(shape=(batch_size, data['p0_recons'].shape[0], data['p0_recons'].shape[1], uc.INPUT_CHANNEL))
    y_result = np.zeros(shape=(batch_size, data['p0_true'].shape[0], data['p0_true'].shape[1], uc.INPUT_CHANNEL))
    x_result[0,:,:,:] = data['p0_recons'][:,:,np.newaxis]
    y_result[0,:,:,:] = data['p0_true'][:,:,np.newaxis]

    for ii in range(start_num, start_num+batch_size):
        data_path_single = data_path + 'homo_2D_high_random_disc_' + str(ii+1) + '.mat'
        # print(data_path_single)
        data = scio.loadmat(data_path_single)
        x_result[ii-start_num,:,:,:] = data['p0_recons'][:,:,np.newaxis]
        y_result[ii-start_num,:,:,:] = data['p0_true'][:,:,np.newaxis]

    return x_result, y_result



'''
Test
'''
if False:
    ROOT_PATH = '/Users/asd1/phd/Duke/DLforPA/'
    DATA_PATH = ROOT_PATH + 'data/'
    TRAIN_DATA_PATH = DATA_PATH + 'train_simu/'

    xxx, yyy = loadBatchData(TRAIN_DATA_PATH, 10)
