import pandas as pd
import tensorflow as tf
import numpy as np
from pandas import Series


def process_file(filename, config):
    """读取输入电压和输出电压"""
    data = pd.read_csv(filename)
    vout = data.columns[1]
    vin = data.columns[2]
    length = len(data)
    # start = 6
    # x_pad = data[6:, vin]
    # y_pad = data[6:, vout]
    time_sample = 1000
    size = config.num_data
    x = np.zeros((size, time_sample, 1), dtype=float)
    y = np.zeros((size, time_sample, 1), dtype=float)
    i = 0
    for start in range(6, length, 1006):
        x_pad = data.ix[start:start + 999, vin].values.reshape(1000, 1)
        y_pad = data.ix[start:start + 999:, vout].values.reshape(1000, 1)
        x[i, :, :] = x_pad
        y[i, :, :] = y_pad
        i += 1
        # x[start-6:start-6+1000] = x_pad.values
        # y[start-6:start-6+1000] = y_pad.values
        # start += 1000

    # x = Series.as_matrix(x_pad)
    # y = Series.as_matrix(y_pad)
    return x, y


def batch_iter(x, y, config):
    """生成批次数据"""
    # data_len = len(x[0])
    # num_batch = data_len / config.time_steps

    indices = np.random.permutation(np.arange(config.num_data))
    x_shuffle = x[indices]
    y_shuffle = y[indices]
    return x_shuffle[0:config.batch_size], y_shuffle[0:config.batch_size]


