
import numpy as np
import urllib
import gzip


def get_dexter_data(filename, mode='online'):
    """
    A function that reads in the original dexter data in sparse form of feature:value
    and transform them into matrix form.
    # Arguments:
    filename: the url to either the dexter_train.data or dexter_valid.data
    mode: either 'text' for unpacked file; 'gz' for .gz file; or 'online' to download from the UCI repo
    # Return:
    the dexter data in matrix form.
    """
    if mode == 'text':
        with open(filename) as f:
            readin_list = f.readlines()
    elif mode == 'gz':
        with gzip.open(filename) as f:
            readin_list = f.readlines()
    elif mode == 'online':
        f = urllib.urlopen(filename)
        readin_list = f.readlines()
        f.close()

    def to_dense_dexter(string_array):
        n = len(string_array)
        inds = np.zeros(n, dtype='int32')
        vals = np.zeros(n, dtype='int32')
        ret = np.zeros(20000, dtype='int32')
        for i in range(n):
            this_split = string_array[i].split(':')
            inds[i] = int(this_split[0])
            vals[i] = int(this_split[1])
        ret[inds] = vals
        return ret

    N = len(readin_list)
    dat = [None]*N

    for i in range(N):
        dat[i] = to_dense_dexter(readin_list[i].split(' ')[0:-1])[None, :]

    dat = np.concatenate(dat, axis=0).astype('float32')
    return dat


def get_dorothea_data(filename, mode='online'):
    """
    A function that reads in the original dorothea data in sparse form of feature:value
    and transform them into matrix form.
    # Arguments:
    filename: the url to either the dorothea_train.data or dorothea_valid.data
    mode: either 'text' for unpacked file; 'gz' for .gz file; or 'online' to download from the UCI repo
    # Return:
    the dexter data in matrix form.
    """
    if mode == 'txt':
        with open(filename) as f:
            readin_list = f.readlines()
    elif mode == 'gz':
        with gzip.open(filename) as f:
            readin_list = f.readlines()
    elif mode == 'online':
        f = urllib.urlopen(filename)
        readin_list = f.readlines()
        f.close()

    def to_dense_dorothea(string_array):
        n = len(string_array)
        inds = np.zeros(n, dtype='int32')
        ret = np.zeros(100001, dtype='int32')
        for i in range(n):
            this_split = string_array[i].split(' ')
            inds[i] = int(this_split[0])
        ret[inds] = 1
        return ret

    N = len(readin_list)
    dat = [None]*N

    for i in range(N):
        dat[i] = to_dense_dorothea(readin_list[i].split(' ')[1:-1])[None, :]

    dat = np.concatenate(dat, axis=0).astype('float32')
    return dat[:, 1::]  # the first column all zero


def load_NIPS2003_data(data_name):

    repo = 'http://archive.ics.uci.edu/ml/machine-learning-databases/'
    url = repo + data_name + '/' + data_name.upper() + '/'

    X_train = None
    Y_train = None
    X_valid = None
    Y_valid = None

    if data_name in ['arcene', 'gisette']:
        X_train = np.genfromtxt(url + data_name + '_train.data')
        Y_train = np.genfromtxt(url + data_name + '_train.labels')
        X_valid = np.genfromtxt(url + data_name + '_valid.data')
        Y_valid = np.genfromtxt(repo + data_name + '/' + data_name + '_valid.labels')
    elif data_name == 'dexter':
        X_train = get_dexter_data(url + data_name + '_train.data')
        Y_train = get_dexter_data(url + data_name + '_train.data')
        X_valid = get_dexter_data(url + data_name + '_valid.data')
        Y_valid = get_dexter_data(repo + data_name + '/' + data_name + '_valid.labels')
    elif data_name == 'dorothea':
        X_train = get_dorothea_data(url + data_name + '_train.data')
        Y_train = get_dorothea_data(url + data_name + '_train.data')
        X_valid = get_dorothea_data(url + data_name + '_valid.data')
        Y_valid = get_dorothea_data(repo + data_name + '/' + data_name + '_valid.labels')

    return [X_train, Y_train, X_valid, Y_valid]
