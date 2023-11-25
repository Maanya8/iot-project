import numpy as np
import pandas as pd
import os
import random
from utils.HAR_utils import *
from sklearn.preprocessing import LabelEncoder
import pdb
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()



random.seed(1)
np.random.seed(1)
data_path = "/home/maanya/FL-IoT/luflow"
dir_path = "/home/maanya/FL-IoT/luflow"


def generate_luflow(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    config_path = dir_path + "/config.json"
    train_path = dir_path + "/train/"
    test_path = dir_path + "/test/"

    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    X, y = load_data_luflow(data_path)
    statistic = []
    num_clients = len(y)
    print(num_clients)
    #print(y.shape)
    num_classes = len(np.unique(y))
    for i in range(num_clients):
        statistic.append([])
        for yy in sorted(np.unique(y[i])):
            idx = y[i] == yy
            statistic[-1].append((int(yy), int(len(X[i][idx]))))

    for i in range(num_clients):
        print(f"Client {i}\t Size of data: {len(X[i])}\t Labels: ", np.unique(y[i]))
        print(f"\t\t Samples of labels: ", [i for i in statistic[i]])
        print("-" * 50)

    train_data, test_data = split_data(X, y)
    # train_data, test_data = [], []
    # num_samples = {'train':[], 'test':[]}

    # X_train, X_test, y_train, y_test = train_test_split( X , y , test_size = 0.5, random_state = 0)
    # X_train = sc_X.fit_transform(X_train)
    # X_test = sc_X.transform(X_test)


    # train_data.append({'x': X_train, 'y': y_train})
    # num_samples['train'].append(len(y_train))
    # test_data.append({'x': X_test, 'y': y_test})
    # num_samples['test'].append(len(y_test))
    
    # print("Total number of samples:", sum(num_samples['train'] + num_samples['test']))
    # print("The number of train samples:", num_samples['train'])
    # print("The number of test samples:", num_samples['test'])
    # print()
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, statistic)
    
def load_data_luflow(data_folder):
    dataset = pd.read_csv(data_folder + '/rawdata/dataset/dataset.csv')

    item_data = dataset.iloc[0:864000, list(range(0, 14)) + [15]].values
    Y = dataset.iloc[0:864000, 14].values
    labelencoder_y = LabelEncoder()
    Y = labelencoder_y.fit_transform(Y)
    X = item_data.astype(np.float32)
    # rows = 864000
    # columns = 66


    # array_2d = np.zeros((rows, columns), dtype=np.float32)
    # X = np.hstack((X, array_2d), dtype=np.float32)
    # adder = X
    # for i in range(383):
    #     X = np.hstack((X,adder))

    Y= Y.reshape(-1, 28800)
    print(X.dtype)
    pdb.set_trace()

    # Convert the 2D NumPy array into a list of arrays
    YY = [arr for arr in Y]
    # rows = 864000
    # XX = np.split(X, rows // 28800)
    num_arrays = 864000 // 28800

# Reshape the 2D array into a list of 3D arrays
    XX = []
    for i in range(num_arrays):
        start_index = i * 28800
        end_index = (i + 1) * 28800
        reshaped_array = X[start_index:end_index].reshape(28800, -1, 15)
        XX.append(reshaped_array)

    for i in range(len(XX)):
        XX[i] = XX[i].reshape(-1,3,1,5)

    


    print("X0 shape", XX[0].shape)
    print("Y0 shape", YY[0].shape)
    print("X length", len(XX))
    print("Y length", len(YY))
    pdb.set_trace()

    return XX, YY

if __name__ == "__main__":
    generate_luflow(dir_path)
