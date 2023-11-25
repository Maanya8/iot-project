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

    item_data = dataset.iloc[1:1014115, list(range(0, 14)) + [15]].values
    Y = dataset.iloc[1:1014115, 14].values
    labelencoder_y = LabelEncoder()
    Y = labelencoder_y.fit_transform(Y)
    X = item_data.astype(np.float32)
    X = np.nan_to_num(X, nan=0)
    Y = np.nan_to_num(Y, nan=0)

    X_label_1 = []
    X_label_2 = []
    X_label_3 = []
    for i, label in enumerate(Y):
        if label == 0:
            X_label_1.append(X[i, :])
        elif label == 1:
            X_label_2.append(X[i, :])
        elif label == 2:
            X_label_3.append(X[i, :])

# Convert the lists to arrays
    X_label_1 = np.array(X_label_1)
    X_label_2 = np.array(X_label_2)
    X_label_3 = np.array(X_label_3)

# Convert the arrays back to pandas DataFrames
    print("1 number", X_label_1.shape)
    print("2 number", X_label_2.shape)
    print("3 number", X_label_3.shape)
    X_label_1 = X_label_1[:-320]
    X_label_1 = X_label_1.reshape(-1,1152)
    X_label_1 = X_label_1[:-25]
    print("1 number", X_label_1.shape)
    X_label_2 = X_label_2[:-158]
    X_label_2 = X_label_2.reshape(-1,1152)  
    print("2 number", X_label_2.shape)
    X_label_3 = X_label_3[:-19]
    X_label_3 = X_label_3.reshape(-1,1152) 
    print("3 number", X_label_3.shape) 
    pdb.set_trace()
    X = np.vstack((X_label_1, X_label_2, X_label_3))
    #rows = 11220
    #columns = 1152
    Y = np.vstack((np.zeros((5830,1)),np.ones((4425,1)),np.full((965, 1), 2)))


    # array_2d = np.ones((rows, columns), dtype=np.float32)
    # X = np.hstack((X, array_2d), dtype=np.float32)
    # adder = X
    # for i in range(76):
    #     X = np.hstack((X,adder))

    # X = np.delete(X, np.s_[-3:], axis=1)
    Y= Y.reshape(30, 374)
    print(X.dtype)
    pdb.set_trace()

    # Convert the 2D NumPy array into a list of arrays
    YY = [arr for arr in Y]
    # rows = 864000

    # XX = np.split(X, rows // 28800)
    num_arrays = 11220 // 374

    # Reshape the 2D array into a list of 3D arrays
    XX = []
    for i in range(num_arrays):
        start_index = i * 374
        end_index = (i + 1) * 374
        reshaped_array = X[start_index:end_index].reshape(374,  -1, 128)
        XX.append(reshaped_array)

    for i in range(len(XX)):
        XX[i] = XX[i].reshape(-1,9,1,128)

    


    print("X0 shape", XX[0].shape)
    print("Y0 shape", YY[0].shape)
    print("X length", len(XX))
    print("Y length", len(YY))
    pdb.set_trace()

    return XX, YY

if __name__ == "__main__":
    generate_luflow(dir_path)
