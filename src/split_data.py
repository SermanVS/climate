import numpy as np
import pandas as pd
import random
import torch

def split_data(data, labels, ratio):
    cyclone_occurences = pd.read_csv("../shuffle_cyclone.csv", header=None)
    no_cyclone = pd.read_csv("../shuffle_no_cyclone.csv", header=None)

    np.nan_to_num(data, nan=0, copy=False)

    cyclone_occurences = cyclone_occurences.values[0][:-1]
    no_cyclone = no_cyclone.values[0][:-1]

    cut_cyclone = int(ratio * len(cyclone_occurences))
    cut_no_cyclone = int(ratio * len(no_cyclone))

    train_id = cyclone_occurences[:cut_cyclone].astype(int)
    train_id = np.append(train_id, no_cyclone[:cut_no_cyclone].astype(int))

    test_id = cyclone_occurences[cut_cyclone:-1].astype(int) 
    test_id = np.append(test_id, no_cyclone[cut_no_cyclone:-1].astype(int))

    random.shuffle(train_id)
    random.shuffle(test_id)

    train_data = torch.tensor(data[:, :, train_id], dtype=torch.double)
    test_data = torch.tensor(data[:, :, test_id], dtype=torch.double)
    labels_train = torch.tensor(labels[train_id], dtype=torch.int)
    labels_test = torch.tensor(labels[test_id], dtype=torch.int)

    return train_data, test_data, labels_train, labels_test, train_id, test_id