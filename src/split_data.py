import numpy as np
import pandas as pd
import random
import torch
from config_reader import Config

def split_data(data, labels, ratio):
    cfg = Config()
    np.nan_to_num(data, nan=0, copy=False)

    if cfg.mode != 'predict':
        # ticks where cyclone is present
        cyclone_occurences = pd.read_csv("../shuffle_cyclone.csv", header=None)

        # ticks where there is no cyclone
        no_cyclone = pd.read_csv("../shuffle_no_cyclone.csv", header=None)

        cyclone_occurences = cyclone_occurences.values[0][:-1]
        no_cyclone = no_cyclone.values[0][:-1]

        cut_cyclone = int(ratio * len(cyclone_occurences))
        cut_no_cyclone = int(ratio * len(no_cyclone))

        train_id = cyclone_occurences[:cut_cyclone].astype(int)
        train_id = np.append(train_id, no_cyclone[:cut_no_cyclone].astype(int))

        train_id = np.sort(train_id)

        test_id = cyclone_occurences[cut_cyclone:-1].astype(int) 
        test_id = np.append(test_id, no_cyclone[cut_no_cyclone:-1].astype(int))
        test_id = np.sort(test_id)

        random.shuffle(train_id)
        random.shuffle(test_id)

    if cfg.mode == 'predict':
        ticks_num = data.shape[2]
        ticks = np.arange(start=0, stop=ticks_num, step=1)

        train_id = ticks[: int(ratio * ticks_num)]
        test_id = ticks[int(ratio * ticks_num):]

    train_data = torch.tensor(data[:, :, train_id], dtype=torch.double)
    test_data = torch.tensor(data[:, :, test_id], dtype=torch.double)
    labels_train = torch.tensor(labels[train_id], dtype=torch.int)
    labels_test = torch.tensor(labels[test_id], dtype=torch.int)

    return train_data, test_data, labels_train, labels_test, train_id, test_id