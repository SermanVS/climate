from show_test_data import show_test_data
import torch
import torch.nn as nn
import numpy as np
from config_reader import Config
from sigma import sigma
import pandas as pd

def get_start_ticks(labels_test):
    labels_test = np.array(labels_test)
    start_ticks = []
    for i in range(len(labels_test) - 1):
        if (labels_test[i] == 0 and labels_test[i+1] == 1):
            start_ticks.append(i+1)
    return np.array(start_ticks)

def predicted_event_id(tick, start_ticks):
    idx = (np.abs(start_ticks - tick)).argmin()
    return idx

def fill_preicted_events(predicted_events, start_ticks, labels_test):
    cfg = Config()
    #res = np.zeros(shape=len(labels_test), dtype=int)
    res = [0] * len(labels_test)
    for k, pr in enumerate(predicted_events):
        if pr == 1:
            for i in range(start_ticks[k], start_ticks[k] + cfg.w):
                res[i] = 1

    res = indices(res, 1)
    return np.array(res, dtype=int)

def indices(lst, item):
    return [i for i, x in enumerate(lst) if x == item]


def test(nn, test_data, labels_test):
    # Test the model
    cfg = Config()
    nn.eval()    
    sigmas = []
    tp_ids = []
    fp_ids = []
    fn_ids = []
    tn_ids = []

    if cfg.mode == 'predict':
        predicted = np.zeros(shape=test_data.shape[2])

    with torch.no_grad():
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        for i in range(test_data.shape[2]):
            if (labels_test[i] == 2):
                continue
            image = test_data[:, :, i]
            label = labels_test[i]
            
            image = image.unsqueeze(0) 
            image = image.unsqueeze(0) 
            
            test_output = nn(image)
            arg = test_output.item()
            sig = sigma(arg)
            sigmas.append(sig)
            
            pred_y = 1 if sig >= 0.5 else 0
            #print(sig)
            # 0 - 1 when cyclone is present
            if (pred_y == label and label == 1):
                tp += 1
                tp_ids.append(i)
            # 1 - 0 when no cyclone is present
            elif (pred_y == label and label == 0):
                tn += 1
                tn_ids.append(i)
            # 0 - 1 when no cyclone is present
            elif (pred_y != label and label == 1):
                fn += 1
                fn_ids.append(i)
            # 1 - 0 when cyclone is present
            elif (pred_y != label and label == 0):
                fp += 1
                fp_ids.append(i)
            pass   
        pass
    
    if cfg.mode == 'predict':
        predicted[tp_ids] = 1
        event_start_ticks = get_start_ticks(labels_test)
        predicted_events = np.zeros(shape=len(event_start_ticks))

        for item in tp_ids:
            predicted_event = predicted_event_id(item, event_start_ticks)
            predicted_events[predicted_event] = 1

        tp_ids = fill_preicted_events(predicted_events, event_start_ticks, labels_test)

        tp = predicted_events.sum()
        fn = len(event_start_ticks) - tp
    
        fp = fp // cfg.w
        tn = tn // cfg.w
    return tp, tn, fp, fn, sigmas, (tp_ids, fp_ids, fn_ids, tn_ids)