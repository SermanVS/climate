from show_test_data import show_test_data
import torch
import torch.nn as nn
import numpy as np

from sigma import sigma
from get_test_stats import get_test_stats

def test(nn, test_data, labels_test):
    # Test the model
    nn.eval()    
    sigmas = []

    with torch.no_grad():
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        for i in range(test_data.shape[2]):
            image = test_data[:, :, i]
            label = labels_test[i]
            
            image = image.unsqueeze(0) 
            image = image.unsqueeze(0) 
            
            test_output = nn(image)
            arg = test_output.item()
            sig = sigma(arg)
            sigmas.append(sig)
            
            pred_y = 1 if sig >= 0.5 else 0
            
            # 0 - 1 when cyclone is present
            if (pred_y == label and label == 1):
                tp += 1
            # 1 - 0 when no cyclone is present
            elif (pred_y == label and label == 0):
                tn += 1
            # 0 - 1 when no cyclone is present
            elif (pred_y != label and label == 1):
                fn += 1
            # 1 - 0 when cyclone is present
            elif (pred_y != label and label == 0):
                fp += 1
            pass   
        pass

    return tp, tn, fp, fn, sigmas