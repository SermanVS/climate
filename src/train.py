import torch
import torch.nn as nn
import numpy as np
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from show_loss import show_loss
from test import test
from get_test_stats import get_test_stats
from show_test_data import show_test_data
from sklearn import metrics
from parameters import Stats
from IPython.display import clear_output

'''
Train the network on given data.

nn           - Network to train.
batch_size   - Batch size to apply to data.
num_epochs   - How many epochs to train for.
train_data   - Data to train on.
labels_train - Labels for train_data.
loss_func    - Loss function.
optimizer    - Optimizer for gradients.
draw         - Whether to display loss dynamics.
step_test    - Whether to test on test data after each epoch.
args         - test data and test labels.

It's not recommended to use 'draw' and 'step_test' at the same time.
'''
def train(nn, batch_size, num_epochs, train_data, labels_train, loss_func, optimizer, args, draw=False, step_test=False):  
    nn.train()
    nn = nn.double()
    loss_vals = []
    epoch_loss = []
    
    test_stats_list = []
    train_stats_list = []

    tprs_test = []
    fprs_test = []
    tprs_train  = []
    fprs_train = []

    fprs_test.append(0)
    tprs_test.append(0)
    fprs_train.append(0)
    tprs_train.append(0)

    for epoch in tqdm(range(num_epochs)):
        running_loss = 0.0
        iters = int(train_data.shape[2] / batch_size)
        for i in range(iters):
            images = train_data[:, :, i:i + batch_size]
            labels = labels_train[i:i + batch_size]        
            
            b_x = images  
            b_y = labels.double()
            
            b_x = b_x.reshape((batch_size, 1, 36, 69))          
                
            optimizer.zero_grad()
            
            output = nn(b_x)
            loss = loss_func(input=output, target=b_y)   
            epoch_loss.append(loss.item())
            running_loss += loss.item() * batch_size
            if (draw == True and i != 0 and i % 100 == 0 and not step_test):
                show_loss(epoch_loss, loss_vals)
            elif (draw and i != 0 and i % 100 == 0 and step_test and epoch > 0):
                show_loss(epoch_loss, loss_vals, clear_after=False)
                show_test_data(test_stats_list, train_stats_list)
                clear_output(True)
            elif (draw and i != 0 and i % 100 == 0 and step_test and epoch == 0):
                show_loss(epoch_loss, loss_vals)
            
            loss.backward()                           
            optimizer.step()  

        if (step_test):
            tp, tn, fp, fn, sigmas, _ = test(nn, args[0], args[1])  
            test_stats = get_test_stats(args[0].shape[2], tp, tn, fp, fn, sigmas)          
            test_stats_list.append(test_stats)          

            tp, tn, fp, fn, sigmas, _ = test(nn, train_data, labels_train)  
            train_stats = get_test_stats(args[0].shape[2], tp, tn, fp, fn, sigmas)       
            train_stats_list.append(train_stats)

            show_test_data(test_stats_list, train_stats_list)
            nn.train()
        pass
        loss_vals.append(running_loss / train_data.shape[2])
        
    tprs_test += [item.tpr for item in test_stats_list]
    tprs_train += [item.tpr for item in train_stats_list]
    fprs_test += [item.fpr for item in test_stats_list]
    fprs_train += [item.fpr for item in train_stats_list]

    fprs_test.append(1)
    tprs_test.append(1)
    fprs_train.append(1)
    tprs_train.append(1)

    fprs_test, tprs_test = zip(*sorted(zip(fprs_test, tprs_test)))
    fprs_train, tprs_train = zip(*sorted(zip(fprs_train, tprs_train)))

    # sort by fpr
    auc_test = metrics.auc(fprs_test, tprs_test)
    auc_train = metrics.auc(fprs_train, tprs_train)
    return auc_test, auc_train 
