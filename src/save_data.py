import numpy
import csv

'''
Saves data about this Neural network in a csv file
'''
def save_data(name, desc, filename, hyperparameters, resulting_hyperparameters, test_stats, auc, comment=''):
    (conf_mx_train, conf_mx_test), (BA_train, BA_test), (F1_train, F1_test), (G_train, G_test), (I_train, I_test), \
        (fpr_train, fpr_test), (tpr_train, tpr_test) = test_stats
    auc_train, auc_test = auc
    path = "..\\temp\\" + name + ".csv"

    with open(path, 'w+') as f:
        writer = csv.writer(f)
        writer.writerow([desc, filename, hyperparameters, resulting_hyperparameters, \
            conf_mx_train[0], conf_mx_train[1], conf_mx_test[0], conf_mx_test[1], \
            BA_train, BA_test, F1_train, F1_test, G_train, G_test, I_train, I_test, \
            fpr_train, fpr_test, tpr_train, tpr_test, auc_train, auc_test, comment])