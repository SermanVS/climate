import numpy as np
import matplotlib.pyplot as plt
from perform_g_test import perform_g_test
from parameters import Stats

def get_test_stats(size, tp, tn, fp, fn, sigmas, print_res=False):
    conf_matrix = np.array([np.array([tp, fp]), np.array([fn, tn])])
    
    # FPR
    try:
        fpr = fp / (fp + tn)
    except:
        fpr = 0

    # TPR
    try:
        tpr = tp / (tp + fn)
    except:
        tpr = 0


    # Balanced accuracy
    try:
        sensitivity = tp / (tp + fn) 
        specificity = tn / (fp + tn) 
        balanced_accuracy = (sensitivity + specificity) / 2
    except:
        balanced_accuracy = 0

    # F1 score
    try:
        f1 = tp / (tp + 0.5 * (fp + fn))
    except:
        f1 = 0

    # G-stats
    try:
        g, p, I = perform_g_test(conf_matrix)
    except:
        g = 0
        I = 0
    if (print_res):
        print(f'Accuracy of the model on the {size} test images: %.5f' % balanced_accuracy)
        print(f'F1 score of the model on the {size} test images: %.5f' % f1)
        print(f'G-stat of the model on the {size} test images: %.2f' % g)

    stats = Stats(conf_matrix, balanced_accuracy, f1, g, I, fpr, tpr)
    
    return stats