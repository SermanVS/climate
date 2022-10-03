import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
from parameters import Stats

def show_test_data(test_stats_list, train_stats_list):
    plt.figure(figsize=[18, 12])
    
    ba_test = [item.ba for item in test_stats_list]
    ba_train = [item.ba for item in train_stats_list]
    plt.subplot(2, 3, 1)
    plt.plot(np.arange(start=1, stop=len(ba_test) + 1, step=1), ba_test, 'o-', label='test')
    plt.plot(np.arange(start=1, stop=len(ba_train) + 1, step=1), ba_train, 'o-', label='train')
    plt.xlim((1, len(ba_test) + 1))
    plt.ylim((0, max(max(ba_test), 1)))
    plt.title("Balanced accuracy change")
    plt.xlabel("epoch")
    plt.ylabel("BA")
    plt.legend()
    plt.grid()

    f1_test = [item.f1 for item in test_stats_list]
    f1_train = [item.f1 for item in train_stats_list]
    plt.subplot(2, 3, 2)
    plt.plot(np.arange(start=1, stop=len(f1_test) + 1, step=1), f1_test, 'o-', label='test')
    plt.plot(np.arange(start=1, stop=len(f1_train) + 1, step=1), f1_train, 'o-', label='train')
    plt.xlim((1, len(f1_test) + 1))
    plt.ylim((0, max(max(f1_test), 1)))
    plt.title("F1 score change")
    plt.xlabel("epoch")
    plt.ylabel("F1")
    plt.legend()
    plt.grid()

    g_test = [item.g for item in test_stats_list]
    g_train = [item.g for item in train_stats_list]
    plt.subplot(2, 3, 3)
    plt.plot(np.arange(start=1, stop=len(g_test) + 1, step=1), g_test, 'o-', label='test')
    plt.plot(np.arange(start=1, stop=len(g_train) + 1, step=1), g_train, 'o-', label='train')
    plt.xlim((1, len(g_test) + 1))
    plt.ylim((0, max(max(g_test), max(g_train), 1)))
    plt.title("G-stat change")
    plt.xlabel("epoch")
    plt.ylabel("G")
    plt.legend()
    plt.grid()

    '''
    plt.subplot(2, 3, 4)
    plt.plot(np.arange(start=1, stop=len(iv_test) + 1, step=1), iv_test, 'o-', label='test')
    plt.plot(np.arange(start=1, stop=len(iv_train) + 1, step=1), iv_train, 'o-', label='train')
    plt.xlim((1, len(iv_test) + 1))
    plt.ylim((0, max(max(iv_test), 0.2)))
    plt.title("I-stat change")
    plt.xlabel("epoch")
    plt.ylabel("I")
    plt.legend()
    plt.grid()
    '''

    tpr_test = [item.tpr for item in test_stats_list]
    tpr_train = [item.tpr for item in train_stats_list]
    fpr_test = [item.fpr for item in test_stats_list]
    fpr_train = [item.fpr for item in train_stats_list]
    plt.subplot(2, 3, 4)
    plt.scatter(fpr_test, tpr_test, label='test')
    plt.scatter(fpr_train, tpr_train, label='train')

    xs = np.arange(start=0, stop=1, step=0.01)
    ys = xs + (tpr_test[0] - fpr_test[0])
    plt.plot(xs, ys, label='fpr = tpr')
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.title("TPR/FPR")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.grid()

    plt.subplot(2, 3, 5)
    plt.plot(np.arange(start=1, stop=len(tpr_test) + 1, step=1), tpr_test, 'o-', label='test')
    plt.plot(np.arange(start=1, stop=len(tpr_train) + 1, step=1), tpr_train, 'o-', label='train')
    plt.xlim((1, len(tpr_test) + 1))
    plt.ylim((0, max(max(tpr_test), 1)))
    plt.title("TPR change")
    plt.xlabel("epoch")
    plt.ylabel("TPR")
    plt.legend()
    plt.grid()

    plt.subplot(2, 3, 6)
    plt.plot(np.arange(start=1, stop=len(fpr_test) + 1, step=1), fpr_test, 'o-', label='test')
    plt.plot(np.arange(start=1, stop=len(fpr_train) + 1, step=1), fpr_train, 'o-', label='train')
    plt.xlim((1, len(fpr_test) + 1))
    plt.ylim((0, max(max(fpr_test), 1)))
    plt.title("FPR change")
    plt.xlabel("epoch")
    plt.ylabel("FPR")
    plt.legend()
    plt.grid()

    clear_output(True)

    plt.show()
