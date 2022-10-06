from cmath import nan
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import numpy as np
import os

def map_ids_to_color(train_id, test_id, results_train, results_test):
    tp_train = train_id[results_train[0]]
    fp_train = train_id[results_train[1]]
    fn_train = train_id[results_train[2]]
    tn_train = train_id[results_train[3]]

    tp_test = test_id[results_test[0]]
    fp_test = test_id[results_test[1]]
    fn_test = test_id[results_test[2]]
    tn_test = test_id[results_test[3]]

    image_data = np.empty(114000)
    for i in range(113960):
        if i in tp_train or i in tp_test:
            image_data[i] =  1
        elif i in fp_train or i in fp_test:
            image_data[i] =  2
        elif i in fn_train or i in fn_test:
            image_data[i] =  3
        elif i in tn_train or i in tn_test:
            image_data[i] =  0

    for i in range(113960, 114000):
        image_data[i] = nan

    image_data_int = np.array([int(i) if not pd.isna(i) else nan for i in image_data])
    image_data_mx = image_data_int.reshape((475, 240))

    return image_data_mx, image_data_int



def draw_colormesh_by_tick(image_data_mx, save=False, filename=''):
    plt.rc('font', size=5)

    sns_cmap = sns.color_palette("Set1")
    tn_color = 'white'
    tp_color = sns_cmap[2]
    fp_color = sns_cmap[1]
    fn_color = sns_cmap[0]

    cmap = clr.ListedColormap([tn_color, tp_color, fp_color, fn_color])


    fn_patch = mpatches.Patch(color=fn_color, label='False Negative (There is a cyclone but we did not detect it)')
    tn_patch = mpatches.Patch(color=tn_color, label='True Negative (There is no cyclone and we did not detect anything)')
    tp_patch = mpatches.Patch(color=tp_color, label='True Positive (There is a cyclone and we detected it)')
    fp_patch = mpatches.Patch(color=fp_color, label='False Positive (There is no cyclone but we did detected it)')

    fig, ax = plt.subplots(1, dpi=1200)
    ax.set_title("Cyclone detection by tick")
    ax.set_xlabel("month")
    ax.set_ylabel("day")

    ax.pcolormesh(image_data_mx.T, cmap=cmap, vmin=0, vmax=3)

    ax.set_xlim(0, 495)
    ax.set_ylim(0, 248)

    ax.set_xticks(np.arange(0, 495, 15))
    ax.set_yticks(np.arange(0, 248, 8))
    ax.set_yticklabels(np.arange(0, 31, 1))

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                    box.width, box.height * 0.9])

    ax.legend(handles=[fn_patch, tn_patch, tp_patch, fp_patch], loc='upper center', bbox_to_anchor=(0.5, -0.05))
    if save:
        path = '../images/MSLP/bytick/' 
        if not os.path.exists(path):
            os.makedirs(path)
        fig.savefig(path + filename + '.png', transparent=False, \
                facecolor='white', edgecolor='white')
        plt.close()
    else:
        fig.show()

def map_events_to_color(events, train_id, test_id, results_train, results_test):
    tp_train = train_id[results_train[0]]
    fn_train = train_id[results_train[2]]

    tp_test = test_id[results_test[0]]
    fn_test = test_id[results_test[2]]

    marked_events = np.full(shape=(len(events), 145), fill_value=nan)

    for i, event in enumerate(events):
        for j, tick in enumerate(event):
            if tick in tp_train or tick in tp_test:
                marked_events[i, j] =  1
            elif tick in fn_train or tick in fn_test:
                marked_events[i, j] =  2

    return np.array(marked_events)


def draw_colormesh_by_event(marked_events, save=False, filename=''):
    plt.rc('font', size=5)

    sns_cmap = sns.color_palette("Set1")
    tp_color = sns_cmap[2]
    fn_color = sns_cmap[0]

    cmap = clr.ListedColormap([tp_color, fn_color])


    fn_patch = mpatches.Patch(color=fn_color, label='False Negative (There is a cyclone but we did not detect it)')
    tp_patch = mpatches.Patch(color=tp_color, label='True Positive (There is a cyclone and we detected it)')

    fig, ax = plt.subplots(1, dpi=1200)
    ax.set_title("Cyclone detection by event")
    ax.set_xlabel("event")
    ax.set_ylabel("tick")

    ax.pcolormesh(marked_events.T, cmap=cmap, vmin=0, vmax=3)

    ax.set_xticks(np.arange(0, len(marked_events), 10))
    ax.set_yticks(np.arange(0, 145, 5))

    ax.set_xlim(0, len(marked_events))
    ax.set_ylim(0, 145)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                    box.width, box.height * 0.9])

    ax.legend(handles=[tp_patch, fn_patch], loc='upper center', bbox_to_anchor=(0.5, -0.05))
    if save:
        path = '../images/MSLP/byevent/' 
        if not os.path.exists(path):
            os.makedirs(path)
        fig.savefig(path + filename + '.png', transparent=False, \
                facecolor='white', edgecolor='white')
        plt.close()
    else:
        fig.show()