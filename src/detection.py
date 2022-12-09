from cmath import nan
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import numpy as np
import os
from config_reader import Config

class DisplayData:
    def __init__(self, test_mx, train_mx):
        self.tp = test_mx[0, 0] + train_mx[0, 0]
        self.fp = test_mx[0, 1] + train_mx[0, 1]
        self.fn = test_mx[1, 0] + train_mx[1, 0]
        self.tn = test_mx[1, 1] + train_mx[1, 1]
        self.fpr = self.fp / (self.fp + self.tn)
        self.tpr = self.tp / (self.tp + self.fn)

def map_ids_to_color(train_id, test_id, results_train, results_test):
    tp_train = train_id[results_train[0]]
    fp_train = train_id[results_train[1]]
    fn_train = train_id[results_train[2]]
    tn_train = train_id[results_train[3]]

    tp_test = test_id[results_test[0]]
    fp_test = test_id[results_test[1]]
    fn_test = test_id[results_test[2]]
    tn_test = test_id[results_test[3]]

    image_data = np.empty(114080)
    for i in range(113960):
        if i in tp_train or i in tp_test:
            image_data[i] =  1
        elif i in fp_train or i in fp_test:
            image_data[i] =  2
        elif i in fn_train or i in fn_test:
            image_data[i] =  3
        elif i in tn_train or i in tn_test:
            image_data[i] =  0
        else:
            image_data[i] = 4

    for i in range(113960, 114080):
        image_data[i] = nan

    image_data_int = np.array([int(i) if not pd.isna(i) else nan for i in image_data])
    image_data_mx = image_data_int.reshape((460, 248))

    return image_data_mx, image_data_int

def days_in_month(month, year):
    thirty_one_days = [1, 3, 5, 7, 8, 10, 12]
    thiry_days = [4, 6, 9, 11]
    feb = [2]
    if month in thirty_one_days:
        return 31
    elif month in thiry_days:
        return 30
    elif month in feb and is_leap(year):
        return 29
    elif month in feb and not is_leap(year):
        return 28

def is_leap(year):
    is_leap = (year % 400 == 0) or \
     ((year % 100 != 0) and \
     (year % 4 == 0))
    
    return is_leap

def prepare_image_shape():
    year = 1982
    month_ticks = np.zeros((460, 248), dtype=int)
    for i, month in enumerate(month_ticks):
        if (i+1) % 12 == 0:
            year += 1
        days = days_in_month((i % 12) + 1, year)
        for j in range(days * 8, 31 * 8):
            month_ticks[i][j] = 4
    return month_ticks


def map_ids_to_color_by_day(train_id, test_id, results_train, results_test):
    tp_train = train_id[results_train[0]]
    fp_train = train_id[results_train[1]]
    fn_train = train_id[results_train[2]]
    tn_train = train_id[results_train[3]]

    tp_test = test_id[results_test[0]]
    fp_test = test_id[results_test[1]]
    fn_test = test_id[results_test[2]]
    tn_test = test_id[results_test[3]]

    month_ticks = prepare_image_shape()
    tick = 0
    for k, month in enumerate(month_ticks):
        for i, day_tick in enumerate(month):
            if month_ticks[k][i] == 4:
                break
            if tick in tp_train or tick in tp_test:
                month_ticks[k][i] =  1
            elif tick in fp_train or tick in fp_test:
                month_ticks[k][i]  =  2
            elif tick in fn_train or tick in fn_test:
                month_ticks[k][i]  =  3
            elif tick in tn_train or tick in tn_test:
                month_ticks[k][i]  =  0
            tick += 1

    return month_ticks

def draw_colormesh_by_month(month_ticks, save=False, filename='', display_data=None):
    cfg = Config()
    plt.rc('font', size=5)

    sns_cmap = sns.color_palette("Set1")
    tn_color = 'white'
    tp_color = sns_cmap[2]
    fp_color = sns_cmap[1]
    fn_color = sns_cmap[0]
    no_day_color = 'grey'
    else_color = sns_cmap[-1]

    cmap = clr.ListedColormap([tn_color, tp_color, fp_color, fn_color, no_day_color, else_color])
    

    handles = []
    if cfg.mode == 'recognize':
        tn_patch = mpatches.Patch(color=tn_color, label=f'True Negative (There is no cyclone and we did not detect anything). Count: {display_data.tn}')
        handles.append(tn_patch)
        tp_patch = mpatches.Patch(color=tp_color, label=f'True Positive (There is a cyclone and we detected it). Count: {display_data.tp}')
        handles.append(tp_patch)
        fp_patch = mpatches.Patch(color=fp_color, label=f'False Positive (There is no cyclone but we did detect it). Count: {display_data.fp}')
        handles.append(fp_patch)
        fn_patch = mpatches.Patch(color=fn_color, label=f'False Negative (There is a cyclone but we did not detect it). Count: {display_data.fn}')
        handles.append(fn_patch)
        no_day_patch = mpatches.Patch(color=no_day_color, label=f'No such day in this month.')
        handles.append(no_day_patch)
        cmap = clr.ListedColormap([tn_color, tp_color, fp_color, fn_color, no_day_color])
    elif cfg.mode == 'predict':
        tn_patch = mpatches.Patch(color=tn_color, label=f'True Negative (There is no cyclone and we did not detect anything). Count: {display_data.tn}')
        handles.append(tn_patch)
        tp_patch = mpatches.Patch(color=tp_color, label=f'True Positive (There is a cyclone and we detected it). Count: {display_data.tp}')
        handles.append(tp_patch)
        fp_patch = mpatches.Patch(color=fp_color, label=f'False Positive (There is no cyclone but we did detect it). Count: {display_data.fp}')
        handles.append(fp_patch)
        fn_patch = mpatches.Patch(color=fn_color, label=f'False Negative (There is a cyclone but we did not detect it). Count: {display_data.fn}')
        handles.append(fn_patch)
        no_day_patch = mpatches.Patch(color=no_day_color, label=f'No such day in this month.')
        handles.append(no_day_patch)
        else_patch = mpatches.Patch(color=else_color, label=f'Cyclones themselves')
        handles.append(else_patch)
        cmap = clr.ListedColormap([tn_color, tp_color, fp_color, fn_color, no_day_color, else_color])
    fig, ax = plt.subplots(1, dpi=1200)
    ax.set_title("Cyclone detection by month. FPR = {:.4f} TPR = {:.4f}".format(display_data.fpr, display_data.tpr))
    ax.set_xlabel("year")
    ax.set_ylabel("day")

    ax.pcolormesh(month_ticks.T, cmap=cmap, vmin=0, vmax=4)
    
    ax.set_xlim(0, 460)
    # 240 ticks in a month. 1 day is 8 ticks, 30 days in month
    ax.set_ylim(0, 248)

    ax.set_xticks(np.arange(0, 460, 24))
    ax.set_yticks(np.arange(0, 256, 8))
    ax.set_yticklabels(np.arange(0, 32, 1))
    ax.set_xticklabels(np.arange(1982, 2022, 2))
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.15,
                    box.width, box.height * 0.9])

    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.1))
    if save:
        path = '../images/'  + cfg.mode + '/' + cfg.metric + '/bymonth/' 
        if not os.path.exists(path):
            os.makedirs(path)
        fig.savefig(path + filename + '.png', transparent=False, \
                facecolor='white', edgecolor='white')
        plt.close()
    else:
        fig.show()


def draw_colormesh_by_tick(image_data_mx, save=False, filename='', display_data=None):
    cfg = Config()
    plt.rc('font', size=5)

    sns_cmap = sns.color_palette("Set1")
    tn_color = 'white'
    tp_color = sns_cmap[2]
    fp_color = sns_cmap[1]
    fn_color = sns_cmap[0]
    else_color = sns_cmap[-1]

    cmap = clr.ListedColormap([tn_color, tp_color, fp_color, fn_color, else_color])
    
    handles = []

    if cfg.mode == 'recognize':
        tn_patch = mpatches.Patch(color=tn_color, label=f'True Negative (There is no cyclone and we did not detect anything). Count: {display_data.tn}')
        handles.append(tn_patch)
        tp_patch = mpatches.Patch(color=tp_color, label=f'True Positive (There is a cyclone and we detected it). Count: {display_data.tp}')
        handles.append(tp_patch)
        fp_patch = mpatches.Patch(color=fp_color, label=f'False Positive (There is no cyclone but we did detect it). Count: {display_data.fp}')
        handles.append(fp_patch)
        fn_patch = mpatches.Patch(color=fn_color, label=f'False Negative (There is a cyclone but we did not detect it). Count: {display_data.fn}')
        handles.append(fn_patch)
    elif cfg.mode == 'predict':
        tn_patch = mpatches.Patch(color=tn_color, label=f'True Negative (There is no cyclone and we did not detect anything). Count: {display_data.tn}')
        handles.append(tn_patch)
        tp_patch = mpatches.Patch(color=tp_color, label=f'True Positive (There is a cyclone and we detected it). Count: {display_data.tp}')
        handles.append(tp_patch)
        fp_patch = mpatches.Patch(color=fp_color, label=f'False Positive (There is no cyclone but we did detect it). Count: {display_data.fp}')
        handles.append(fp_patch)
        fn_patch = mpatches.Patch(color=fn_color, label=f'False Negative (There is a cyclone but we did not detect it). Count: {display_data.fn}')
        handles.append(fn_patch)
        else_patch = mpatches.Patch(color=else_color, label=f'Cyclones themselves')
        handles.append(else_patch)

    fig, ax = plt.subplots(1, dpi=1200)
    ax.set_title("Cyclone detection by tick. FPR = {:.4f} TPR = {:.4f}".format(display_data.fpr, display_data.tpr))
    ax.set_xlabel("tick")
    ax.set_ylabel("tick")

    ax.pcolormesh(image_data_mx.T, cmap=cmap, vmin=0, vmax=5)
    
    ax.set_xlim(0, 460)
    # 240 ticks in a month. 1 day is 8 ticks, 30 days in month
    ax.set_ylim(0, 248)

    ax.set_xticks(np.arange(0, 460, 24))
    ax.set_yticks(np.arange(0, 256, 8))
    ax.set_yticklabels(np.arange(0, 32, 1))

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.15,
                    box.width, box.height * 0.9])

    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.1))
    if save:
        path = '../images/'  + cfg.mode + '/' + cfg.metric + '/bytick/' 
        if not os.path.exists(path):
            os.makedirs(path)
        fig.savefig(path + filename + '.png', transparent=False, \
                facecolor='white', edgecolor='white')
        plt.close()
    else:
        fig.show()

def map_events_to_color(events, train_id, test_id, results_train, results_test):
    cfg = Config()
    if cfg.mode == 'predict':
        return np.empty(shape=1)
    tp_train = train_id[results_train[0]]
    fn_train = train_id[results_train[2]]

    tp_test = test_id[results_test[0]]
    fn_test = test_id[results_test[2]]

    # 145 is y max limit
    marked_events = np.full(shape=(len(events), 145), fill_value=nan)

    for i, event in enumerate(events):
        for j, tick in enumerate(event):
            if tick in tp_train or tick in tp_test:
                marked_events[i, j] =  1
            elif tick in fn_train or tick in fn_test:
                marked_events[i, j] =  2

    return np.array(marked_events)


def draw_colormesh_by_event(marked_events, save=False, filename='', display_data=None):
    cfg = Config()
    if cfg.mode == 'predict':
        return
    plt.rc('font', size=5)

    sns_cmap = sns.color_palette("Set1")
    tp_color = sns_cmap[2]
    fn_color = sns_cmap[0]

    cmap = clr.ListedColormap([tp_color, fn_color])


    fn_patch = mpatches.Patch(color=fn_color, label=f'False Negative (There is a cyclone but we did not detect it). Count: {display_data.fn}')
    tp_patch = mpatches.Patch(color=tp_color, label=f'True Positive (There is a cyclone and we detected it). Count: {display_data.tp}')

    fig, ax = plt.subplots(1, dpi=1200)
    ax.set_title("Cyclone detection by event. FPR = {:.4f} TPR = {:.4f}".format(display_data.fpr, display_data.tpr))
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

    ax.legend(handles=[tp_patch, fn_patch], loc='upper center', bbox_to_anchor=(0.5, -0.1))
    if save:
        path = '../images/' + cfg.mode + '/' + cfg.metric + '/byevent/' 
        if not os.path.exists(path):
            os.makedirs(path)
        fig.savefig(path + filename + '.png', transparent=False, \
                facecolor='white', edgecolor='white')
        plt.close()
    else:
        fig.show()