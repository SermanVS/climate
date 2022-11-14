import numpy as np
from config_reader import Config

'''
Maybe create a class and make cyclone_events accessible
'''
def is_cyclone_present(cyclone_events, tick):
    return cyclone_events[:, :, tick].sum() > 0

def is_cyclone_expected(cyclone_events, tick, w):
    for i in range(w):
        try:
            if is_cyclone_present(cyclone_events, tick + i) \
                and not is_cyclone_present(cyclone_events, tick):
                return True
        except:
            break
    return False

def label_data(cyclone_events):
    cfg = Config()
    ticks = cyclone_events.shape[2]
    labels = np.zeros(ticks)
    events = []
    event_count = 0
    in_event = False
    temp_arr = []

    if (cfg.mode == 'recognize'):
        for i in range(ticks):
            if is_cyclone_present(cyclone_events, i):
                labels[i] = 1
                if not in_event:
                    in_event = True
                    event_count += 1
                temp_arr.append(i)
            else:
                in_event = False
                if len(temp_arr) > 0:
                    events.append(temp_arr)
                    # don't use .clear()
                    temp_arr = []
    elif (cfg.mode == 'predict'):
        w = 8
        for i in range(ticks):
            if is_cyclone_expected(cyclone_events, i, w):
                labels[i] = 1
            if is_cyclone_present(cyclone_events, i):    
                labels[i] = 2
                if not in_event:
                    in_event = True
                    event_count += 1
                temp_arr.append(i)
            else:
                in_event = False
                if len(temp_arr) > 0:
                    events.append(temp_arr)
                    # don't use .clear()
                    temp_arr = []
    return labels, events