import numpy as np
from config_reader import Config

def label_data(cyclone_events):
    cfg = Config()
    labels = np.zeros(113960)
    events = []
    event_count = 0
    in_event = False
    temp_arr = []

    if (cfg.mode == 'recognize'):
        for i in range(113960):
            if len(cyclone_events[:, :, i][cyclone_events[:, :, i] != False]) > 0:
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
        for i in range(w, 113960):
            if len(cyclone_events[:, :, i][cyclone_events[:, :, i] != False]) > 0 and \
            not len(cyclone_events[:, :, i - w][cyclone_events[:, :, i - w] != False]) > 0:
                labels[i - w] = 1
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