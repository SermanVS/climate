import numpy as np

def label_data(cyclone_events):
    labels = np.zeros(113960)
    events = []
    event_count = 0
    in_event = False
    temp_arr = []
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
    return labels, events