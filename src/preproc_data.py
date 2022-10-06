import numpy as np

class CycloneEvents:
    def __init__(self, cyclone_events):
        super().__init__()
        self.ce2 = cyclone_events[cyclone_events.files[0]]
        self.ce4 = cyclone_events[cyclone_events.files[1]]
        self.ce6 = cyclone_events[cyclone_events.files[2]]
        self.ce8 = cyclone_events[cyclone_events.files[3]]
        self.ce10 = cyclone_events[cyclone_events.files[4]]
        self.ce12 = cyclone_events[cyclone_events.files[5]]

def preproc_data(cyclone_events, metrics):
    ce = CycloneEvents(cyclone_events)

    metrics = np.reshape(metrics, (36, 69, 113960))
    data = metrics.copy().astype(np.float32)
    data = -np.log(1 - data + 1e-10)

    return ce, data, metrics