import numpy as np
from config_reader import Config
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
    cfg = Config()
    ce = CycloneEvents(cyclone_events)
    

    less = ['network_metrics/LCC', 'network_metrics/LCC_w', 'network_metrics/closeness_w', 'network_metrics/LCC_0.9',
             'network_metrics/LCC_0.95', 'diff_metrics/network_metrics/LCC', 'diff_metrics/network_metrics/LCC_w',
             'diff_metrics/network_metrics/LCC_0.9', 'diff_metrics/network_metrics/LCC_0.95',
             'diff_metrics/network_metrics/closeness_w']
    greater = ['network_metrics/degree', 'network_metrics/degree_w', 'network_metrics/EVC', 'network_metrics/EVC_w',
                'network_metrics/closeness', 'network_metrics/degree_0.9', 'network_metrics/EVC_0.9',
                'network_metrics/closeness_0.9', 'network_metrics/degree_0.95', 'network_metrics/EVC_0.95',
                'network_metrics/closeness_0.95', 'input_data/MSLP', 'input_data/MSLP_preproc', 'input_data/MSLP_land',
                'diff_metrics/input_data/MSLP', 'diff_metrics/input_data/MSLP_preproc', 'diff_metrics/input_data/MSLP_land',
                'diff_metrics/network_metrics/degree', 'diff_metrics/network_metrics/degree_w',
                'diff_metrics/network_metrics/EVC', 'diff_metrics/network_metrics/EVC_w',
                'diff_metrics/network_metrics/closeness', 'diff_metrics/network_metrics/degree_0.9',
                'diff_metrics/network_metrics/degree_0.95', 'diff_metrics/network_metrics/EVC_0.9',
                'diff_metrics/network_metrics/EVC_0.95', 'diff_metrics/network_metrics/closeness_0.9',
                'diff_metrics/network_metrics/closeness_0.95']

    metrics = np.reshape(metrics, (36, 69, 113960))
    data = metrics.copy().astype(np.float32)
    # expand to all
    np.nan_to_num(data, nan=0, copy=False)
    metric_name = cfg.metric_path.split('.')[0]
    if metric_name in greater:
        data = -np.log(1 - data + 1e-10)
    elif metric_name in less:
        data = -np.log(data + 1e-10)

    return ce, data, metrics