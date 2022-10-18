import numpy as np
from config_reader import Config

def read_data(path):
    cfg = Config()
    cyclone_events = np.load(path + '/ERA5/ERA5_MSL_1982_2020_3h_0.75/cyclones_events.npz')

    dir_path = '/ERA5/ERA5_MSL_1982_2020_3h_0.75/metrics_corr_land_masked_and_preproc_window_2d_delay_0d/probability_for_metrics/'

    #closeness_w = np.load(path + dir_path + 'network_metrics/closeness_w.npy')
    #degree_w = np.load(path + dir_path + 'network_metrics/degree_w.npy')
    #EVC_w = np.load(path + dir_path + 'network_metrics/EVC_w.npy')
    #LCC_w = np.load(path + dir_path + 'network_metrics/LCC_w.npy')
    #MSLP_preproc = np.load(path + dir_path + 'input_data/MSLP_preproc.npy')

    data = np.load(path + dir_path + cfg.metric_path)
    return cyclone_events, data