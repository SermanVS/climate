import numpy as np

def read_data(path):
    cyclone_events = np.load(path + '/ERA5/ERA5_MSL_1982_2020_3h_0.75/cyclones_events.npz')
    closeness_w = np.load(path + '/ERA5/ERA5_MSL_1982_2020_3h_0.75/metrics_corr_land_masked_and_preproc_window_2d_delay_0d/probability_for_metrics/diff_metrics/network_metrics/closeness_w.npy')
    degree_w = np.load(path + '/ERA5/ERA5_MSL_1982_2020_3h_0.75/metrics_corr_land_masked_and_preproc_window_2d_delay_0d/probability_for_metrics/diff_metrics/network_metrics/degree_w.npy')
    EVC_w = np.load(path + '/ERA5/ERA5_MSL_1982_2020_3h_0.75/metrics_corr_land_masked_and_preproc_window_2d_delay_0d/probability_for_metrics/diff_metrics/network_metrics/EVC_w.npy')
    LCC_w = np.load(path + '/ERA5/ERA5_MSL_1982_2020_3h_0.75/metrics_corr_land_masked_and_preproc_window_2d_delay_0d/probability_for_metrics/diff_metrics/network_metrics/LCC_w.npy')
    MSLP_preproc = np.load(path + '/ERA5/ERA5_MSL_1982_2020_3h_0.75/metrics_corr_land_masked_and_preproc_window_2d_delay_0d/probability_for_metrics/input_data/MSLP_preproc.npy')
    return cyclone_events, closeness_w, degree_w, EVC_w, LCC_w, MSLP_preproc