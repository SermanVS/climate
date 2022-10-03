import numpy as np
from scipy.stats import chi2_contingency

def perform_g_test(contigency_table):
    if ((contigency_table[0, 0] == 0) and (contigency_table[0, 1] == 0)) or ((contigency_table[1, 0] == 0) and (contigency_table[1, 1] == 0)):
        I_stat = g_stat = 0
        p_val = np.nan
    else:
        g_stat, p_val, dof, expctd = chi2_contingency(contigency_table, lambda_="log-likelihood", correction=False)
        I_stat = g_stat / contigency_table.flatten().sum() / 2
    return g_stat, p_val, I_stat