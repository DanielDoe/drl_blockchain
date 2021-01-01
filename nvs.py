import numpy as np


# slice resource allocation/reservation by NVS
def slice_ra(rate_demands, association_ues, slice_num, bs_num):
    r_sk = np.zeros([slice_num, bs_num])
    total_rate_demand = 0
    slice_ue_num = np.zeros(slice_num)
    for si in range(slice_num):
        slice_ue_num[si] = association_ues[association_ues['slice_id'] == si].shape[0]

    for si in range(slice_num):
        total_rate_demand += rate_demands[si] * slice_ue_num[si]

    for si in range(slice_num):
        r_sk[si, :] = rate_demands[si] * slice_ue_num[si] / total_rate_demand

    return r_sk

