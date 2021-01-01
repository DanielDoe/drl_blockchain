import numpy as np
from scipy.optimize import minimize


# slice resource allocation/reservation by NetShare
def slice_ra(user_qos, association_ues, slice_num, bs_num, sub_channel_num, sub_frame_num):
    # 计算d_jb矩阵
    d_jb = np.zeros([slice_num, bs_num])
    # 非ULL切片
    for si in range(3):
        for bi in range(bs_num):
            ues = association_ues[(association_ues['slice_id'] == si) &
                                  (association_ues['bs_id'] == bi)]
            if ues.shape[0] > 0:
                d_jb[si, bi] = compute_resource_demand(ues, user_qos, sub_channel_num, sub_frame_num)

    # ULL切片
    for bi in range(bs_num):
        ues = association_ues[(association_ues['slice_id'] == 3) &
                              (association_ues['bs_id'] == bi)]
        for index, row in ues.iterrows():
            RB_Thr = row['rb_thr']
            Ri = sub_channel_num * sub_frame_num * RB_Thr / 1000  # /kbps
            d_jb[3, bi] += user_qos[0, 3] / Ri
    slice_ue_num = np.zeros(slice_num)
    for si in range(slice_num):
        slice_ue_num[si] = association_ues[association_ues['slice_id'] == si].shape[0]

    # 计算Lj
    Lj = np.zeros([slice_num, bs_num])
    total_rate_demand = 0
    sigma = 1
    for si in range(slice_num):
        if si != 3:
            # kbps ==> bps
            total_rate_demand += user_qos[0, si] * slice_ue_num[si] * 1000
        else:
            # ULL : 1 / d + lamda
            total_rate_demand += sigma * (1 / user_qos[1, si] * 1000 + user_qos[2, si]) * \
                                 user_qos[3, si] * slice_ue_num[si]

    for si in range(slice_num):
        if si != 3:
            Lj[si, :] = user_qos[0, si] * slice_ue_num[si] * 1000 / total_rate_demand
        else:
            Lj[si, :] = sigma * (1 / user_qos[1, si] * 1000 + user_qos[2, si]) * \
                        user_qos[3, si] * slice_ue_num[si] / total_rate_demand
    frb = 1
    alpha = 0.2
    args = (Lj, frb, slice_num, bs_num, alpha)
    cons = con(args)
    # 设置初始猜测值
    x0 = Lj.flatten()
    res = minimize(objective(d_jb), x0, method='SLSQP', constraints=cons)
    r_sk = np.ones([4, 4]) * 0.1
    if res.success:
        r_sk = res.x.reshape([4, 4])
    return r_sk, res.success


# ues['slice_id', 'bs_id', 'ue_id', 'ue_x', 'ue_y', 'n_uh', 'n_uv', 'rb_thr', 'ue_rate']
def compute_resource_demand(ues, user_qos, sub_channel_num, sub_frame_num):
    d_jb = 0
    beta = 1
    slice_id = ues.iloc[0]['slice_id']
    min_rate = min(beta * user_qos[2, slice_id] * user_qos[3, slice_id],
                   user_qos[0, slice_id] * 1000)
    for index, row in ues.iterrows():
        RB_Thr = row['rb_thr']
        Ri = sub_channel_num * sub_frame_num * RB_Thr
        d_jb += min_rate / Ri
    return d_jb


# objective function
def objective(d_jb):
    return lambda x: -np.sum(d_jb.flatten() * np.log10(x))


# constraint conditions
def con(args):
    Lj, frb, slice_num, bs_num, alpha = args
    con1 = lambda t_jb: frb - (t_jb[0] + t_jb[4] + t_jb[8] + t_jb[12])
    con2 = lambda t_jb: frb - (t_jb[1] + t_jb[5] + t_jb[9] + t_jb[13])
    con3 = lambda t_jb: frb - (t_jb[2] + t_jb[6] + t_jb[10] + t_jb[14])
    con4 = lambda t_jb: frb - (t_jb[3] + t_jb[7] + t_jb[11] + t_jb[15])

    L_j = bs_num * Lj[:, 0]
    con5 = lambda t_jb: np.linalg.norm(np.array([np.sum(t_jb[0:4]) - L_j[0],
                                                 np.sum(t_jb[4:8]) - L_j[1],
                                                 np.sum(t_jb[8:12]) - L_j[2],
                                                 np.sum(t_jb[12:16]) - L_j[3]]), ord=1)
    cons = ({'type': 'ineq', 'fun': con1},
            {'type': 'ineq', 'fun': con2},
            {'type': 'ineq', 'fun': con3},
            {'type': 'ineq', 'fun': con4},
            {'type': 'eq', 'fun': con5},
            {'type': 'ineq', 'fun': lambda t_jb: t_jb - alpha * Lj.flatten()},
            {'type': 'ineq', 'fun': lambda t_jb: np.ones([slice_num, bs_num]).flatten() - t_jb})
    return cons






