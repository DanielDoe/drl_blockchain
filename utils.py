# _*_ coding:utf-8 _*_

import numpy as np
import math
from scipy.spatial.distance import cdist
import datetime as dt


# generate RB throughput distribution at each ue, variance为方差
def generate_rb_thr(mean_throughput, variance, rows, cols):
    rb_thr_map = np.random.randn(rows, cols) * math.sqrt(variance) + mean_throughput
    return rb_thr_map


def compute_ue_thr(rb_thr_map, index_matrix):
    ue_thr = 0
    for row in index_matrix:
        ue_thr += rb_thr_map[row[0], row[1]]
    return ue_thr


# 计算回报
def compute_reward(slice_sat_ratio, slice_avg_RU):
    w1 = 5
    w2 = 1
    # reward = w1 * np.mean(slice_sat_ratio) + w2 * np.mean(slice_avg_RU)
    # if slice_sat_ratio[0] < 0.5 and slice_avg_RU[0] >= 0.9:
    #     reward = -1
    # else:
    #     reward = w1 * slice_sat_ratio[0] + w2 * slice_avg_RU[0]  # 资源利用率越高，回报越高
    reward = np.zeros(5)
    for si in range(4):
        # reward[si] = (slice_sat_ratio[si] - 0.5) + (slice_avg_RU[si] - 0.5)
        reward[si] = (slice_sat_ratio[si] - 0.5)

    # reward[3] = slice_sat_ratio[3] - 0.5  # ULL切片的reward
    reward[4] = np.sum(reward[0:4])  # overall reward
    return reward


# 计算RB形状 ue_info['slice_id', 'bs_id', 'ue_id', 'ue_x', 'ue_y']
def compute_rb_num(ue_info, TPs, user_qos):
    # -------------RB initial calculation------------
    Pt = 30  # TP 's transmitting power in dBm
    TP_subchannel = 25
    TP_freq = 2400  # TP frequence
    TP_Bndw = 5000000  # bandwidth in Hz
    sub_frame = 10  # the number of subFrame
    Nsd = -192.5  # noise spectral density in dBm/Hz
    # -------------resource block data------------
    subfrm_len = 1  # subframe length in mille seconds
    subch_Pt = Pt / TP_subchannel
    RB_Bndw = TP_Bndw / TP_subchannel
    Nsd_mW = math.pow(10, (Nsd / 10))  # --in mW--convert in =
    Nsd_subch = Nsd_mW * (TP_Bndw / TP_subchannel)
    NoB_subch = 10 * math.log10(Nsd_subch)

    sliceId = ue_info[0]
    bsId = ue_info[1]
    ueId = ue_info[2]
    ue_x = ue_info[3]
    ue_y = ue_info[4]
    # 计算与BS的距离，cdist 得到的结果为二维数组，并且需要传入的两个参数也是二维数组
    distance = cdist(np.array([[ue_x, ue_y]]), np.array(TPs.loc[bsId, ['x', 'y']]).reshape(1, 2),
                     metric='euclidean')
    # 计算路径损耗
    path_loss = (20 * math.log(distance[0, 0])) + (20 * math.log(TP_freq)) - 27.55
    rij = subch_Pt - path_loss - np.random.randint(1, 10)  # rssi nith nic of ith UE to jth AP
    RB_Thr = (RB_Bndw / sub_frame) * math.log2(1 + (rij / NoB_subch))  # bps
    # 根据速率要求预估出大概需要占用的RB数量，此处用的是平均SINR
    n_uh = math.ceil(sub_frame * subfrm_len / user_qos[1][sliceId])
    n_uv = math.ceil(user_qos[0][sliceId] * 1000 / (RB_Thr * n_uh))

    # 进行形状转换
    if n_uv > TP_subchannel:
        n_uh = math.ceil(n_uh * n_uv / TP_subchannel)
        n_uv = TP_subchannel
    return n_uh, n_uv, RB_Thr


# 返回当前时间
def cur_time():
    t = dt.datetime.now()
    return dt.datetime.strftime(t, '%Y%m%d_%H%M%S')


# 10进制转化为任意进制
def convert(num, b):
    chas = '0123456789abcdefghijklmnopqrstuvwxyz'
    return ((num == 0) and "0") or (convert(num // b, b).lstrip("0") + chas[num % b])


# 10进制转化为固定位数的任意进制数，不够补0
def convert_fix_bit(num, b, bit_len):
    res = str(convert(num, b))
    if len(res) != bit_len:
        arr = ['0' for i in range(bit_len)]
        start = bit_len - len(res)
        arr[start:] = res
        return ''.join(arr)
    return res

