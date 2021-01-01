# _*_ coding:utf-8 _*_

import numpy as np
import pandas as pd
import math
from scipy.spatial.distance import cdist
import utils


class SimulationEnv(object):

    bs_num = 4  # bs数量
    slice_num = 4  # 切片数量
    sys_bandwidth = 5000000  # 系统带宽 HZ
    sub_channel_num = 25  # 子信道数量
    sub_frame_num = 10  # 子帧数量

    # slice parameters configuration
    rate_demands = [100, 500, 12, 10]  # / kbps
    # rate_demands = [10, 25, 7, 9]  # / kbps (0.8 * lamda)
    delay_demands = [100, 120, 105, 10]  # / ms
    lamda_k = [100, 100, 100, 100]  # packets per second
    packet_size = [400, 4000, 500, 120]  # bit
    user_qos = [rate_demands, delay_demands, lamda_k, packet_size]
    slice_user_seq = []  # 用户发送packet的子帧下标序列
    slice_ue_num = []  # 切片的用户数量

    # -------rate_demands--delay_demands---lamda_k---packet_size---------
    # slice0: [10kpbs, 100ms, 30packet/s, 400bit]  UEb slice
    # slice1: [500kpbs, 120ms, 20packet/s, 1000bit]  HDTV slice
    # slice2: [12kpbs, 105ms, 30packet/s, 500bit]  MIoT slice
    # slice3: [10kpbs, 10ms, 100packet/s, 120bit]  ULL slice

    # 产生UE信息
    def generate_ue(self, ue_num):
        columns_ue = ['ue_x', 'ue_y', 'rate_demand', 'delay_demand', 'slice_id', 'ue_id']
        ue_df = pd.DataFrame(index=np.arange(sum(ue_num)),
                             columns=columns_ue)
        ue_id = 0
        for si in range(self.slice_num):
            for ui in range(ue_num[si]):
                deta = np.random.randint(0, 360)
                r = np.random.randint(0, 80)
                ue_X = 350 + r * math.cos(math.radians(deta))
                ue_Y = 350 + r * math.sin(math.radians(deta))
                ue_df.loc[ue_id] = [ue_X, ue_Y, self.rate_demands[si], self.delay_demands[si], si, ue_id]
                ue_id += 1
        self.slice_ue_num = ue_num  # 初始化用户数量
        return ue_df

    # 产生BS信息
    def generate_bs(self, bs_num):
        teta = 0
        r = 120
        columns_bs = ['x', 'y', 'backhaul']
        TPs = pd.DataFrame(index=np.arange(bs_num), columns=columns_bs)
        for bi in range(bs_num):
            x = 350 + r * math.cos(math.radians(teta))
            y = 350 + r * math.sin(math.radians(teta))
            teta += 90
            backhaul = np.random.randint(30, 40) * 1e6  # backhaul data rate /bps
            TPs.loc[bi] = [x, y, backhaul]
        return TPs

    # 用户关联（ue--bs）
    def ue_association(self, admission_ues, TPs, mi=3):
        columns_ue = ['slice_id', 'bs_id', 'ue_id', 'ue_x', 'ue_y', 'n_uh', 'n_uv', 'rb_thr', 'ue_rate']
        ue_num = admission_ues.shape[0]
        ue_df = pd.DataFrame(index=np.arange(ue_num),
                             columns=columns_ue)
        rb_size = np.zeros(self.slice_num)  # 每类切片的用户的单个flow需要的RB最大数量
        if mi == 3:  # 基于DQN或DDPG的方法：就近关联+负载均衡
            for si in range(self.slice_num):
                slice_ues = admission_ues[admission_ues['slice_id'] == si]  # 获取该类切片的用户
                su_num = slice_ues.shape[0]  # 该类切片的总用户数量
                avg_num = math.floor(self.slice_ue_num[si] / self.bs_num)
                sb_num = np.zeros(self.bs_num)  # 在各个BS上的用户数量
                for index, row in slice_ues.iterrows():
                    sliceId = row['slice_id']
                    ue_x = row['ue_x']
                    ue_y = row['ue_y']
                    ueId = row['ue_id']
                    bsId = self.find_bs([ue_x, ue_y], TPs, sb_num, avg_num)
                    n_uh, n_uv, RB_Thr = utils.compute_rb_num(ue_info=[sliceId, bsId, ueId, ue_x, ue_y], TPs=TPs,
                                                              user_qos=self.user_qos)  # /bps
                    ue_rate = math.floor(RB_Thr / 1e3)  # 用户速率 /kbps
                    ue_df.loc[index] = [sliceId, bsId, ueId, ue_x, ue_y, n_uh, n_uv, RB_Thr, ue_rate]
                    sb_num[bsId] += 1
                    rb_size[si] = max(rb_size[si], n_uh * n_uv)
        else:  # NVS或NetShare：就近关联
            for index, row in admission_ues.iterrows():
                sliceId = row['slice_id']
                ue_x = row['ue_x']
                ue_y = row['ue_y']
                ueId = row['ue_id']
                TP = np.array(TPs.loc[:, ['x', 'y']])  # BS 信息
                distance = cdist([[ue_x, ue_y]], TP, metric='euclidean')  # 计算该 UE 与 所有 BS 之间的距离
                # print(distance)
                bsId = distance[0].argmin()  # 找距离最近的BS
                n_uh, n_uv, RB_Thr = utils.compute_rb_num(ue_info=[sliceId, bsId, ueId, ue_x, ue_y], TPs=TPs,
                                                          user_qos=self.user_qos)  # /bps
                ue_rate = math.floor(RB_Thr / 1e3)  # 用户速率 /kbps
                ue_df.loc[index] = [sliceId, bsId, ueId, ue_x, ue_y, n_uh, n_uv, RB_Thr, ue_rate]
                rb_size[sliceId] = max(rb_size[sliceId], n_uh * n_uv)

        print('The number of users of slice on each bs:')
        for si in range(self.slice_num):
            sb_num = np.zeros(self.bs_num)
            for bi in range(self.bs_num):
                sb_num[bi] = ue_df[(ue_df['slice_id'] == si) & (ue_df['bs_id'] == bi)].shape[0]
            print('slice ', si, ' : ', sb_num[0], sb_num[1], sb_num[2], sb_num[3])
        # for index, row in admission_ues.iterrows():
        #     sliceId = row['slice_id']
        #     ue_x = row['ue_x']
        #     ue_y = row['ue_y']
        #     ueId = row['ue_id']
        #     TP = np.array(TPs.loc[:, ['x', 'y']])  # BS 信息
        #     distance = cdist([[ue_x, ue_y]], TP, metric='euclidean')  # 计算该 UE 与 所有 BS 之间的距离
        #     # print(distance)
        #     bsId = distance[0].argmin()  # 找距离最近的BS
        #     ue_df.loc[index] = [sliceId, bsId, ueId, ue_x, ue_y]
        return ue_df, rb_size

    # 寻找一个BS关联
    def find_bs(self, ue_info, TPs, sb_num, avg_num):
        ue_x, ue_y = ue_info[0], ue_info[1]
        TP = np.array(TPs.loc[:, ['x', 'y']])  # BS 信息
        distance = cdist([[ue_x, ue_y]], TP, metric='euclidean')
        bsId = -1
        bs_indexs = np.argsort(distance[0])  # 按照距离进行升序排序，并返回原位置的下标
        for bi in range(self.bs_num):
            if sb_num[bs_indexs[bi]] < avg_num:
                bsId = bs_indexs[bi]
                break
        # 如果每个基站的数量都达到了avgNum，则找一个最小的距离的基站
        if bsId == -1:
            # 重新找一个数量等于avg_num的基站
            for bi in range(self.bs_num):
                if sb_num[bs_indexs[bi]] == avg_num:
                    bsId = bs_indexs[bi]
                    break
            # bsId = bs_indexs[0]
        return bsId

    # 生成用户子帧下标 mi=0（DRL算法） mi=1（slice isolation） mi=2（buffer length）
    def generate_subframe_index(self, association_ues, lamda_k, data_num=70, mi=0):
        # 此处的 slice_num 在本实验中设定为4，0-2 为非ULL切片，3 为ULL切片（可以根据自己的需求设置） 
        #print(pd_association_ues.head())
        slice_user_seq = [np.array([]) for i in range(4)]
        ue_num = association_ues[0].shape[0]
        max_index = 2147483647
        for index, row in association_ues[0].iterrows():
            print('row value: {}'.format(row))
            sliceId = row['slice_id']
            ueId = row['ue_id']
            if sliceId != 3:
                if mi == 0:
                    # 通过均匀分布模拟UE发送packet，主要用在跑DRL算法的情况
                    data_num = int(lamda_k[sliceId] / 5)  # int(np.ceil(lamda_k[sliceId] / 5)) 改成了1s -> 得出buffer length
                    frame_index = np.linspace(1, 199, data_num)  # 均匀分布模拟数据流 np.linspace(1, 199, data_num)
                    frame_index = np.ceil(frame_index)
                else:
                    # 通过指数分布模拟UE发送packet，主要用在跑slice isolatoin 和 buffer length的情况
                    frame_index = np.random.exponential(1 / lamda_k[sliceId] * 1000, data_num)
                    frame_index = np.ceil(np.cumsum(frame_index))
                    frame_index = np.ceil(frame_index)
                # 需要产生的数据包个数，其中5 = 1s / 200ms(每一次slicing的时间)
                # data_num = int(lamda_k[sliceId])  # int(np.ceil(lamda_k[sliceId] / 5)) 改成了1s -> 得出buffer length
                # frame_index = np.linspace(1, 199, data_num)  # 均匀分布模拟数据流 np.linspace(1, 199, data_num)
                # frame_index = np.ceil(frame_index)
                max_index = min(max(frame_index), max_index)
                frame_index = np.concatenate((np.array([ueId]), frame_index), axis=0).reshape(1, data_num + 1)
                if slice_user_seq[sliceId].shape[0] == 0:
                    slice_user_seq[sliceId] = frame_index
                else:
                    slice_user_seq[sliceId] = np.concatenate((slice_user_seq[sliceId], frame_index), axis=0)
                # slice_user_seq[sliceId].append(frame_index.tolist())  # 注意将np.array 转化为 list
        # max_index = 1000  # 200
        if mi == 0 or mi == 1:  # DRL 或者 slice isolation
            max_index = 200
        else:  # buffer length (100 frames)
            max_index = 1000
        # ULL切片按周期性产生数据，可以设置间隔周期为10ms，数据包长度为100bits
        # 【Ultra-Reliable and Low-Latency Communication for Wireless Factory Automation: From LTE to 5G】
        ull_inter_time = int(1 / lamda_k[3] * 1000)
        slice_user_seq[3] = list(range(1, max_index, ull_inter_time))

        print('maxSubframeIndex : ', max_index)
        self.slice_user_seq = slice_user_seq  # 保存为类属性
        return slice_user_seq, max_index

    # 产生下行链路数据lastQueue[1...s]['bs_id', 'ue_id', 'ue_x', 'ue_y', 'n_uh', 'n_uv', 'rb_thr', 'subframe_index']
    def generate_downlink_data(self, last_queue, association_ues, cur_subframe):
        # queue = last_queue  # 将上一次未分配完的请求放入新的队列中
        columns = ['bs_id', 'ue_id', 'ue_x', 'ue_y', 'n_uh', 'n_uv', 'rb_thr', 'subframe_index']
        # 按照预先产生的子帧下标来产生下行链路数据
        for index, row in association_ues.iterrows():
            sliceId = row['slice_id']
            bsId = row['bs_id']
            ueId = row['ue_id']
            if sliceId != 3:
                seq = self.slice_user_seq[sliceId]
                seq = seq[seq[:, 0] == ueId]  # 找出该ueId的数据
                if seq.shape[0] != 0:   # 如果可以找到，则添加进队列
                    seq = seq[0, 1:]
                    if seq[seq == cur_subframe].shape[0] != 0:
                        data = row[1:-1].tolist()  # ['bs_id', 'ue_id', 'ue_x', 'ue_y', 'n_uh', 'n_uv', 'rb_thr']
                        data.append(cur_subframe)
                        data = [data]  # 注意初始化DataFrame的data属性必须是二维的list或者np.array
                        # queue[sliceId] = queue[sliceId].append(pd.DataFrame(data=data, columns=columns),
                        #                                        ignore_index=True)
                        # 使用np.array的合并操作可以极大的提升效率
                        last_queue[sliceId] = np.concatenate((last_queue[sliceId], np.array(data)), axis=0)
        # 返回队列
        # return queue

    # 产生ULL的packet队列  ull_ues['bs_id', 'ue_id', 'ue_x', 'ue_y', 'n_uh', 'n_uv', 'rb_thr', 'subframe_index']
    def generate_ull_data(self, ull_ues, cur_subframe):
        ue_num = ull_ues.shape[0]
        columns = ['bs_id', 'ue_id', 'ue_x', 'ue_y', 'subframe_index']
        # 预先分配一个[ue_num * 10, 5]大小的DataFrame
        # data_queue = pd.DataFrame(index=list(range(ue_num * 10)), columns=columns)
        # 使用 np.array 做扩容可以提高100倍左右的速度
        data_queue = np.zeros([ue_num * 10, 8])
        times, offset = 0, 0
        for fi in range(cur_subframe, cur_subframe + 10):
            if fi in self.slice_user_seq[3]:  # 如果存在该子帧下标，则添加进队列中
                row = np.array(ull_ues.loc[:, ['bs_id', 'ue_id', 'ue_x', 'ue_y', 'n_uh', 'n_uv', 'rb_thr']])
                row = np.c_[row, np.ones([row.shape[0], 1]) * fi]  # 给np.array添加一列
                # data_queue.loc[offset:(offset + ue_num - 1)] = row.tolist()  # 给DataFrame行列赋值可以通过list
                data_queue[offset:(offset + ue_num)] = row.tolist()  # np.array 的切片不包括第二个下标
                offset += ue_num  # 更新偏移量
                times += 1  # 更新计数器
        # 判断是否有packet要发送
        if times != 0:
            # data_queue = data_queue.loc[0:(times * ue_num - 1)]
            data_queue = data_queue[0:(times * ue_num)]
        else:
            # data_queue = pd.DataFrame()  # 直接清空队列
            data_queue = np.empty([0, 5])
        return data_queue


if __name__ == '__main__':
    # rate_demands = [200, 1000, 150, 10]  # / kbps
    # delay_demands = [100, 120, 105, 10]  # / ms
    # lamda_k = [10, 10, 10, 100]  # packets per second
    # packet_size = [400, 1000, 300, 120]  # bit
    # user_qos = [rate_demands, delay_demands, lamda_k, packet_size]
    # print(user_qos[0][2])
    env = SimulationEnv()
    ud = 10
    ues = env.generate_ue([130 + ud, 8 + math.floor(ud / 6), 220 + ud, 0 + ud])
    TPs = env.generate_bs(4)
    env.ue_association(ues, TPs)

