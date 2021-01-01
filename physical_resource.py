'''
numpy 中的 array 可以代替 matlab 中的矩阵，其中常用的函数np.where() 可以代替 matlab 中的 find()
pandas 中的 DataFrame 可以用来存储一些结构化的数据，比如：UE、BS、Packet等
matplotlib 可以和 numpy 结合使用，其主要功能是用来画图

例子：
x = np.zeros([4, 4])
y = np.ones([3, 4])

1、拼接两个数组：z = np.concatenate((x, y), axis=0)  # x,y必须都是二维array
2、删除数组的某一或多行：z = np.delete(z, [0, 2], axis=0)
3、查找数组第1列为3的数据：w = z[z[:, 1] == 3]

注意：pandas中的DataFrame扩容速度比numpy中的array要慢很多，所以推荐使用numpy中的 np.concatenate() 来拼接数组
'''

import numpy as np
import pandas as pd
import math
from env import SimulationEnv
import utils
#from memory_profiler import profile


class PhysicalResource(object):

    # env必须传入参数，不能使用默认值
    def __init__(self, TPs, user_qos, env=SimulationEnv()):
        self.env = env
        self.TPs = TPs
        self.user_qos = user_qos

    # 获取切片的调度顺序 ue['slice_id', 'bs_id', 'ue_id', 'ue_x', 'ue_y']
    def get_schedule_order(self, ues, TPs):
        slice_order = pd.DataFrame(index=list(range(3)), columns=['slice_id', 'n_uv'])
        for index, row in ues.iterrows():
            n_uh, n_uv, RB_thr = utils.compute_rb_num(ue_info=row, TPs=self.TPs, user_qos=self.user_qos)
            slice_order.loc[index] = [index, n_uv]
        # 按照n_uv进行降序排序
        slice_order.sort_values(by='n_uv', ascending=False)
        return slice_order['slice_id']  # 返回一列

    # 根据RB形状使每个切片在每个基站上的RB数量刚好为整数倍的n_uv * n_uh
    def get_shaped_resource_ratio(self, ue_info, r_sk, rb_size):
        sub_channel_num = self.env.sub_channel_num
        sub_frame_num = self.env.sub_frame_num
        # 每个切片在每个基站上的RB数量
        slice_bs_rb_num = np.zeros([4, 4])  # 此处为了方便，直接默认1个切片，4个基站

        # rb_size = np.zeros(4)
        # for i in range(4):
        #     n_uh, n_uv, RB_thr = utils.compute_rb_num(ue_info=ue_info.iloc[i], TPs=self.TPs, user_qos=self.user_qos)
        #     rb_size[i] = n_uh * n_uv
        for si in range(4):
            for bi in range(4):
                slice_bs_rb_num[si, bi] = math.floor(sub_channel_num * sub_frame_num
                                                     * r_sk[si, bi] / rb_size[si]) * rb_size[si]
        return slice_bs_rb_num

    # 物理资源分配---用于slice isolation
    # @profile(precision=4, stream=open('memory_profiler.log', 'w+'))
    def allocate_isolation(self, association_ues, r_sk, total_subframe, rb_size):
        cur_subframe = 1  # 当前子帧下标
        frame_index = 1  # 当前帧下标
        data_queue = [np.empty([0, 8]) for i in range(4)]  # 此处直接初始化4类切片的队列

        columns_ue = ['slice_id', 'bs_id', 'ue_id', 'ue_x', 'ue_y']
        slice_ues = pd.DataFrame(index=list(range(3)), columns=columns_ue)
        for index, row in slice_ues.iterrows():
            ue = association_ues[association_ues['slice_id'] == index].iloc[0][columns_ue]
            slice_ues.iloc[index] = ue
        # 根据n_uv确定切片的调度顺序（降序）
        slice_order = self.get_schedule_order(ues=slice_ues, TPs=self.TPs)

        # 根据RB形状使每个切片在每个基站上的RB数据量刚好为整数倍的n_uv * n_uh
        # slice_ues.iloc[0] = association_ues[association_ues['slice_id'] == 0].iloc[0][columns_ue].tolist()
        slice_ues = slice_ues.append(pd.DataFrame(
            data=[association_ues[association_ues['slice_id'] == 3].iloc[0][columns_ue].tolist()],
            columns=columns_ue),
            ignore_index=True)
        slice_bs_rb_num = self.get_shaped_resource_ratio(ue_info=slice_ues, r_sk=r_sk, rb_size=rb_size)

        ull_ues = association_ues[association_ues['slice_id'] == 3]  # 超低时延用户，在该实验中配置sliceId为3

        sub_channel_num = self.env.sub_channel_num
        sub_frame_num = self.env.sub_frame_num

        # last_ull_queue = pd.DataFrame(columns=['bs_id', 'ue_id', 'ue_x', 'ue_y', 'subframe_index'])  # 初始化一个空的packet队列
        last_ull_queue = np.empty([0, 8])

        # 用户满意度统计量
        columns_sat = ['slice_id', 'ue_id', 'subframe_index', 'ue_delay', 'sat_ratio', 'bs_id']
        # global_sat_ratio = pd.DataFrame(columns=columns_sat)
        global_sat_ratio = np.empty([0, 6])  # 新增bsId用于统计BS级的满意度
        # 切片资源利用率统计量
        columns_RU = ['slice_id', 'frame_index', 'slice_RU']
        # global_slice_RU = pd.DataFrame(columns=columns_RU)
        global_slice_RU = np.empty([0, 3])
        # BS资源利用率统计量 ['slice_id', 'bs_id', 'frame_index', 'slice_RU']
        global_bs_RU = np.empty([0, 4])
        # slice的buffer队列长度['slice_id', 'frame_index', 'queue_len']
        global_slice_queue_len = np.empty([0, 3])
        # ULL切片的满意度列表
        ull_sat_ratio_list = []

        while cur_subframe <= total_subframe:
            # 模拟packet发送，packet会被存放在队列中data_queue[1...s]['bs_id', 'ue_id', 'ue_x', 'ue_y', 'subframe_index']
            # ['bs_id', 'ue_id', 'ue_x', 'ue_y', 'n_uh', 'n_uv', 'rb_thr', 'subframe_index']
            self.env.generate_downlink_data(data_queue, association_ues, cur_subframe)

            # -----------------每一帧做一次物理资源分配-----------------------
            if cur_subframe % 10 != 0:
                cur_subframe += 1
                continue
            if cur_subframe == 200:
                print(cur_subframe)
            bs_throughput = np.zeros(4)  # 基站吞吐量
            bs_rb_map = [np.ones([sub_channel_num, sub_frame_num]) * -1 for i in range(4)]  # 基站RB的map

            # 对非ULL切片的用户分配资源
            for index in range(3):
                sliceId = slice_order[index]
                # slice在各个BS上已经使用的RB，此处直接初始化为4个BS，可以根据自己的需求更改
                used_rb = np.zeros(4)
                # slice在所有BS上已经使用的RB
                slice_used_rb = 0
                del_index = []  # 已经分配资源的packet下标
                unsat_packet_num = 0  # 未分配资源的packet数量

                queue_len = data_queue[sliceId].shape[0]
                if global_slice_queue_len.shape[0] == 0:
                    global_slice_queue_len = np.array([[sliceId, frame_index, queue_len]])
                else:
                    queue_len_row = np.array([[sliceId, frame_index, queue_len]])
                    # 统计slice的队列长度
                    global_slice_queue_len = np.concatenate((global_slice_queue_len, queue_len_row), axis=0)

                # 遍历该slice的数据队列
                for i2 in range(data_queue[sliceId].shape[0]):
                    row = data_queue[sliceId][i2]
                    ue_info = row[:-1].tolist()
                    ue_info.insert(0, sliceId)
                    # n_uh, n_uv, RB_thr = utils.compute_rb_num(ue_info=ue_info, TPs=self.TPs, user_qos=self.user_qos)
                    n_uh, n_uv, RB_thr = int(ue_info[5]), int(ue_info[6]), ue_info[7]  # 改成提前计算，无需每次重复计算
                    # 既要满足前程容量，又要满足回程容量
                    bsId = int(row[0])  # row['bs_id']
                    if (used_rb[bsId] + n_uh * n_uv > slice_bs_rb_num[sliceId, bsId]) \
                            or (bs_throughput[bsId] >= self.TPs.loc[bsId]['backhaul']):
                        continue

                    # ----------------物理资源映射------------------------
                    rb_thr_map = utils.generate_rb_thr(mean_throughput=RB_thr, variance=4,
                                                       rows=sub_channel_num, cols=sub_frame_num)
                    for offset in range(0, sub_frame_num + 2 - n_uh):
                        temp_matrix = bs_rb_map[bsId][:, offset:offset + n_uh]
                        loc_index = np.where(temp_matrix == -1)  # sliceId = {0, 1, 2, 3}

                        if loc_index[0].shape[0] < n_uh * n_uv:  # 再次判断BS上的资源是否足够
                            continue
                        length = n_uh * n_uv
                        ri = loc_index[0][0:length]
                        ci = loc_index[1][0:length]
                        temp_matrix[(ri, ci)] = sliceId  # 使bs_rb_map中符合条件的元素值设置为sliceId

                        # index_matrix(二维矩阵)：第0列为行下标，第一列为列下标
                        index_matrix = np.concatenate((ri.reshape(length, 1), ci.reshape(length, 1)),
                                                      axis=1)  # 下标矩阵
                        bs_rb_map[bsId][:, offset:offset + n_uh] = temp_matrix
                        # 计算用户的实际吞吐量
                        ue_thr = utils.compute_ue_thr(rb_thr_map=rb_thr_map[:, offset:offset + n_uh],
                                                      index_matrix=index_matrix) / 1000  # /kpbs
                        bs_throughput[bsId] += ue_thr * 1000  # /bps
                        used_rb[bsId] += n_uh * n_uv
                        slice_used_rb += n_uh * n_uv

                        # 计算用户满意度 sigmoid函数
                        sat_u = 1 / (1 + math.exp(1e-2 * (self.user_qos[0][sliceId] - ue_thr)))
                        t_u = 10  # 非ULL切片不把时延作为满意度统计指标，所以默认为10ms
                        if global_sat_ratio.shape[0] == 0:
                            # [sliceId, row['ue_id'], row['subframe_index'], t_u, sat_u, row['bs_id']]
                            global_sat_ratio = np.array([[sliceId, row[1], row[7], t_u, sat_u, row[0]]])
                        else:
                            sat_ratio_row = np.array([[sliceId, row[1], row[7], t_u, sat_u, row[0]]])
                            global_sat_ratio = np.concatenate((global_sat_ratio, sat_ratio_row), axis=0)

                        del_index.append(i2)
                        break

                # if sliceId == 1:
                #     print(used_rb)

                # 计算切片的资源利用率
                # slice_RU = slice_used_rb / np.sum(slice_bs_rb_num[sliceId, :])
                slice_rb = np.sum(slice_bs_rb_num[sliceId, :])
                slice_RU = slice_used_rb / slice_rb if slice_rb != 0 else 1
                if global_slice_RU.shape[0] == 0:
                    global_slice_RU = np.array([[sliceId, frame_index, slice_RU]])
                else:
                    slice_RU_row = np.array([[sliceId, frame_index, slice_RU]])
                    global_slice_RU = np.concatenate((global_slice_RU, slice_RU_row), axis=0)
                # 计算BS级的资源利用率
                for bi in range(4):
                    bs_RU = used_rb[bi] / slice_bs_rb_num[sliceId, bi] if slice_bs_rb_num[sliceId, bi] != 0 else 1
                    # bs_RU = used_rb[bi] / slice_bs_rb_num[sliceId, bi]
                    if np.isnan(bs_RU):
                        continue
                    if global_bs_RU.shape[0] == 0:
                        global_bs_RU = np.array([[sliceId, bi, frame_index, bs_RU]])
                    else:
                        bs_RU_row = np.array([[sliceId, bi, frame_index, bs_RU]])
                        global_bs_RU = np.concatenate((global_bs_RU, bs_RU_row), axis=0)

                # 移除已经分配资源的数据
                data_queue[sliceId] = np.delete(data_queue[sliceId], del_index, axis=0)

            # 对ULL切片的用户分配资源 ull_ues['bs_id', 'ue_id', 'ue_x', 'ue_y', 'n_uh', 'n_uv', 'rb_thr', 'subframe_index']
            ull_data_queue = self.env.generate_ull_data(ull_ues=ull_ues, cur_subframe=(cur_subframe - 9))
            # 将两个队列拼接到一起，并重建索引 ignore_index=True
            # last_ull_queue = pd.concat([last_ull_queue, ull_data_queue], ignore_index=True, sort=False)
            last_ull_queue = np.concatenate((last_ull_queue, ull_data_queue), axis=0)
            used_rb = np.zeros(4)  # 切片在各个BS上已经使用的RB，本实验设置为4个BS
            slice_used_rb = 0  # 切片在所有BS上已经使用的RB
            del_index = []
            sliceId = 3  # ULL slice id

            # 统计slice的队列长度
            queue_len = last_ull_queue.shape[0]
            if global_slice_queue_len.shape[0] == 0:
                global_slice_queue_len = np.array([[sliceId, frame_index, queue_len]])
            else:
                queue_len_row = np.array([[sliceId, frame_index, queue_len]])
                global_slice_queue_len = np.concatenate((global_slice_queue_len, queue_len_row), axis=0)

            for ui in range(last_ull_queue.shape[0]):
                row = last_ull_queue[ui]
                ue_info = row.tolist()  # 注意list的append()或者insert()方法都是返回的下标，而不是list对象
                ue_info.insert(0, sliceId)  # 设置sliceId为3，本实验表示ULL切片

                # (n_uh, n_uv, RB_thr) = utils.compute_rb_num(ue_info=ue_info, TPs=self.TPs, user_qos=self.user_qos)
                n_uh, n_uv, RB_thr = int(ue_info[5]), int(ue_info[6]), ue_info[7]  # 改成提前计算，无需每次重复计算
                bsId = int(row[0])  # row['bs_id']
                # 既要满足前程容量，又要满足回程容量  | 这里记得改sliceId = 3
                if (used_rb[bsId] + n_uh * n_uv > slice_bs_rb_num[sliceId, bsId]) \
                        or (bs_throughput[bsId] >= self.TPs.loc[bsId]['backhaul']):
                    continue
                # ------------物理资源映射------------
                rb_thr_map = utils.generate_rb_thr(mean_throughput=RB_thr, variance=4,
                                                   rows=sub_channel_num, cols=sub_frame_num)
                temp_map = bs_rb_map[bsId]  # 为什么要单独定义这个变量，纯粹为了减少变量名长度
                loc_index = np.where(temp_map == -1)  # sliceId = {0, 1, 2, 3}
                if loc_index[0].shape[0] < n_uh * n_uv:  # 再次判断BS上的资源是否足够
                    continue

                # 计算用户的实际吞吐量
                length = n_uh * n_uv
                loc_index = (loc_index[0][0:length], loc_index[1][0:length])
                bs_rb_map[bsId][loc_index] = sliceId  # 将bs_rb_map中符合条件的元素值设置为sliceId=3
                ue_thr = np.sum(rb_thr_map[loc_index]) / 1000  # /kbps
                bs_throughput[bsId] += ue_thr * 1000  # /bps
                used_rb[bsId] += n_uh * n_uv
                slice_used_rb += n_uh * n_uv

                # 计算用户的时延 = 等待时间
                enter_queue_subframe = row[7]  # 入队列的子帧下标
                t_wait = max(0, cur_subframe - enter_queue_subframe) + np.random.randint(1, 4)  # 等待时间，添加随机性
                t_u = t_wait  # /ms
                # t_u = 1 / (ue_thr * 1000 / self.user_qos[3][3] - self.user_qos[2][3]) * 1000 + t_wait  # /ms
                # 计算用户满意度 sigmoid函数
                sat_u = 1 / (1 + math.exp(1 * (t_u * 0.8 - self.user_qos[1][sliceId])))
                # global_sat_ratio = global_sat_ratio.append(
                #     pd.DataFrame(data=[[3, row['ue_id'], row['subframe_index'], t_u, sat_u]],
                #                  columns=columns_sat), ignore_index=True)
                if global_sat_ratio.shape[0] == 0:
                    global_sat_ratio = np.array([[sliceId, row[1], row[7], t_u, sat_u, row[0]]])
                else:
                    sat_ratio_row = np.array([[sliceId, row[1], row[7], t_u, sat_u, row[0]]])
                    global_sat_ratio = np.concatenate((global_sat_ratio, sat_ratio_row), axis=0)

                del_index.append(ui)  # 将索引添加入del_index中

            # 计算切片的资源利用率
            slice_rb = np.sum(slice_bs_rb_num[sliceId, :])
            slice_RU = slice_used_rb / slice_rb if slice_rb != 0 else 1
            # slice_RU = 1 if np.isnan(slice_RU) else slice_RU
            if global_slice_RU.shape[0] == 0:
                global_slice_RU = np.array([[sliceId, frame_index, slice_RU]])
            else:
                slice_RU_row = np.array([[sliceId, frame_index, slice_RU]])
                global_slice_RU = np.concatenate((global_slice_RU, slice_RU_row), axis=0)
            # 计算BS级的资源利用率
            for bi in range(4):
                bs_RU = used_rb[bi] / slice_bs_rb_num[sliceId, bi] if slice_bs_rb_num[sliceId, bi] != 0 else 1
                if np.isnan(bs_RU):
                    continue
                if global_bs_RU.shape[0] == 0:
                    global_bs_RU = np.array([[sliceId, bi, frame_index, bs_RU]])
                else:
                    bs_RU_row = np.array([[sliceId, bi, frame_index, bs_RU]])
                    global_bs_RU = np.concatenate((global_bs_RU, bs_RU_row), axis=0)

            # 统计ULL切片的满意度
            if last_ull_queue.shape[0] != 0:
                ull_sat = len(del_index) / last_ull_queue.shape[0]
                ull_sat_ratio_list.append(ull_sat)

            # 移除已经分配资源的数据
            last_ull_queue = np.delete(last_ull_queue, del_index, axis=0)

            # 当前子帧下标自增
            cur_subframe += 1
            frame_index += 1

        # 统计切片的平均资源利用率和BS级的资源利用率
        slice_avg_RU = np.zeros(4)
        slice_bs_RU = np.zeros([4, 4])
        for si in range(4):
            slice_avg_RU[si] = np.mean(global_slice_RU[global_slice_RU[:, 0] == si][:, 2])
            for bi in range(4):
                # 使用平均的资源利用率做为该BS的资源利用率
                y = global_bs_RU[(global_bs_RU[:, 0] == si) & (global_bs_RU[:, 1] == bi)]
                if y.shape[0] == 0:
                    slice_bs_RU[si, bi] = 1
                else:
                    slice_bs_RU[si, bi] = np.mean(y[:, 3])

        # 统计切片的用户满意度和BS级的满意度
        slice_sat_ratio = np.zeros(4)
        slice_bs_sat = np.zeros([4, 4])
        # 频谱效率 = slice发送包的数量 * packet_size / (r_allocated[si] * 4 * B(HZ))
        slice_spectral_efficiency = np.zeros(4)
        for si in range(4):
            # ['slice_id', 'ue_id', 'subframe_index', 'ue_delay', 'sat_ratio', 'bs_id']
            ues = global_sat_ratio[global_sat_ratio[:, 0] == si]
            # 计算频谱效率
            r_allocated = np.sum(r_sk[si, :]) / 4
            slice_spectral_efficiency[si] = ues.shape[0] * self.env.packet_size[si] / 0.2 / (r_allocated *
                                                                                             4 * self.env.sys_bandwidth)
            ueIds = pd.unique(ues[:, 1])
            ul = ueIds.shape[0]
            # slice_avg_sat_ratio = 0
            sat_u_list = []
            bs_ue_sat = np.zeros([ul, 3])  # ['bs_id', 'ue_id', 'sat_ratio']
            sat_ue_num = 0  # 满意度>0.5的用户数量
            for uj in range(ul):
                ue_info = ues[ues[:, 1] == ueIds[uj]][0]
                bs_id = ue_info[5]
                ue_id = ueIds[uj]
                if si != 3:
                    packet_num = ues[ues[:, 1] == ueIds[uj]].shape[0]
                    ue_rate = packet_num * self.user_qos[3][si] / (total_subframe / 1e3) / 1e3  # /kbps
                    if si == 0:
                        arrive_rate = 0.75 * self.user_qos[2][si] * self.user_qos[3][si] / 1e3
                    else:
                        arrive_rate = 0.75 * self.user_qos[2][si] * self.user_qos[3][si] / 1e3
                    avg_sat_u = 1 / (1 + math.exp(1 * (arrive_rate - ue_rate)))  # 1e-2
                    if avg_sat_u >= 0.5:
                        sat_ue_num += 1
                    sat_u_list.append(avg_sat_u)
                else:
                    avg_sat_u = np.mean(ues[ues[:, 1] == ue_id][:, 4])
                    sat_u_list.append(avg_sat_u)
                bs_ue_sat[uj, :] = [bs_id, ue_id, avg_sat_u]
                # slice_avg_sat_ratio += avg_sat_u
            # 判断队列中是否有剩余的packet
            # if data_queue[si].shape[0] != 0:
            #     # 切片的UE数量
            #     slice_ue_num = association_ues[association_ues['slice_id'] == si].shape[0]
            #     slice_sat_ratio[si] = slice_avg_sat_ratio / slice_ue_num
            # else:
            #     slice_sat_ratio[si] = slice_avg_sat_ratio / ul
            # 使用平均满意度
            slice_sat_ratio[si] = sum(sat_u_list) / len(sat_u_list) if len(sat_u_list) != 0 else 0
            # slice_sat_ratio[si] = sat_ue_num / ul if ul != 0 else 0  # 修改了切片平均满意度的定义方法
            # if si == 3:
            #     slice_sat_ratio[si] = sum(ull_sat_ratio_list) / len(ull_sat_ratio_list) if \
            #         len(ull_sat_ratio_list) != 0 else 0
            # 该切片在各个基站上的满意度
            for bi in range(4):
                bs_ue_sat_ = bs_ue_sat[bs_ue_sat[:, 0] == bi]
                min_sat_u = 0
                if bs_ue_sat_.shape[0] != 0:
                    # min_sat_u = np.min(bs_ue_sat_[:, 2])  # 该BS上的最小用户满意度
                    min_sat_u = np.mean(bs_ue_sat_[:, 2])  # 该BS上的平均用户满意度
                slice_bs_sat[si, bi] = min_sat_u

        return slice_sat_ratio, slice_avg_RU, slice_bs_sat, slice_bs_RU, slice_spectral_efficiency

    # 物理资源分配
    # @profile(precision=4, stream=open('memory_profiler.log', 'w+'))
    def allocate(self, association_ues, r_sk, total_subframe, rb_size):
        cur_subframe = 1  # 当前子帧下标
        frame_index = 1  # 当前帧下标
        data_queue = [np.empty([0, 8]) for i in range(4)]  # 此处直接初始化4类切片的队列

        columns_ue = ['slice_id', 'bs_id', 'ue_id', 'ue_x', 'ue_y']
        slice_ues = pd.DataFrame(index=list(range(3)), columns=columns_ue)
        for index, row in slice_ues.iterrows():
            ue = association_ues[association_ues['slice_id'] == index].iloc[0][columns_ue]
            slice_ues.iloc[index] = ue
        # 根据n_uv确定切片的调度顺序（降序）
        slice_order = self.get_schedule_order(ues=slice_ues, TPs=self.TPs)

        # 根据RB形状使每个切片在每个基站上的RB数据量刚好为整数倍的n_uv * n_uh
        # slice_ues.iloc[0] = association_ues[association_ues['slice_id'] == 0].iloc[0][columns_ue].tolist()
        slice_ues = slice_ues.append(pd.DataFrame(
            data=[association_ues[association_ues['slice_id'] == 3].iloc[0][columns_ue].tolist()],
            columns=columns_ue),
            ignore_index=True)
        slice_bs_rb_num = self.get_shaped_resource_ratio(ue_info=slice_ues, r_sk=r_sk, rb_size=rb_size)

        ull_ues = association_ues[association_ues['slice_id'] == 3]  # 超低时延用户，在该实验中配置sliceId为3

        sub_channel_num = self.env.sub_channel_num
        sub_frame_num = self.env.sub_frame_num

        # last_ull_queue = pd.DataFrame(columns=['bs_id', 'ue_id', 'ue_x', 'ue_y', 'subframe_index'])  # 初始化一个空的packet队列
        last_ull_queue = np.empty([0, 8])

        # 用户满意度统计量
        columns_sat = ['slice_id', 'ue_id', 'subframe_index', 'ue_delay', 'sat_ratio', 'bs_id']
        # global_sat_ratio = pd.DataFrame(columns=columns_sat)
        global_sat_ratio = np.empty([0, 6])  # 新增bsId用于统计BS级的满意度
        # 切片资源利用率统计量
        columns_RU = ['slice_id', 'frame_index', 'slice_RU']
        # global_slice_RU = pd.DataFrame(columns=columns_RU)
        global_slice_RU = np.empty([0, 3])
        # BS资源利用率统计量 ['slice_id', 'bs_id', 'frame_index', 'slice_RU']
        global_bs_RU = np.empty([0, 4])
        # slice的buffer队列长度['slice_id', 'frame_index', 'queue_len']
        global_slice_queue_len = np.empty([0, 3])
        # ULL切片的满意度列表
        ull_sat_ratio_list = []

        while cur_subframe <= total_subframe:
            # 模拟packet发送，packet会被存放在队列中data_queue[1...s]['bs_id', 'ue_id', 'ue_x', 'ue_y', 'subframe_index']
            # ['bs_id', 'ue_id', 'ue_x', 'ue_y', 'n_uh', 'n_uv', 'rb_thr', 'subframe_index']
            self.env.generate_downlink_data(data_queue, association_ues, cur_subframe)

            # -----------------每一帧做一次物理资源分配-----------------------
            if cur_subframe % 10 != 0:
                cur_subframe += 1
                continue
            if cur_subframe == 200:
                print(cur_subframe)
            bs_throughput = np.zeros(4)  # 基站吞吐量
            bs_rb_map = [np.ones([sub_channel_num, sub_frame_num]) * -1 for i in range(4)]  # 基站RB的map

            # 对非ULL切片的用户分配资源
            for index in range(3):
                sliceId = slice_order[index]
                # slice在各个BS上已经使用的RB，此处直接初始化为4个BS，可以根据自己的需求更改
                used_rb = np.zeros(4)
                # slice在所有BS上已经使用的RB
                slice_used_rb = 0
                del_index = []  # 已经分配资源的packet下标
                unsat_packet_num = 0  # 未分配资源的packet数量

                queue_len = data_queue[sliceId].shape[0]
                if global_slice_queue_len.shape[0] == 0:
                    global_slice_queue_len = np.array([[sliceId, frame_index, queue_len]])
                else:
                    queue_len_row = np.array([[sliceId, frame_index, queue_len]])
                    # 统计slice的队列长度
                    global_slice_queue_len = np.concatenate((global_slice_queue_len, queue_len_row), axis=0)

                # 遍历该slice的数据队列
                for i2 in range(data_queue[sliceId].shape[0]):
                    row = data_queue[sliceId][i2]
                    ue_info = row[:-1].tolist()
                    ue_info.insert(0, sliceId)
                    # n_uh, n_uv, RB_thr = utils.compute_rb_num(ue_info=ue_info, TPs=self.TPs, user_qos=self.user_qos)
                    n_uh , n_uv, RB_thr = int(ue_info[5]), int(ue_info[6]), ue_info[7]  # 改成提前计算，无需每次重复计算
                    # 既要满足前程容量，又要满足回程容量
                    bsId = int(row[0])  # row['bs_id']
                    if (used_rb[bsId] + n_uh * n_uv > slice_bs_rb_num[sliceId, bsId]) \
                            or (bs_throughput[bsId] >= self.TPs.loc[bsId]['backhaul']):
                        continue

                    # ----------------物理资源映射------------------------
                    rb_thr_map = utils.generate_rb_thr(mean_throughput=RB_thr, variance=4,
                                                      rows=sub_channel_num, cols=sub_frame_num)
                    for offset in range(0, sub_frame_num + 2 - n_uh):
                        temp_matrix = bs_rb_map[bsId][:, offset:offset + n_uh]
                        loc_index = np.where(temp_matrix == -1)  # sliceId = {0, 1, 2, 3}

                        if loc_index[0].shape[0] < n_uh * n_uv:  # 再次判断BS上的资源是否足够
                            continue
                        length = n_uh * n_uv
                        ri = loc_index[0][0:length]
                        ci = loc_index[1][0:length]
                        temp_matrix[(ri, ci)] = sliceId  # 使bs_rb_map中符合条件的元素值设置为sliceId

                        # index_matrix(二维矩阵)：第0列为行下标，第一列为列下标
                        index_matrix = np.concatenate((ri.reshape(length, 1), ci.reshape(length, 1)), axis=1)  # 下标矩阵
                        bs_rb_map[bsId][:, offset:offset + n_uh] = temp_matrix
                        # 计算用户的实际吞吐量
                        ue_thr = utils.compute_ue_thr(rb_thr_map=rb_thr_map[:, offset:offset + n_uh],
                                                      index_matrix=index_matrix) / 1000  # /kpbs
                        bs_throughput[bsId] += ue_thr * 1000  # /bps
                        used_rb[bsId] += n_uh * n_uv
                        slice_used_rb += n_uh * n_uv

                        # 计算用户满意度 sigmoid函数
                        sat_u = 1 / (1 + math.exp(1e-2 * (self.user_qos[0][sliceId] - ue_thr)))
                        t_u = 10  # 非ULL切片不把时延作为满意度统计指标，所以默认为10ms
                        if global_sat_ratio.shape[0] == 0:
                            # [sliceId, row['ue_id'], row['subframe_index'], t_u, sat_u, row['bs_id']]
                            global_sat_ratio = np.array([[sliceId, row[1], row[7], t_u, sat_u, row[0]]])
                        else:
                            sat_ratio_row = np.array([[sliceId, row[1], row[7], t_u, sat_u, row[0]]])
                            global_sat_ratio = np.concatenate((global_sat_ratio, sat_ratio_row), axis=0)

                        del_index.append(i2)
                        break

                # if sliceId == 1:
                #     print(used_rb)

                # 计算切片的资源利用率
                # slice_RU = slice_used_rb / np.sum(slice_bs_rb_num[sliceId, :])
                slice_rb = np.sum(slice_bs_rb_num[sliceId, :])
                slice_RU = slice_used_rb / slice_rb if slice_rb != 0 else 1
                if global_slice_RU.shape[0] == 0:
                    global_slice_RU = np.array([[sliceId, frame_index, slice_RU]])
                else:
                    slice_RU_row = np.array([[sliceId, frame_index, slice_RU]])
                    global_slice_RU = np.concatenate((global_slice_RU, slice_RU_row), axis=0)
                # 计算BS级的资源利用率
                for bi in range(4):
                    bs_RU = used_rb[bi] / slice_bs_rb_num[sliceId, bi] if slice_bs_rb_num[sliceId, bi] != 0 else 1
                    # bs_RU = used_rb[bi] / slice_bs_rb_num[sliceId, bi]
                    if np.isnan(bs_RU):
                        continue
                    if global_bs_RU.shape[0] == 0:
                        global_bs_RU = np.array([[sliceId, bi, frame_index, bs_RU]])
                    else:
                        bs_RU_row = np.array([[sliceId, bi, frame_index, bs_RU]])
                        global_bs_RU = np.concatenate((global_bs_RU, bs_RU_row), axis=0)

                # 移除已经分配资源的数据
                data_queue[sliceId] = np.delete(data_queue[sliceId], del_index, axis=0)

            # 对ULL切片的用户分配资源 ull_ues['bs_id', 'ue_id', 'ue_x', 'ue_y', 'n_uh', 'n_uv', 'rb_thr', 'subframe_index']
            ull_data_queue = self.env.generate_ull_data(ull_ues=ull_ues, cur_subframe=(cur_subframe - 9))
            # 将两个队列拼接到一起，并重建索引 ignore_index=True
            # last_ull_queue = pd.concat([last_ull_queue, ull_data_queue], ignore_index=True, sort=False)
            last_ull_queue = np.concatenate((last_ull_queue, ull_data_queue), axis=0)
            used_rb = np.zeros(4)  # 切片在各个BS上已经使用的RB，本实验设置为4个BS
            slice_used_rb = 0  # 切片在所有BS上已经使用的RB
            del_index = []
            sliceId = 3  # ULL slice id

            # 统计slice的队列长度
            queue_len = last_ull_queue.shape[0]
            if global_slice_queue_len.shape[0] == 0:
                global_slice_queue_len = np.array([[sliceId, frame_index, queue_len]])
            else:
                queue_len_row = np.array([[sliceId, frame_index, queue_len]])
                global_slice_queue_len = np.concatenate((global_slice_queue_len, queue_len_row), axis=0)

            for ui in range(last_ull_queue.shape[0]):
                row = last_ull_queue[ui]
                ue_info = row.tolist()  # 注意list的append()或者insert()方法都是返回的下标，而不是list对象
                ue_info.insert(0, sliceId)  # 设置sliceId为3，本实验表示ULL切片

                # (n_uh, n_uv, RB_thr) = utils.compute_rb_num(ue_info=ue_info, TPs=self.TPs, user_qos=self.user_qos)
                n_uh, n_uv, RB_thr = int(ue_info[5]), int(ue_info[6]), ue_info[7]  # 改成提前计算，无需每次重复计算
                bsId = int(row[0])  # row['bs_id']
                # 既要满足前程容量，又要满足回程容量  | 这里记得改sliceId = 3
                if (used_rb[bsId] + n_uh * n_uv > slice_bs_rb_num[sliceId, bsId]) \
                        or (bs_throughput[bsId] >= self.TPs.loc[bsId]['backhaul']):
                    continue
                # ------------物理资源映射------------
                rb_thr_map = utils.generate_rb_thr(mean_throughput=RB_thr, variance=4,
                                                  rows=sub_channel_num, cols=sub_frame_num)
                temp_map = bs_rb_map[bsId]  # 为什么要单独定义这个变量，纯粹为了减少变量名长度
                loc_index = np.where(temp_map == -1)  # sliceId = {0, 1, 2, 3}
                if loc_index[0].shape[0] < n_uh * n_uv:  # 再次判断BS上的资源是否足够
                    continue

                # 计算用户的实际吞吐量
                length = n_uh * n_uv
                loc_index = (loc_index[0][0:length], loc_index[1][0:length])
                bs_rb_map[bsId][loc_index] = sliceId  # 将bs_rb_map中符合条件的元素值设置为sliceId=3
                ue_thr = np.sum(rb_thr_map[loc_index]) / 1000  # /kbps
                bs_throughput[bsId] += ue_thr * 1000  # /bps
                used_rb[bsId] += n_uh * n_uv
                slice_used_rb += n_uh * n_uv

                # 计算用户的时延 = 等待时间
                enter_queue_subframe = row[7]  # 入队列的子帧下标
                t_wait = max(0, cur_subframe - enter_queue_subframe)  # 等待时间，添加随机性 + np.random.randint(1, 5)
                t_u = t_wait  # /ms
                # t_u = 1 / (ue_thr * 1000 / self.user_qos[3][3] - self.user_qos[2][3]) * 1000 + t_wait  # /ms
                # 计算用户满意度 sigmoid函数
                sat_u = 1 / (1 + math.exp(1 * (t_u * 0.8 - self.user_qos[1][sliceId])))
                # global_sat_ratio = global_sat_ratio.append(
                #     pd.DataFrame(data=[[3, row['ue_id'], row['subframe_index'], t_u, sat_u]],
                #                  columns=columns_sat), ignore_index=True)
                if global_sat_ratio.shape[0] == 0:
                    global_sat_ratio = np.array([[sliceId, row[1], row[7], t_u, sat_u, row[0]]])
                else:
                    sat_ratio_row = np.array([[sliceId, row[1], row[7], t_u, sat_u, row[0]]])
                    global_sat_ratio = np.concatenate((global_sat_ratio, sat_ratio_row), axis=0)

                del_index.append(ui)  # 将索引添加入del_index中

            # 计算切片的资源利用率
            slice_rb = np.sum(slice_bs_rb_num[sliceId, :])
            slice_RU = slice_used_rb / slice_rb if slice_rb != 0 else 1
            # slice_RU = 1 if np.isnan(slice_RU) else slice_RU
            if global_slice_RU.shape[0] == 0:
                global_slice_RU = np.array([[sliceId, frame_index, slice_RU]])
            else:
                slice_RU_row = np.array([[sliceId, frame_index, slice_RU]])
                global_slice_RU = np.concatenate((global_slice_RU, slice_RU_row), axis=0)
            # 计算BS级的资源利用率
            for bi in range(4):
                bs_RU = used_rb[bi] / slice_bs_rb_num[sliceId, bi] if slice_bs_rb_num[sliceId, bi] != 0 else 1
                if np.isnan(bs_RU):
                    continue
                if global_bs_RU.shape[0] == 0:
                    global_bs_RU = np.array([[sliceId, bi, frame_index, bs_RU]])
                else:
                    bs_RU_row = np.array([[sliceId, bi, frame_index, bs_RU]])
                    global_bs_RU = np.concatenate((global_bs_RU, bs_RU_row), axis=0)

            # 统计ULL切片的满意度
            if last_ull_queue.shape[0] != 0:
                ull_sat = len(del_index) / last_ull_queue.shape[0]
                ull_sat_ratio_list.append(ull_sat)

            # 移除已经分配资源的数据
            last_ull_queue = np.delete(last_ull_queue, del_index, axis=0)

            # 当前子帧下标自增
            cur_subframe += 1
            frame_index += 1

        # 统计切片的平均资源利用率和BS级的资源利用率
        slice_avg_RU = np.zeros(4)
        slice_bs_RU = np.zeros([4, 4])
        for si in range(4):
            slice_avg_RU[si] = np.mean(global_slice_RU[global_slice_RU[:, 0] == si][:, 2])
            for bi in range(4):
                # 使用平均的资源利用率做为该BS的资源利用率
                y = global_bs_RU[(global_bs_RU[:, 0] == si) & (global_bs_RU[:, 1] == bi)]
                if y.shape[0] == 0:
                    slice_bs_RU[si, bi] = 1
                else:
                    slice_bs_RU[si, bi] = np.mean(y[:, 3])

        # 统计切片的用户满意度和BS级的满意度
        slice_sat_ratio = np.zeros(4)
        slice_bs_sat = np.zeros([4, 4])
        # 频谱效率 = slice发送包的数量 * packet_size / (r_allocated[si] * 4 * B(HZ))
        slice_spectral_efficiency = np.zeros(4)
        for si in range(4):
            # ['slice_id', 'ue_id', 'subframe_index', 'ue_delay', 'sat_ratio', 'bs_id']
            ues = global_sat_ratio[global_sat_ratio[:, 0] == si]
            # 计算频谱效率
            r_allocated = np.sum(r_sk[si, :]) / 4
            slice_spectral_efficiency[si] = ues.shape[0] * self.env.packet_size[si] / 0.2 / (r_allocated *
                                                                                             4 * self.env.sys_bandwidth)
            ueIds = pd.unique(ues[:, 1])
            ul = ueIds.shape[0]
            # slice_avg_sat_ratio = 0
            sat_u_list = []
            bs_ue_sat = np.zeros([ul, 3])  # ['bs_id', 'ue_id', 'sat_ratio']
            sat_ue_num = 0  # 满意度>0.5的用户数量
            for uj in range(ul):
                ue_info = ues[ues[:, 1] == ueIds[uj]][0]
                bs_id = ue_info[5]
                ue_id = ueIds[uj]
                if si != 3:
                    packet_num = ues[ues[:, 1] == ueIds[uj]].shape[0]
                    ue_rate = packet_num * self.user_qos[3][si] / (total_subframe / 1e3) / 1e3  # /kbps
                    if si == 0:
                        arrive_rate = 0.75 * self.user_qos[2][si] * self.user_qos[3][si] / 1e3
                    else:
                        arrive_rate = 0.75 * self.user_qos[2][si] * self.user_qos[3][si] / 1e3
                    avg_sat_u = 1 / (1 + math.exp(1 * (arrive_rate - ue_rate)))  # 1e-2
                    if avg_sat_u >= 0.5:
                        sat_ue_num += 1
                    sat_u_list.append(avg_sat_u)
                else:
                    avg_sat_u = np.mean(ues[ues[:, 1] == ue_id][:, 4])
                    sat_u_list.append(avg_sat_u)
                bs_ue_sat[uj, :] = [bs_id, ue_id, avg_sat_u]
                # slice_avg_sat_ratio += avg_sat_u
            # 判断队列中是否有剩余的packet
            # if data_queue[si].shape[0] != 0:
            #     # 切片的UE数量
            #     slice_ue_num = association_ues[association_ues['slice_id'] == si].shape[0]
            #     slice_sat_ratio[si] = slice_avg_sat_ratio / slice_ue_num
            # else:
            #     slice_sat_ratio[si] = slice_avg_sat_ratio / ul
            # 使用平均满意度
            slice_sat_ratio[si] = sum(sat_u_list) / len(sat_u_list) if len(sat_u_list) != 0 else 0
            # slice_sat_ratio[si] = sat_ue_num / ul if ul != 0 else 0  # 修改了切片平均满意度的定义方法
            # if si == 3:
            #     slice_sat_ratio[si] = sum(ull_sat_ratio_list) / len(ull_sat_ratio_list) if \
            #         len(ull_sat_ratio_list) != 0 else 0
            # 该切片在各个基站上的满意度
            for bi in range(4):
                bs_ue_sat_ = bs_ue_sat[bs_ue_sat[:, 0] == bi]
                min_sat_u = 0
                if bs_ue_sat_.shape[0] != 0:
                    # min_sat_u = np.min(bs_ue_sat_[:, 2])  # 该BS上的最小用户满意度
                    min_sat_u = np.mean(bs_ue_sat_[:, 2])  # 该BS上的平均用户满意度
                slice_bs_sat[si, bi] = min_sat_u
        """
        print('slice_sat_ratio: ', slice_sat_ratio)
        print('slice_avg_RU: ', slice_avg_RU)
        print('slice_bs_sat: ', slice_bs_sat)
        print('slice_bs_RU: ', slice_bs_RU)
        print('slice_spectral_efficiency: ', slice_spectral_efficiency)
        print('global_slice_queue_len: ', global_slice_queue_len)

        return_allocated = {"slice_sat_ratio": slice_sat_ratio, 
                "slice_avg_RU": slice_avg_RU, 
                "slice_bs_sat": slice_bs_sat, 
                "slice_bs_RU":slice_bs_RU,
                "slice_spectral_efficiency": slice_spectral_efficiency, 
                "global_slice_queue_len": global_slice_queue_len}
        """        
        return slice_sat_ratio, slice_avg_RU, slice_bs_sat, slice_bs_RU, slice_spectral_efficiency, global_slice_queue_len

    # 执行动作 action = {-30%, -10%, 10%, 30%}
    def step(self, action, association_ues, r_sk, total_subframe, rb_size):
        # action_str = '{:08b}'.format(action)   # 进十进制转化为二进制字符串，高位补0
        # action_space = [-0.3, -0.15, 0, 0.15, 0.3]  # 动作空间，调整切片资源的百分比
        # action_space = [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5]

        # action_space = [-0.3, -0.15, 0, 0.3, 0.5]
        # action_space = [-0.4, -0.2, 0, 0.4, 0.6]
        # action_space = [-0.4, -0.3, 0, 0.4, 0.7]
        action_space = [-0.2, 0, 0.4]
        print('action: ', action[0])
        action_str = utils.convert_fix_bit(action[0], 3, bit_len=4)  # 将action下标转化为动作字符串
        real_action = np.zeros(4)  # 实际的调整百分比
        for i in range(4):
            index = int(action_str[i])  # 确定每个slice的动作下标
            real_action[i] = action_space[index]

        r_s = np.zeros(4)  # 切片级的资源比例
        for i in range(4):
            r_s[i] = np.sum(r_sk[i, :]) / 4  # 本实验中有4个BS，因此总资源为4
            r_s[i] *= (1 + real_action[i])  # 调整切片虚拟资源百分比

        # 计算reversed resource和allocated resource
        r_allocated = r_s.copy()
        if np.sum(r_s) >= 1:
            r_reserved = np.zeros(4)
        else:
            r_reserved = (1 - np.sum(r_allocated)) * r_s / np.sum(r_s)

        # 映射到BS上的比例
        for si in range(4):
            ue_rate_arr = np.zeros(4)  # 切片在各个基站上的UE的速率总和
            slice_ues = association_ues[association_ues['slice_id'] == si]
            for bi in range(4):
                if si != 3:  # 如果是非ULL切片，则使用速率
                    ue_rate_arr[bi] = np.sum(slice_ues[slice_ues['bs_id'] == bi]['ue_rate'])
                else:  # 如果是ULL切片，则使用UE的数量
                    ue_rate_arr[bi] = slice_ues[slice_ues['bs_id'] == bi].shape[0]
            for bi in range(4):
                # 根据速率或者UE数量分配比例
                r_sk[si, bi] = self.env.bs_num * r_allocated[si] * (ue_rate_arr[bi] / np.sum(ue_rate_arr))

        r_sk_allocated = np.zeros([4, 4])  # 保存allocated资源映射之后的结果
        # 保证每个BS上的资源比例总和不大于1
        for bi in range(4):
            if np.sum(r_sk[:, bi]) > 1:
                r_sk_allocated[:, bi] = r_sk[:, bi] / np.sum(r_sk[:, bi])
            else:
                r_sk_allocated[:, bi] = r_sk[:, bi]
            r_sk[:, bi] *= (1 / np.sum(r_sk[:, bi]))  # 全部分完

        print('After adjusting ', real_action, ' , the r_sk is:')
        print(r_sk)

        # 进行物理资源分配，如果需要使用allocated的结果，则传入r_sk_allocated，否则传入r_sk
        slice_sat_ratio, slice_avg_RU, slice_bs_sat, slice_bs_RU, slice_spectral_efficiency, global_slice_queue_len = \
            self.allocate(association_ues, r_sk_allocated.copy(), total_subframe, rb_size)
        reward = utils.compute_reward(slice_sat_ratio=slice_sat_ratio, slice_avg_RU=slice_avg_RU)
        # 手动调控reward，如果调整后的切片级资源比例之和大于1，则reward为-2
        if np.sum(r_s) > 1:
            reward = np.array([-1, -1, -1, -1, -4])

        # 下一个状态 [r_s, sat, ru]
        observation_ = np.concatenate((r_s, slice_sat_ratio, slice_avg_RU))
        print('The state is : ')
        print(observation_)
        print('The reward is : ')
        print(reward)
        print('The slice_bs_sat is : ', slice_bs_sat, '\nThe slice_bs_RU is : ', slice_bs_RU)

        return observation_, reward, r_sk_allocated, slice_bs_sat, slice_bs_RU, real_action, r_allocated, r_reserved, slice_spectral_efficiency

    def step_ddpg(self, action, association_ues, r_sk, total_subframe, rb_size):
        action_bound = [-0.2, 0.4]
        # clip这个函数将将数组中的元素限制在a_min, a_max之间，大于a_max 的就使得它等于 a_max，小于a_min 的就使得它等于a_min
        action = np.clip(action, *action_bound)  # * 的作用其实就是把序列 args 中的每个元素，当作位置参数传进去
        real_action = action.copy()

        r_s = np.zeros(4)  # 切片所占资源的百分比
        for i in range(4):
            r_s[i] = np.sum(r_sk[i, :]) / 4  # 本实验中有4个BS，因此总资源为4
            r_s[i] *= (1 + action[i])  # 调整切片虚拟资源百分比

        # 计算reversed resource和allocated resource
        r_allocated = r_s.copy()
        if np.sum(r_s) >= 1:
            r_reserved = np.zeros(4)
        else:
            r_reserved = (1 - np.sum(r_allocated)) * r_s / np.sum(r_s)

        # 映射到BS上的比例
        for si in range(4):
            ue_rate_arr = np.zeros(4)  # 切片在各个基站上的UE的速率总和
            slice_ues = association_ues[association_ues['slice_id'] == si]
            for bi in range(4):
                if si != 3:  # 如果是非ULL切片，则使用速率
                    ue_rate_arr[bi] = np.sum(slice_ues[slice_ues['bs_id'] == bi]['ue_rate'])
                else:  # 如果是ULL切片，则使用UE的数量
                    ue_rate_arr[bi] = slice_ues[slice_ues['bs_id'] == bi].shape[0]
            for bi in range(4):
                # 根据速率或者UE数量分配比例
                r_sk[si, bi] = self.env.bs_num * r_s[si] * (ue_rate_arr[bi] / np.sum(ue_rate_arr))

        r_sk_allocated = np.zeros([4, 4])  # 保存allocated资源映射之后的结果
        # 保证每个BS上的资源比例总和不大于1
        for bi in range(4):
            if np.sum(r_sk[:, bi]) > 1:
                r_sk_allocated[:, bi] = r_sk[:, bi] / np.sum(r_sk[:, bi])
            else:
                r_sk_allocated[:, bi] = r_sk[:, bi]
            r_sk[:, bi] *= (1 / np.sum(r_sk[:, bi]))  # 全部分完

        print('After adjusting ', action, ', the r_sk is:')
        print(r_sk)

        # 进行物理资源分配
        slice_sat_ratio, slice_avg_RU, slice_bs_sat, slice_bs_RU, slice_spectral_efficiency, global_slice_queue_len = \
            self.allocate(association_ues, r_sk_allocated.copy(), total_subframe, rb_size)
        reward = utils.compute_reward(slice_sat_ratio=slice_sat_ratio, slice_avg_RU=slice_avg_RU)
        # 手动调控reward，如果调整后的切片级资源比例之和大于1，则reward为-2
        if np.sum(r_s) > 1:
            reward = np.array([-1, -1, -1, -1, -4])

        # 下一个状态 [r_s, sat, ru]
        observation_ = np.concatenate((r_s, slice_sat_ratio, slice_avg_RU))
        print('The state is : ')
        print(observation_)
        print('The reward is : ')
        print(reward)
        print('The slice_bs_sat is : ', slice_bs_sat, '\nThe slice_bs_RU is : ', slice_bs_RU)

        return observation_, reward, r_sk_allocated, slice_bs_sat, slice_bs_RU, real_action, r_allocated, r_reserved, slice_spectral_efficiency

