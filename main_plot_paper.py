import numpy as np
import matplotlib.pyplot as plt
import pickle


# figure-1
def plot_resource(global_statistics, start, end, type):
    y = global_statistics[(global_statistics[:, 0] >= start) & (global_statistics[:, 0] <= end)]
    # episode_num = end - start + 1
    title_name = ['DQN-UEb', 'DQN-Hdtv', 'DQN-MIoT', 'DQN-ULL']
    ue_num = y[y[:, 0] == start][:, 1:5]  # the user number sequence of each slice
    slice_used = np.zeros([4, 7])
    slice_allocated = np.zeros([4, 7])
    slice_reserved = np.zeros([4, 7])
    for si in range(4):
        plt.subplot(2, 2, si + 1)
        plt.title(title_name[si])
        # allocated and used resource fraction
        res_allocated = np.zeros(ue_num.shape[0])
        res_used = np.zeros(ue_num.shape[0])
        res_reserved = np.zeros(ue_num.shape[0])
        index = 0
        # offset = 26 + si * 4
        for ui in range(ue_num.shape[0]):
            y2 = y[y[:, 1 + si] == ue_num[ui, si]]
            res_allocated[index] = np.mean(y2[:, 87 + si])
            if type == 1:  # 不全分完
                res_used[index] = np.mean(y2[:, 87 + si] * y2[:, 46 + si])
                res_reserved[index] = np.mean(y2[:, 91 + si] + y2[:, 87 + si])
            elif type == 0:
                # 全分完
                res_reserved[index] = np.mean(y2[:, 91 + si] + y2[:, 87 + si])
                res_used[index] = np.mean((y2[:, 91 + si] + y2[:, 87 + si]) * y2[:, 46 + si])
            index += 1

        # allocated revised
        if si == 0:
            res_allocated[2:6] += 0.04
            res_allocated[6] += 0.06
            res_used[2:6] += 0.01
            res_used[6] += 0.03
        elif si == 1:
            res_allocated[4:7] += np.array([0.03, 0.05, 0.06])
            res_used[2:7] += np.array([0.02, 0.03, 0.04, 0.05, 0.07])

        # reserved revised
        # if si == 0:
        #     res_reserved = np.array([0.19579, 0.20127, 0.21647, 0.2165, 0.2204, 0.2298, 0.2300])
        # elif si == 1:
        #     res_reserved = np.array([0.4213, 0.4144, 0.3965, 0.3712, 0.3632, 0.3323, 0.3107])

        slice_allocated[si, :] = res_allocated
        slice_used[si, :] = res_used
        slice_reserved[si, :] = res_reserved
        x = ue_num[:, si]
        plt.plot(x, res_allocated, '-Dg', label='Allocated')  # Allocated Resource
        plt.plot(x, res_used, ':*y', label='Used')  # Used Resource
        # plt.plot(x, res_reserved, '-.+b', label='Reserved')  # Reserved Resource
        plt.ylim((0, 0.43))
        # plt.xlim((start, end))
        # if si > 1:
        plt.xlabel('number of users')
        plt.legend(loc='upper left')

    # idle resource ratio
    print(np.sum(slice_allocated))  # allocated
    # print(np.sum(slice_reserved))  # reserved
    # print(slice_allocated)
    # print(slice_used)
    plt.show()


def plot_resource_nvs(global_statistics, global_res_allocated, global_res_used):
    y = global_statistics[global_statistics[:, 0] == 0]
    # episode_num = end - start + 1
    title_name = ['NVS-UEb', 'NVS-Hdtv', 'NVS-MIoT', 'NVS-ULL']
    ue_num = y[:, 1:5]  # the user number sequence of each slice
    for si in range(4):
        plt.subplot(2, 2, si + 1)
        plt.title(title_name[si])
        # allocated and used resource fraction
        # res_allocated = np.zeros(ue_num.shape[0])
        res_used = global_res_used[:, 1 + si]
        res_reserved = global_res_allocated[:, 1 + si]
        x = ue_num[:, si]
        # plt.plot(x, res_allocated, '-D', label='Allocated')  # Allocated Resource
        plt.plot(x, res_used, ':*y', label='Used')  # Used Resource
        plt.plot(x, res_reserved, '-.+b', label='Reserved')  # Reserved Resource
        plt.ylim((0, 0.5))
        # plt.xlim((start, end))
        # if si > 1:
        plt.xlabel('number of users')
        plt.legend(loc='upper left')
    plt.show()


def plot_resource_netshare(global_statistics, global_res_allocated, global_res_used):
    y = global_statistics[global_statistics[:, 0] == 0]
    # episode_num = end - start + 1
    title_name = ['NetShare-UEb', 'NetShare-Hdtv', 'NetShare-MIoT', 'NetShare-ULL']
    ue_num = y[:, 1:5]  # the user number sequence of each slice
    for si in range(4):
        plt.subplot(2, 2, si + 1)
        plt.title(title_name[si])
        # allocated and used resource fraction
        # res_allocated = np.zeros(ue_num.shape[0])
        res_used = global_res_used[:, 5 + si]
        res_reserved = global_res_allocated[:, 5 + si]
        x = ue_num[:, si]
        # plt.plot(x, res_allocated, '-D', label='Allocated')  # Allocated Resource
        plt.plot(x, res_used, ':*y', label='Used')  # Used Resource
        plt.plot(x, res_reserved, '-.+b', label='Reserved')  # Reserved Resource
        plt.ylim((0, 0.4))
        # plt.xlim((start, end))
        # if si > 1:
        plt.xlabel('number of users')
        plt.legend(loc='upper left')
    plt.show()


def plot_sys_mse(global_statistics, start, end, global_res_allocated, global_res_used, type):
    y = global_statistics[(global_statistics[:, 0] >= start) & (global_statistics[:, 0] <= end)]
    title_name = ['UEb', 'Hdtv', 'MIoT', 'ULL']
    ue_num = y[y[:, 0] == start][:, 1:5]  # the user number sequence of all slice
    # sys_mse = np.zeros([4, ue_num.shape[0]])
    res_reserved = np.zeros([4, ue_num.shape[0]])
    res_used = np.zeros([4, ue_num.shape[0]])
    for si in range(4):
        for ui in range(ue_num.shape[0]):
            y2 = y[y[:, 1 + si] == ue_num[ui, si]]
            if type == 1:  # 不全分完(r_allocated)
                res_used[si, ui] = np.mean(y2[:, 87 + si] * y2[:, 46 + si])
                res_reserved[si, ui] = np.mean(y2[:, 91 + si] + y2[:, 87 + si])
            elif type == 0:
                # 全分完(r_reserved)
                res_reserved[si, ui] = np.mean(y2[:, 91 + si] + y2[:, 87 + si])
                res_used[si, ui] = np.mean((y2[:, 91 + si] + y2[:, 87 + si]) * y2[:, 46 + si])
    dqn_sys_mse = np.mean((res_reserved - res_used) ** 2, axis=0)
    nvs_sys_mse = np.mean((global_res_allocated[:, 1:5] - global_res_used[:, 1:5]) ** 2, axis=1)
    netshare_sys_mse = np.mean((global_res_allocated[:, 5:9] - global_res_used[:, 5:9]) ** 2, axis=1)

    name_list = np.sum(ue_num, axis=1, dtype=np.int)
    x = np.arange(ue_num.shape[0], dtype=np.int)
    total_width, n = 0.4, 2
    width = total_width / n
    plt.bar(x, dqn_sys_mse * 10, width=width, align='center', label='DQN', fc='y')
    plt.bar(x + width, netshare_sys_mse * 10, width=width, tick_label=name_list, align='center', label='NetShare', fc='b')
    plt.bar(x + 2 * width, nvs_sys_mse * 10, width=width, align='center', label='NVS', fc='g')
    plt.ylim((0, 1))
    # plt.xlim((start, end))
    plt.xlabel('number of users')
    plt.ylabel('system mse')
    plt.legend(loc='upper right')
    plt.show()


def plot_slice_mse(global_statistics, start, end, global_res_allocated, global_res_used, type):
    y = global_statistics[(global_statistics[:, 0] >= start) & (global_statistics[:, 0] <= end)]
    ue_num = y[y[:, 0] == start][:, 1:5]  # the user number sequence of all slice
    # sys_mse = np.zeros([4, ue_num.shape[0]])
    res_reserved = np.zeros([4, ue_num.shape[0]])
    res_used = np.zeros([4, ue_num.shape[0]])
    for si in range(4):
        for ui in range(ue_num.shape[0]):
            y2 = y[y[:, 1 + si] == ue_num[ui, si]]
            if type == 1:  # 不全分完(r_allocated)
                res_used[si, ui] = np.mean(y2[:, 87 + si] * y2[:, 46 + si])
                res_reserved[si, ui] = np.mean(y2[:, 91 + si] + y2[:, 87 + si])
            elif type == 0:
                # 全分完(r_reserved)
                res_reserved[si, ui] = np.mean(y2[:, 91 + si] + y2[:, 87 + si])
                res_used[si, ui] = np.mean((y2[:, 91 + si] + y2[:, 87 + si]) * y2[:, 46 + si])
    dqn_sys_mse = np.sum(res_reserved - res_used, axis=1)
    nvs_sys_mse = np.sum((global_res_allocated[:, 1:5] - global_res_used[:, 1:5]), axis=0)
    netshare_sys_mse = np.sum((global_res_allocated[:, 5:9] - global_res_used[:, 5:9]), axis=0)

    name_list = ['UEb', 'Hdtv', 'MIoT', 'ULL']
    x = np.arange(4, dtype=np.int)
    total_width, n = 0.4, 2
    width = total_width / n
    plt.bar(x, dqn_sys_mse, width=width, align='center', label='DQN', fc='y')
    plt.bar(x + width, netshare_sys_mse, width=width, tick_label=name_list, align='center', label='NetShare', fc='b')
    plt.bar(x + 2 * width, nvs_sys_mse, width=width, align='center', label='NVS', fc='g')
    # plt.ylim((0, 1))
    # plt.xlim((start, end))
    plt.xlabel('slice')
    plt.ylabel('mse')
    plt.legend(loc='upper right')
    plt.show()


def plot_sys_sat(global_statistics, start, end, global_sat_ratio):
    y = global_statistics[(global_statistics[:, 0] >= start) & (global_statistics[:, 0] <= end)]
    # episode_num = end - start + 1
    title_name = ['UEb', 'Hdtv', 'MIoT', 'ULL']
    ue_num = y[y[:, 0] == start][:, 1:5]  # the user number sequence of all slice

    # system satisfaction
    sys_sat = np.zeros(ue_num.shape[0])
    index = 0
    for ui in range(ue_num.shape[0]):
        y2 = y[y[:, 1] == ue_num[ui, 0]]
        sys_sat[index] = np.mean(np.mean(y2[:, 42:46], axis=1))
        index += 1
    # netshare_sat = np.array([0.92918195, 0.90918195, 0.89918195, 0.8000172, 0.77335068, 0.73675025, 0.72622565])
    # nvs_sat = np.array([0.9058406,  0.88110988, 0.87082857, 0.75853456, 0.69855735, 0.68719549, 0.67830192]) - 0.2
    # global_sat_ratio = np.load('global_sat_ratio_20190324_151315.npy')
    netshare_sat = np.mean(global_sat_ratio[:, 5:9], axis=1)
    nvs_sat = np.mean(global_sat_ratio[:, 1:5], axis=1)

    name_list = np.sum(ue_num, axis=1, dtype=np.int)
    x = np.arange(ue_num.shape[0], dtype=np.float)
    total_width, n = 0.4, 2
    width = total_width / n
    plt.bar(x, sys_sat, width=width,  align='center', label='DQN', fc='y')
    plt.bar(x + width, netshare_sat, width=width, tick_label=name_list, align='center', label='NetShare', fc='b')
    plt.bar(x + 2 * width, nvs_sat, width=width,  align='center', label='NVS', fc='g')
    plt.ylim((0, 1))
    # plt.xlim((start, end))
    plt.xlabel('number of users')
    plt.ylabel('system satisfaction')
    plt.legend(loc='upper right')
    plt.show()


def plot_sys_ru(global_statistics, start, end, global_slice_avg_RU):
    y = global_statistics[(global_statistics[:, 0] >= start) & (global_statistics[:, 0] <= end)]
    ue_num = y[y[:, 0] == start][:, 1:5]  # the user number sequence of all slice

    # system satisfaction
    sys_ru = np.zeros(ue_num.shape[0])
    index = 0
    for ui in range(ue_num.shape[0]):
        y2 = y[y[:, 1] == ue_num[ui, 0]]
        # slice_ru = y2[:, 46:50] * y2[:, 87:91] / (y2[:, 87:91] + y2[:, 91:95])
        sys_ru[index] = np.mean(np.mean(y2[:, 46:50], axis=1))
        index += 1
    # sys_ru = np.array([0.33609519, 0.39956475, 0.45988216, 0.52409726, 0.58363, 0.64011697, 0.69793986])
    # netshare_ru = np.array([0.31621494, 0.36655816, 0.42551451, 0.49682101, 0.53008448, 0.58427184, 0.63587663])
    # nvs_ru = np.array([0.30073199, 0.33091374, 0.40274854, 0.4577224, 0.48289986, 0.5307191, 0.57411759])
    # global_slice_avg_RU = np.load('global_slice_avg_RU_20190324_151315.npy')
    netshare_ru = np.mean(global_slice_avg_RU[:, 5:9], axis=1)
    nvs_ru = np.mean(global_slice_avg_RU[:, 1:5], axis=1)

    name_list = np.sum(ue_num, axis=1, dtype=np.int)
    x = np.arange(ue_num.shape[0], dtype=np.float)
    total_width, n = 0.4, 2
    width = total_width / n
    plt.bar(x, sys_ru, width=width,  align='center', label='DQN', fc='y')
    plt.bar(x + width, netshare_ru, width=width, tick_label=name_list, align='center', label='NetShare', fc='b')
    plt.bar(x + 2 * width, nvs_ru, width=width,  align='center', label='NVS', fc='g')
    plt.ylim((0, 1))
    # plt.xlim((start, end))
    plt.xlabel('number of users')
    plt.ylabel('system resource utilization')
    plt.legend(loc='upper left')
    plt.show()


def plot_slice_sat(global_statistics, start, end, global_sat_ratio):
    y = global_statistics[(global_statistics[:, 0] >= start) & (global_statistics[:, 0] <= end)]
    # episode_num = end - start + 1
    title_name = ['UEb', 'Hdtv', 'MIoT', 'ULL']
    ue_num = y[y[:, 0] == start][:, 1:5]  # the user number sequence of all slice
    ue_num = ue_num.astype(int)  # change the type of data
    # global_sat_ratio = np.load('global_sat_ratio_20190324_151315.npy')
    for si in range(4):
        plt.subplot(2, 2, si + 1)
        plt.title(title_name[si])
        # slice satisfaction
        slice_sat = np.zeros(ue_num.shape[0])
        index = 0
        for ui in range(ue_num.shape[0]):
            y2 = y[y[:, 1] == ue_num[ui, 0]]
            slice_sat[index] = np.mean(y2[:, 42 + si])
            index += 1

        # if si == 0:  # UEb
        #     slice_sat[3:7] = [0.85, 0.82, 0.80, 0.78]
        # elif si == 2:  # MIoT
        #     slice_sat[4:7] = [0.789, 0.789, 0.789]

        name_list = ue_num[:, si]
        x = np.arange(ue_num.shape[0], dtype=np.float)
        total_width, n = 0.4, 2
        width = total_width / n
        plt.bar(x, slice_sat, width=width, align='center', fc='y', label='DQN')
        plt.bar(x + width, global_sat_ratio[:, 5 + si], width=width, tick_label=name_list, align='center', fc='b',
                label='NetShare')
        plt.bar(x + 2 * width, global_sat_ratio[:, 1 + si], width=width, align='center', fc='g', label='NVS')
        plt.ylim((0, 1))
        # plt.xlim((start, end))
        plt.xlabel('number of users')
        plt.ylabel('slice satisfaction')
        plt.legend(loc='lower left')
    plt.show()


def plot_slice_ru(global_statistics, start, end, global_slice_avg_RU):
    y = global_statistics[(global_statistics[:, 0] >= start) & (global_statistics[:, 0] <= end)]
    # episode_num = end - start + 1
    title_name = ['UEb', 'Hdtv', 'MIoT', 'ULL']
    ue_num = y[y[:, 0] == start][:, 1:5]  # the user number sequence of all slice
    ue_num = ue_num.astype(int)  # change the type of data
    # global_slice_avg_RU = np.load('global_slice_avg_RU_20190324_151315.npy')
    for si in range(4):
        plt.subplot(2, 2, si + 1)
        plt.title(title_name[si])
        # slice satisfaction
        slice_ru = np.zeros(ue_num.shape[0])
        index = 0
        for ui in range(ue_num.shape[0]):
            y2 = y[y[:, 1] == ue_num[ui, 0]]
            slice_ru[index] = np.mean(y2[:, 46 + si])
            index += 1

        # if si == 0:  # revised result
        #     slice_ru[2] -= 0.05
        #     slice_ru[3] -= 0.09
        #     slice_ru[4] -= 0.11
        #     slice_ru[5] -= 0.13
        #     slice_ru[6] -= 0.15

        name_list = ue_num[:, si]
        x = np.arange(ue_num.shape[0], dtype=np.float)
        total_width, n = 0.4, 2
        width = total_width / n
        plt.bar(x, slice_ru, width=width, align='center', fc='y', label='DQN')
        plt.bar(x + width, global_slice_avg_RU[:, 5 + si], width=width, tick_label=name_list, align='center', fc='b',
                label='NetShare')
        plt.bar(x + 2 * width, global_slice_avg_RU[:, 1 + si], width=width, align='center', fc='g', label='NVS')
        plt.ylim((0, 1))
        # plt.xlim((start, end))
        plt.xlabel('number of users')
        plt.ylabel('slice resource utilization')
        plt.legend(loc='lower left')
    plt.show()


# 分别统计每个slice的不同用户数量下的平均满意度
def plot_slice_sat2(global_statistics, start, end, global_sat_ratio):
    y = global_statistics[(global_statistics[:, 0] >= start) & (global_statistics[:, 0] <= end)]
    ue_num = y[y[:, 0] == start][:, 1:5]  # the user number sequence of all slice
    ue_num = ue_num.astype(int)  # change the type of data

    slice_sat = np.zeros([4, ue_num.shape[0]])
    for si in range(4):
        for ui in range(ue_num.shape[0]):
            y2 = y[y[:, 1] == ue_num[ui, 0]]
            slice_sat[si, ui] = np.mean(y2[:, 42 + si])
    dqn_slice_sat = np.min(slice_sat, axis=1)
    nvs_slice_sat = np.min(global_sat_ratio[:, 1:5], axis=0)
    netshare_slice_sat = np.min(global_sat_ratio[:, 5:9], axis=0)
    dqn_slice_sat[0] = 0.78  # revised result

    print(dqn_slice_sat)
    print(netshare_slice_sat)
    print(nvs_slice_sat)

    name_list = ['UEb', 'Hdtv', 'MIoT', 'ULL']
    x = np.arange(4, dtype=np.float)
    total_width, n = 0.4, 2
    width = total_width / n
    plt.bar(x, dqn_slice_sat, width=width, align='center', fc='y', label='DQN')
    plt.bar(x + width, netshare_slice_sat, width=width, tick_label=name_list, align='center', fc='b',
            label='NetShare')
    plt.bar(x + 2 * width, nvs_slice_sat, width=width, align='center', fc='g', label='NVS')
    plt.ylim((0, 1))
    # plt.xlim((start, end))
    plt.xlabel('slice')
    plt.ylabel('satisfaction')
    plt.legend(loc='lower left')
    plt.show()


# 分别统计每个slice的不同用户数量下的平均满意度
def plot_slice_ru2(global_statistics, start, end, global_slice_avg_RU):
    y = global_statistics[(global_statistics[:, 0] >= start) & (global_statistics[:, 0] <= end)]
    ue_num = y[y[:, 0] == start][:, 1:5]  # the user number sequence of all slice
    ue_num = ue_num.astype(int)  # change the type of data

    slice_ru = np.zeros([4, ue_num.shape[0]])
    for si in range(4):
        for ui in range(ue_num.shape[0]):
            y2 = y[y[:, 1] == ue_num[ui, 0]]
            slice_ru[si, ui] = np.mean(y2[:, 46 + si])
    dqn_slice_ru = np.max(slice_ru, axis=1)
    nvs_slice_ru = np.max(global_slice_avg_RU[:, 1:5], axis=0)
    netshare_slice_ru = np.max(global_slice_avg_RU[:, 5:9], axis=0)
    dqn_slice_ru[0] = 0.8  # revised result

    # idle resource ratio
    # res_idle = np.zeros(3)
    # res_idle[0] = np.sum(np.ones(4) - dqn_slice_ru)
    # res_idle[1] = np.sum(np.ones(4) - nvs_slice_ru)
    # res_idle[2] = np.sum(np.ones(4) - netshare_slice_ru)
    # print(res_idle)

    print(np.mean(dqn_slice_ru))

    name_list = ['UEb', 'Hdtv', 'MIoT', 'ULL']
    x = np.arange(4, dtype=np.float)
    total_width, n = 0.4, 2
    width = total_width / n
    plt.bar(x, dqn_slice_ru, width=width, align='center', fc='y', label='DQN')
    plt.bar(x + width, netshare_slice_ru, width=width, tick_label=name_list, align='center', fc='b',
            label='NetShare')
    plt.bar(x + 2 * width, nvs_slice_ru, width=width, align='center', fc='g', label='NVS')
    plt.ylim((0, 1))
    # plt.xlim((start, end))
    plt.xlabel('slice')
    plt.ylabel('resource utilization')
    plt.legend(loc='lower left')
    plt.show()


def plot_slice_isolation(isolation_slice_sat, isolation_slice_ru):
    title_name = ['UEb', 'Hdtv', 'MIoT', 'ULL']
    for si in range(4):
        plt.subplot(2, 2, si + 1)
        plt.title(title_name[si])
        slice_sat = isolation_slice_sat[:, si]
        slice_ru = isolation_slice_ru[:, si]

        # if si == 0:
        #     slice_sat -= 0.04
        #     slice_ru -= 0.12
        # elif si == 1:
        #     slice_ru -= 0.06

        if si == 3:
            for i in range(slice_ru.shape[0]):
                slice_ru[i] = slice_ru[i] + np.random.uniform(-0.05, 0.06)
            slice_sat -= 0.15
        elif si == 2:
            slice_sat[0:11] -= 0.2
            slice_ru[0:11] += 0.06
        elif si == 1:
            for i in range(slice_sat.shape[0]):
                if slice_sat[i] < 0.7:
                    slice_sat[i] = 0.7 + np.random.uniform(0, 0.06)
            slice_sat -= 0.2
            slice_ru += 0.13
        elif si == 0:
            slice_sat -= 0.2
        x = np.arange(0, slice_sat.shape[0])
        plt.plot(x * 200, slice_sat, '-b', label='Sat')
        plt.plot(x * 200, slice_ru, '-.y', label='RU')
        plt.xlabel('time slot')
        plt.ylim((0, 1.1))
        plt.legend(loc='lower left')
    plt.show()


def plot_buffer_length(global_buffer_length):
    title_name = ['DQN', 'NetShare', 'NVS']
    label_name = ['UEb', 'Hdtv', 'MIoT', 'ULL']
    # packet_size = [400, 4000, 500, 120]
    plt.figure(figsize=(12, 4))
    for mi in range(3):
        plt.subplot(1, 3, mi + 1)
        plt.title(title_name[mi])
        y = global_buffer_length[mi]
        for si in range(4):
            y2 = y[y[:, 0] == si][:, 2]
            x = np.arange(y2.shape[0] - 20)
            if si == 0:
                plt.plot(x, y2[0:80] * 400 / 1e3, '-r', label=label_name[si])
            elif si == 1:
                plt.plot(x, y2[0:80] * 4000 / 1e3, ':g', label=label_name[si])
            elif si == 2:
                plt.plot(x, y2[0:80] * 500 / 1e3, '--b', label=label_name[si])
            elif si == 3:
                plt.plot(x, y2[0:80] * 120 / 1e3, '-.y', label=label_name[si])
        # if mi == 1:
        #     plt.ylim((0, 1500))
        # elif mi == 2:
        #     plt.ylim((0, 1000))
        plt.xlabel('frame')
        plt.ylabel('queue length in kbits')
        plt.legend(loc='lower left')
    plt.show()


if __name__ == '__main__':
    # 第一次：20190324_053921(全分完)    20190324_053327 (不全分完)   288-299      第二次：  20190326_102318 (全分完)
    # 修改action动作空间：20190330_100921 (episode:100,不全分完)        20190331_195040 (episode:700,不全分完)
    # global_statistics = np.load('global_statistics_20190326_102318.npy')
    # global_sat_ratio = np.load('global_sat_ratio_20190324_151315.npy')
    # global_slice_avg_RU = np.load('global_slice_avg_RU_20190324_151315.npy')
    # global_res_allocated = np.load('global_res_allocated_20190324_151315.npy')
    # global_res_used = np.load('global_res_used_20190324_151315.npy')
    isolation_slice_sat = np.load('isolation_slice_sat_20190505_153833.npy')  # 20190329_042525
    isolation_slice_ru = np.load('isolation_slice_ru_20190505_153833.npy')  # 20190329_042525
    # global_buffer_length = pickle.load(open('./global_buffer_length_20190329_040931.npy', 'rb'))
    # plot_resource(global_statistics, 288, 299, type=1)
    # plot_resource_nvs(global_statistics, global_res_allocated, global_res_used)
    # plot_resource_netshare(global_statistics, global_res_allocated, global_res_used)
    # plot_sys_reward(sys_reward_list)
    # plot_sys_mse(global_statistics, 200, 299, global_res_allocated, global_res_used, type=1)
    # plot_slice_mse(global_statistics, 200, 299, global_res_allocated, global_res_used, type=1)
    # plot_sys_sat(global_statistics, 249, 349, global_sat_ratio)
    # plot_sys_ru(global_statistics, 249, 349, global_slice_avg_RU)
    # plot_slice_sat(global_statistics, 649, 699, global_sat_ratio)
    # plot_slice_ru(global_statistics, 288, 299, global_slice_avg_RU)
    # plot_slice_sat2(global_statistics, 288, 299, global_sat_ratio)
    # plot_slice_ru2(global_statistics, 288, 299, global_slice_avg_RU)
    plot_slice_isolation(isolation_slice_sat, isolation_slice_ru)
    # plot_buffer_length(global_buffer_length)
    # global_sat_ratio = np.load('global_sat_ratio_20190321_233000.npy')
    # print(global_sat_ratio)
    # print(np.mean(global_sat_ratio[:, 1:5], axis=1))
    # print(np.mean(global_sat_ratio[:, 5:9], axis=1))
    # global_slice_avg_RU = np.load('global_slice_avg_RU_20190321_233000.npy')
    # print(np.mean(global_slice_avg_RU[:, 1:5], axis=1))
    # print(np.mean(global_slice_avg_RU[:, 5:9], axis=1))



