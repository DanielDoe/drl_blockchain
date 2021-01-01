import numpy as np
import matplotlib.pyplot as plt


def plot_statistics(global_statistics, episode):
    y = global_statistics[global_statistics[:, 0] == episode]
    title_name = ['UEb', 'Hdtv', 'MIoT', 'ULL']
    for slice_id in range(4):
        plt.subplot(2, 2, slice_id + 1)
        plt.title(title_name[slice_id] + ': Episode ' + str(episode + 1))
        x = y[:, 1 + slice_id]
        offset = 26 + slice_id * 4
        r_sk = y[:, offset:(offset + 4)]  # r_sk_为执行action之后的资源比例
        r_s = np.zeros(x.shape[0])
        r_used = np.zeros(x.shape[0])  # used resource
        for i in range(x.shape[0]):
            r_s[i] = np.sum(r_sk[i, :]) / 4
            r_used[i] = r_s[i] * y[i, 46 + slice_id]
        if slice_id == 1:
            x = range(8)
        r_allocated = y[:, 87 + slice_id]
        r_reserved = y[:, 91 + slice_id]
        plt.plot(x, r_allocated, '.-', label='Allocated')  # r_allocated
        plt.plot(x, r_used, 'o-', label='Used')  # r_used
        plt.plot(x, r_reserved, '*-', label='Reserved')  # r_reserved
        plt.plot(x, y[:, 21 + slice_id], '+-', label='Action')  # action
        plt.plot(x, y[:, 42 + slice_id], 's-', label='Sat')  # sat
        plt.plot(x, y[:, 46 + slice_id], 'D-', label='RU')  # ru
        plt.plot(x, y[:, 50 + slice_id] / 10, 'x-', label='Reward')  # reward

        if slice_id != 1:
            plt.xticks(x, x)
        # plt.ylim((-1, 1))
        # plt.xlim((50, 260))
        plt.legend(loc='upper left')  # 左上角
        if slice_id > 1:
            plt.xlabel('ue num')
    plt.show()


def plot_resource(global_statistics, start, end):
    y = global_statistics[(global_statistics[:, 0] >= start) & (global_statistics[:, 0] <= end)]
    episode_num = end - start + 1
    title_name = ['UEb', 'Hdtv', 'MIoT', 'ULL']
    for si in range(4):
        plt.subplot(2, 2, si + 1)
        plt.title(title_name[si] + ': Episode ' + str(start) + '~' + str(end))
        # allocated and used resource fraction
        res_allocated = np.zeros(episode_num)
        res_used = np.zeros(episode_num)
        res_reserved = np.zeros(episode_num)
        index = 0
        offset = 26 + si * 4
        for ei in range(start, end + 1):
            y2 = y[y[:, 0] == ei]  # episode == ei
            r_sk = y2[:, offset:(offset + 4)]  # r_sk_为执行action之后的资源比例
            r_s = np.zeros(y2.shape[0])
            r_s_used = np.zeros(y2.shape[0])
            for j in range(y2.shape[0]):
                r_s[j] = np.sum(r_sk[j, :]) / 4
                r_s_used[j] = r_s[j] * y2[j, 46 + si]
            res_allocated[index] = np.mean(y2[:, 87 + si])
            res_used[index] = np.mean(r_s_used)
            res_reserved[index] = np.mean(y2[:, 91 + si])
            res_reserved[index] += res_allocated[index]
            index += 1
        x = np.arange(start, end + 1)
        plt.plot(x, res_allocated, '-', label='Allocated')  # Allocated Resource
        plt.plot(x, res_used, '-', label='Used')  # Used Resource
        plt.plot(x, res_reserved, '-.', label='Reserved')  # Reserved Resource
        plt.ylim((0, 1))
        plt.xlim((start, end))
        if si > 1:
            plt.xlabel('episode')
        plt.legend(loc='upper left')
    plt.show()


# index = [51, 66]
def plot_bs_sat(global_statistics, start, end, slice_id):
    y = global_statistics[(global_statistics[:, 0] >= start) & (global_statistics[:, 0] <= end)]
    episode_num = end - start + 1
    # bs level sat ratio
    bs_sat = np.zeros([4, episode_num])
    index = 0
    offset = 51 + slice_id * 4  # 偏移量
    for ei in range(start, end + 1):
        y2 = y[y[:, 0] == ei]  # episode == ei
        bs_sat[0][index] = np.mean(y2[:, offset])
        bs_sat[1][index] = np.mean(y2[:, offset + 1])
        bs_sat[2][index] = np.mean(y2[:, offset + 2])
        bs_sat[3][index] = np.mean(y2[:, offset + 3])
        index += 1
    x = range(start, end + 1)
    for bi in range(4):
        plt.subplot(2, 2, bi + 1)
        plt.plot(x, bs_sat[bi], '.-')
        plt.title('BS-' + str(bi))
        plt.ylabel('bs sat')
        if bi > 1:
            plt.xlabel('episode')
        plt.ylim((0, 1))
    plt.show()


# index = [67, 82]
def plot_bs_ru(global_statistics, start, end, slice_id):
    y = global_statistics[(global_statistics[:, 0] >= start) & (global_statistics[:, 0] <= end)]
    episode_num = end - start + 1
    # bs level sat ratio
    bs_sat = np.zeros([4, episode_num])
    index = 0
    offset = 67 + slice_id * 4  # 偏移量
    for ei in range(start, end + 1):
        y2 = y[y[:, 0] == ei]  # episode == ei
        bs_sat[0][index] = np.mean(y2[:, offset])
        bs_sat[1][index] = np.mean(y2[:, offset + 1])
        bs_sat[2][index] = np.mean(y2[:, offset + 2])
        bs_sat[3][index] = np.mean(y2[:, offset + 3])
        index += 1
    x = range(start, end + 1)
    for bi in range(4):
        plt.subplot(2, 2, bi + 1)
        plt.plot(x, bs_sat[bi], '.-')
        plt.title('BS-' + str(bi))
        plt.ylabel('bs RU')
        if bi > 1:
            plt.xlabel('episode')
        # plt.ylim((0, 1))
    plt.show()


# index = [21, 24]
def plot_action(global_statistics):
    title_name = ['UEb', 'Hdtv', 'MIoT', 'ULL']
    for si in range(4):
        plt.subplot(2, 2, si + 1)
        plt.title(title_name[si])
        if si == 1:
            ue_num = 8
        else:
            ue_num = 100
        y = global_statistics[global_statistics[:, 1 + si] == ue_num]
        x = y[:, 0]
        plt.plot(x, y[:, 21 + si], '.-')
        plt.title(title_name[si] + ': ue_num = ' + str(ue_num))
        plt.ylim((-1, 1))
        # plt.xlim((0, 500))
        if si > 1:
            plt.xlabel('episode')
        plt.ylabel('action')
    plt.show()


# index = [42, 45]
def plot_sat(global_statistics, slice_id):
    y = global_statistics[global_statistics[:, 1] == 180]
    x = y[:, 0]
    plt.plot(x, y[:, 42 + slice_id], '.-')
    plt.title('ue_num = 80')
    plt.ylim((0, 1))
    plt.xlim((0, 500))
    plt.xlabel('episode')
    plt.ylabel('sat')
    plt.show()


# index = [5, 20]
def plot_rs(global_staticstics, slice_id):
    y = global_staticstics[global_staticstics[:, 1] == 60]
    x = y[:, 0]
    offset = 5 + slice_id * 4
    r_sk = y[:, offset:(offset + 4)]
    r_s = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        r_s[i] = np.sum(r_sk[i, :]) / 4
    plt.plot(x, r_s, '.-')
    plt.title('ue_num = 80')
    plt.ylim((0, 1))
    plt.xlim((0, 500))
    plt.xlabel('episode')
    plt.ylabel('resource')
    plt.show()


def plot_cost(cost_his):
    y = cost_his
    x = range(len(y))
    plt.plot(x, y, '-')
    # plt.ylim((0, 1))
    # plt.xlim((0, 100))
    plt.xlabel('training steps')
    plt.ylabel('loss')
    plt.show()


def plot_sys_reward(sys_reward_list):
    y = sys_reward_list
    x = range(len(y))
    # y2 = np.zeros(28)
    # x = range(28)
    # for ei in range(28):
    #     start = ei * 25
    #     end = start + 25
    #     y2[ei] = np.max(y[start:end])
    plt.plot(x, y, '.-')
    # plt.ylim((0, 3.5))
    # plt.xlim((0, 100))
    plt.xlabel('episode')
    plt.ylabel('system reward')
    plt.show()


def plot_slice_reward(global_statistics):
    episode_num = int(global_statistics.shape[0] / 7)  # 回合数量
    title_name = ['UEb', 'Hdtv', 'MIoT', 'ULL']
    for si in range(4):
        plt.subplot(2, 2, si + 1)
        plt.title(title_name[si])
        y = np.zeros(episode_num)
        for ei in range(episode_num):
            slice_statistics = global_statistics[global_statistics[:, 0] == ei]
            slice_reward = np.mean(slice_statistics[:, 50 + si])
            y[ei] = slice_reward
        x = np.arange(1, episode_num + 1)
        plt.plot(x, y, '.-')
        plt.ylabel('reward')
        if si > 1:
            plt.xlabel('episode')
        plt.ylim((-1, 1))
    plt.show()


def plot_sys_sat(sys_sat_list):
    y = sys_sat_list
    x = range(len(y))
    plt.plot(x, y, '.-')
    plt.ylim((0, 1))
    # plt.xlim((0, 800))
    plt.xlabel('episode')
    plt.ylabel('system sat')
    plt.show()


def plot_slice_sat(global_statistics):
    episode_num = int(global_statistics.shape[0] / 7)  # 回合数量
    title_name = ['UEb', 'Hdtv', 'MIoT', 'ULL']
    for si in range(4):
        plt.subplot(2, 2, si + 1)
        plt.title(title_name[si])
        y = np.zeros(episode_num)
        for ei in range(episode_num):
            slice_statistics = global_statistics[global_statistics[:, 0] == ei]
            slice_sat = np.mean(slice_statistics[:, 42 + si])
            y[ei] = slice_sat
        x = np.arange(1, episode_num + 1)
        plt.plot(x, y, '.-')
        plt.ylabel('sat ratio')
        if si > 1:
            plt.xlabel('episode')
        plt.ylim((0, 1))
    plt.show()


def plot_sys_ru(sys_ru_list):
    y = sys_ru_list
    x = range(len(y))
    plt.plot(x, y, '.-')
    plt.ylim((0.4, 1))
    # plt.xlim((0, 100))
    plt.xlabel('episode')
    plt.ylabel('system ru')
    plt.show()


def plot_slice_ru(global_statistics):
    episode_num = int(global_statistics.shape[0] / 7)  # 回合数量
    title_name = ['UEb', 'Hdtv', 'MIoT', 'ULL']
    for si in range(4):
        plt.subplot(2, 2, si + 1)
        plt.title(title_name[si])
        y = np.zeros(episode_num)
        for ei in range(episode_num):
            slice_statistics = global_statistics[global_statistics[:, 0] == ei]
            slice_ru = np.mean(slice_statistics[:, 46 + si])
            y[ei] = slice_ru
        x = np.arange(1, episode_num + 1)
        plt.plot(x, y, '.-')
        plt.ylabel('ru')
        if si > 1:
            plt.xlabel('episode')
        plt.ylim((0, 1))
    plt.show()


# 分别统计不同用户数量下满意度>0.5的概率
def plot_sat_probability(global_statistics):
    ue_num = [60, 80, 100, 120, 140, 160, 180, 200, 220, 240]
    prob_list = []
    for i in range(len(ue_num)):
        y = global_statistics[global_statistics[:, 1] == ue_num[i]][:, 12]
        prob = y[y > 0.5].shape[0] / y.shape[0]
        prob_list.append(prob)
    x = range(len(ue_num))
    plt.bar(x, prob_list, width=0.5, tick_label=ue_num)
    plt.xlabel('ue num')
    plt.ylabel('sat ratio')
    plt.ylim((0, 1))
    # plt.xlim((0, 100))
    plt.show()


# 分别统计不同用户数量下的平均资源利用率
def plot_avg_ru(global_statistics):
    ue_num = [60, 80, 100, 120, 140, 160, 180, 200, 220, 240]
    ru_list = []
    for i in range(len(ue_num)):
        y = global_statistics[global_statistics[:, 1] == ue_num[i]][:, 13]
        ru_list.append(np.mean(y))
    x = range(len(ue_num))
    plt.bar(x, ru_list, width=0.5, tick_label=ue_num)
    plt.xlabel('ue num')
    plt.ylabel('average RU')
    plt.ylim((0, 1))
    # plt.xlim((0, 100))
    plt.show()


def plot_sys_reward2(global_statistics):
    episode_num = int(global_statistics.shape[0] / 7)  # 回合数量
    sys_reward_arr = np.zeros(episode_num)
    for ei in range(episode_num):
        y = global_statistics[global_statistics[:, 0] == ei]
        sys_reward_arr[ei] = np.mean(np.min(y[:, 50:54], axis=1))
    x = range(episode_num)
    plt.plot(x, sys_reward_arr, '*')
    # plt.ylim((0, 1))
    # plt.xlim((0, 100))
    plt.xlabel('episode')
    plt.ylabel('system reward')
    plt.show()


if __name__ == '__main__':
    global_statistics = np.load('global_statistics_20190503_232152.npy')
    cost_his = np.load('cost_his_20190429_132251.npy')
    sys_reward_list = np.load('sys_reward_list_20190503_232152.npy')
    sys_ru_list = np.load('sys_ru_list_20190503_232152.npy')
    sys_sat_list = np.load('sys_sat_list_20190503_232152.npy')
    # plot_statistics(global_statistics, 179)
    # plot_resource(global_statistics, 299, 499)
    # plot_bs_sat(global_statistics, 0, 349)
    # plot_bs_ru(global_statistics, 0, 299)
    # plot_action(global_statistics)
    # plot_sat(global_statistics)
    # plot_rs(global_statistics)
    # plot_cost(cost_his)  # 1
    plot_sys_reward(sys_reward_list)  # 2
    # plot_sys_reward2(global_statistics)  # unused
    # plot_sys_sat(sys_sat_list)  # 3
    # plot_sys_ru(sys_ru_list)  # 4
    # plot_sat_probability(global_statistics)
    # plot_slice_reward(global_statistics)  # 5
    # plot_slice_sat(global_statistics)  # 6
    # plot_slice_ru(global_statistics)  # 7

