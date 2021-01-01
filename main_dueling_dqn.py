# coding:utf8

import matplotlib.pyplot as plt
import numpy as np
from DRL_Resouce_Allocation import utils
from DRL_Resouce_Allocation.Dueling_DQN import DuelingDQN
from DRL_Resouce_Allocation.env import SimulationEnv
from DRL_Resouce_Allocation.physical_resource import PhysicalResource

MAX_EPISODES = 700
MAX_EP_STEPS = 7

# 统计量 [episode, ue_num[4], r_sk[4, 4], action[4], is_random, r_sk_[4, 4],
# sat[4], ru[4], reward[5], slice_bs_sat[4, 4], slice_bs_RU[4, 4], r_allocated[4], r_reserved[4],
# slice_spectral_efficiency[4]]
global_statistics = np.zeros([MAX_EPISODES * MAX_EP_STEPS, 99])
# action_space = [-0.3, -0.15, 0, 0.3, 0.5]
# action_space = [-0.4, -0.2, 0, 0.4, 0.6]
# action_space = [-0.4, -0.3, 0, 0.4, 0.7]
# action_space = [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5]  # 动作空间，调整切片资源的百分比
action_space = [-0.2, 0, 0.4]


# 训练
# @profile(precision=4, stream=open('memory_profiler.log', 'w+'))
def train():
    step = 0  # 用来控制什么时候学习
    sys_sat_list = []
    sys_RU_list = []
    sys_reward_list = []
    index = 0  # 统计变量数组的下标
    max_ue_num = [60, 11, 240, 124]  # 每类切片的最大用户数量，用来做归一化[180, 12, 220, 200]

    for episode in range(MAX_EPISODES):
        print('-----------NEW EPISODE %d STARTING-------------' % episode)
        # 初始化环境
        ud = 0
        slice1_ud = np.arange(1, 8)  # [0, 4, 4, 4, 8, 8, 8]
        ue_num = [4, 4, 100, 40]
        # ue_num = [60, 12, 240, 124]
        # ue_num = [130 + ud, 8 + int(np.floor(ud / 6)), 220 + ud, 0 + ud]
        env = SimulationEnv()
        TPs = env.generate_bs(bs_num=4)
        ues = env.generate_ue(ue_num=ue_num)

        r_sk = np.ones([4, 4]) * 0.1  # 资源比例初始化
        r_sk[1, :] = 0.2
        r_sk[2, :] = 0.13
        # r_sk = np.array([[0.04, 0.04, 0.04, 0.04],
        #                  [0.05, 0.05, 0.05, 0.05],
        #                  [0.06, 0.06, 0.06, 0.06],
        #                  [0.05, 0.05, 0.05, 0.05]])

        # user association
        association_ues, rb_size = env.ue_association(admission_ues=ues, TPs=TPs)

        # generate all ue subframe index
        data_num = 20
        slice_user_seq, max_index = env.generate_subframe_index(association_ues, lamda_k=env.lamda_k,
                                                                data_num=data_num)

        # physical resource allocation
        pr = PhysicalResource(TPs=TPs, user_qos=env.user_qos, env=env)
        slice_sat_ratio, slice_avg_RU, slice_bs_sat, slice_bs_RU, slice_spectral_efficiency, global_slice_queue_len = \
            pr.allocate(association_ues, r_sk, max_index, rb_size)

        # r_s = np.zeros(4)
        # for si in range(4):
        #     r_s[si] = np.sum(r_sk[si, :]) / 4
        lamda_k = np.array(env.lamda_k)
        packet_size = np.array(env.packet_size)
        slice_load = np.array(ue_num, dtype=np.float) / np.array(max_ue_num, dtype=np.float)  # 切片负载情况
        # for si in range(4):
        #     slice_load[si] /= np.sum(max_ue_num * lamda_k * packet_size)
        observation = np.concatenate((slice_load, slice_sat_ratio))  # 初始化状态

        sys_sat, sys_RU, sys_reward = [], [], []
        for j in range(MAX_EP_STEPS):
            # 刷新环境
            ud += 20
            ue_num = [4 + int(np.ceil(ud * 2 / 5)), 4 + slice1_ud[j], 100 + ud, 40 + int(np.ceil(ud * 3 / 5))]
            # ue_num = [4 + 6 * (j + 1), 4 + slice1_ud[j], 100 + ud, 40 + int(np.ceil(ud * 3 / 5))]
            ues = env.generate_ue(ue_num=ue_num)

            # user association
            association_ues, rb_size = env.ue_association(admission_ues=ues, TPs=TPs)

            # generate all ue subframe index
            data_num = 20
            slice_user_seq, max_index = env.generate_subframe_index(association_ues, lamda_k=env.lamda_k,
                                                                    data_num=data_num)

            # DQN 根据观测值选择行为
            action, is_random = RL.choose_action(observation)
            # action = np.random.randint(0, 4, 1)[0]

            # 环境根据行为给出下一个 state, reward
            observation_, reward, r_sk_, slice_bs_sat, slice_bs_RU, real_action, \
                r_allocated, r_reserved, slice_spectral_efficiency = pr.step(action, association_ues, r_sk.copy(),
                                                                             max_index, rb_size)

            slice_load = np.array(ue_num, dtype=np.float) / np.array(max_ue_num, dtype=np.float)  # 切片负载情况
            # for si in range(4):
            #     slice_load[si] /= np.sum(max_ue_num * lamda_k * packet_size)
            next_state = np.concatenate((slice_load, observation_[4:8]))  # 下一个状态
            # DQN存储记忆
            # min_reward = (np.min(observation_[4:8]) - 0.5) + (np.min(observation_[8:]) - 0.5)
            min_reward = np.min(reward[0:4])
            RL.store_transition(observation, action, min_reward, next_state)  # 修改为最小的reward

            # 控制学习起始时间和频率（先积累一些记忆再开始学习）
            if (step > 300) and (step % 2 == 0):
                RL.learn()

            # 记录统计量
            sys_sat.append(np.mean(observation_[4:8]))
            sys_RU.append(np.mean(observation_[8:12]))
            # sys_reward.append(reward)
            sys_reward.append(reward[4])
            global_statistics[index, :] = np.concatenate(([episode], ue_num, r_sk.flatten(),
                                                          real_action, [is_random],
                                                          r_sk_.flatten(), observation_[4:8], observation_[8:12],
                                                          reward, slice_bs_sat.flatten(), slice_bs_RU.flatten(),
                                                          r_allocated, r_reserved, slice_spectral_efficiency))

            # if (np.abs(observation - observation_) < 0.001).all():
            #     print('..............convergence...........')
            #     break

            # 将下一个 state_ 变为下次循环的state
            observation = next_state
            r_sk = r_sk_

            step += 1  # 总步数
            index += 1  # 下标自增

        sys_RU_list.append(np.mean(sys_RU))
        sys_sat_list.append(np.mean(sys_sat))
        sys_reward_list.append(np.mean(sys_reward))
    # 保存变量
    time_str = utils.cur_time()
    print('-----------------STATISTICAL VARIABLES %s HAVE SAVED-----------------' % time_str)
    np.save('sys_sat_list_' + time_str + '.npy', sys_sat_list)
    np.save('sys_RU_list_' + time_str + '.npy', sys_RU_list)
    np.save('sys_reward_list_' + time_str + '.npy', sys_reward_list)
    np.save('cost_his_' + time_str + '.npy', RL.cost_his)
    np.save('global_statistics_' + time_str + '.npy', global_statistics)
    # 保存模型
    RL.save(time_str)
    print('-----------------MODEL HAS SAVED-----------------')

    # plot(sys_RU_list, sys_sat_list)
    # end of train
    print('-----------------TRAIN OVER--------------------')


if __name__ == '__main__':
    RL = DuelingDQN(n_actions=81,
                    n_features=8,
                    learning_rate=0.01,
                    reward_decay=0.9,
                    e_greedy=0.93,
                    replace_target_iter=100,  # 每 100 步替换一次 target_net 的参数
                    memory_size=6000,  # 记忆上限
                    e_greedy_increment=0.005,
                    batch_size=60
                    )
    train()


