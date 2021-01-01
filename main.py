from env import SimulationEnv
from physical_resource import PhysicalResource
from RL_brain import DeepQNetwork
import math
import numpy as np
import matplotlib.pyplot as plt

MAX_EPISODES = 5
MAX_EP_STEPS = 10


# 训练
def train():
    step = 0  # 用来控制什么时候学习
    sys_sat_list = []
    sys_RU_list = []

    for episode in range(MAX_EPISODES):
        # 初始化环境
        ud = 0
        ue_num = [130, 8, 200 + ud, 12]
        env = SimulationEnv()
        TPs = env.generate_bs(bs_num=4)
        ues = env.generate_ue(ue_num=ue_num)

        r_sk = np.ones([4, 4]) * 0.25  # 等比例初始化

        # user association
        association_ues = env.ue_association(admission_ues=ues, TPs=TPs)
        #print(association_ues[0])

        # generate all ue subframe index
        data_num = 20
        slice_user_seq, max_index = env.generate_subframe_index(association_ues, lamda_k=env.lamda_k,
                                                                data_num=data_num)

        # physical resource allocation
        pr = PhysicalResource(TPs=TPs, user_qos=env.user_qos, env=env)
        slice_sat_ratio, slice_avg_RU, slice_bs_sat, slice_bs_RU, slice_spectral_efficiency, global_slice_queue_len = pr.allocate(association_ues[0], r_sk, max_index, association_ues[1])

        r_s = np.zeros(4)
        for si in range(4):
            r_s[si] = np.sum(r_sk[si, :]) / 4
        observation = np.concatenate((r_s, slice_sat_ratio, slice_avg_RU))  # 初始化状态

        sys_sat, sys_RU = 0, 0
        for j in range(MAX_EP_STEPS):
            # 刷新环境
            ud += 12
            ue_num = [130, 8, 200 + ud, 12]
            ues = env.generate_ue(ue_num=ue_num)

            # user association
            association_ues = env.ue_association(admission_ues=ues, TPs=TPs)

            # generate all ue subframe index
            data_num = 20
            slice_user_seq, max_index = env.generate_subframe_index(association_ues, lamda_k=env.lamda_k,
                                                                    data_num=data_num)

            # DQN 根据观测值选择行为
            action = RL.choose_action(observation)

            # 环境根据行为给出下一个 state, reward, 是否终止
            observation_, reward, r_sk_allocated, slice_bs_sat, slice_bs_RU, real_action, r_allocated, r_reserved, slice_spectral_efficiency = pr.step(action, association_ues[0], r_sk, max_index, association_ues[1])

            # DQN存储记忆
            RL.store_transition(observation, action, reward, observation_)

            # 控制学习起始时间和频率（先积累一些记忆再开始学习）
            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # 将下一个 state_ 变为下次循环的state
            observation = observation_
            r_sk = r_sk_allocated

            sys_sat += np.mean(observation_[4:8])
            sys_RU += np.mean(observation_[8:])

            step += 1  # 总步数

        sys_RU_list.append(sys_RU / MAX_EP_STEPS)
        sys_sat_list.append(sys_sat / MAX_EP_STEPS)

    plot(sys_RU_list, sys_sat_list)
    # end of train
    print('train over')


def plot(sys_RU_list, sys_sat_list):
    x1 = range(len(sys_RU_list))
    x2 = range(len(sys_sat_list))
    y1 = sys_RU_list
    y2 = sys_sat_list
    plt.subplot(2, 1, 1)
    plt.plot(x1, y1, '.-')
    plt.title('resource utilization')
    plt.ylabel('ru')

    plt.subplot(2, 1, 2)
    plt.plot(x2, y2, '.-')
    plt.title('system satisfaction')
    plt.ylabel('sat')


def main():
    env = SimulationEnv()
    TPs = env.generate_bs(bs_num=4)
    for ud in range(10, 280, 20):
        ue_num = [130 + ud, 8 + math.floor(ud / 6), 220 + ud, 0 + ud]

        # 每个方法在每种用户数量的情况下跑10次取平均值，从而取消用户位置的影响
        for ci in range(1):
            # generate UEs
            ues = env.generate_ue(ue_num=ue_num)

            # user association
            association_ues = env.ue_association(admission_ues=ues, TPs=TPs)

            # generate all ue subframe index
            data_num = 70
            slice_user_seq, max_index = env.generate_subframe_index(association_ues, lamda_k=env.lamda_k,
                                                                    data_num=data_num)

            # virtual resource allocation
            r_sk = np.ones([4, 4]) * 0.25

            # physical resource allocation
            total_subframe = max_index
            pr = PhysicalResource(TPs=TPs, user_qos=env.user_qos, env=env)
            slice_sat_ratio, slice_avg_RU = pr.allocate(association_ues, r_sk, total_subframe)

            print('slice_sat_ratio: ', slice_sat_ratio, ' \n slice_avg_RU: ', slice_avg_RU)
            print('-------end----')


if __name__ == '__main__':
    # main()
    RL = DeepQNetwork(n_actions=256, n_features=12,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,  # 每 200 步替换一次 target_net 的参数
                      memory_size=3000,  # 记忆上限
                      # output_graph=True   # 是否输出 tensorboard 文件
                      )
    train()

