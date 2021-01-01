import numpy as np
from DRL_Resouce_Allocation import utils
from DRL_Resouce_Allocation.RL_brain import DeepQNetwork
from DRL_Resouce_Allocation.env import SimulationEnv
from DRL_Resouce_Allocation.physical_resource import PhysicalResource


MAX_EPISODES = 30


def main():
    ue_num = [60, 11, 240, 124]
    r_s = [0.23, 0.255, 0.25, 0.137]  # [0.23, 0.3, 0.26, 0.21]

    # ue_num = [36, 8, 180, 88]
    # ue_num = [44, 9, 200, 100]
    # ue_num = [52, 10, 220, 112]
    # ue_num = [60, 11, 240, 124]

    # r_s = [0.2165, 0.3712, 0.2123, 0.2]
    # r_s = [0.2204, 0.3632, 0.2163, 0.2]
    # r_s = [0.2298, 0.3323, 0.2379, 0.2]
    # r_s = [0.23, 0.3107, 0.2593, 0.2]
    isolation_slice_sat = np.zeros([MAX_EPISODES, 4])
    isolation_slice_ru = np.zeros([MAX_EPISODES, 4])
    for episode in range(MAX_EPISODES):
        env = SimulationEnv()
        TPs = env.generate_bs(bs_num=4)
        if episode > 10:
            ue_num[2] = 420
        ues = env.generate_ue(ue_num=ue_num)
        r_sk = np.ones([4, 4])  # 资源比例初始化
        for si in range(4):
            r_sk[si, :] = r_s[si]

        # user association
        association_ues, rb_size = env.ue_association(admission_ues=ues, TPs=TPs)

        # generate all ue subframe index
        data_num = 20
        slice_user_seq, max_index = env.generate_subframe_index(association_ues, lamda_k=env.lamda_k,
                                                                data_num=data_num, mi=1)

        # physical resource allocation
        pr = PhysicalResource(TPs=TPs, user_qos=env.user_qos, env=env)
        slice_sat_ratio, slice_avg_RU, slice_bs_sat, slice_bs_RU, slice_spectral_efficiency = pr.allocate_isolation(
            association_ues, r_sk, max_index, rb_size)
        print(slice_sat_ratio)
        print(slice_avg_RU)

        isolation_slice_sat[episode, :] = slice_sat_ratio
        isolation_slice_ru[episode, :] = slice_avg_RU
    # 保存变量
    time_str = utils.cur_time()
    print('-----------------STATISTICAL VARIABLES %s HAVE SAVED-----------------' % time_str)
    np.save('isolation_slice_sat_' + time_str + '.npy', isolation_slice_sat)
    np.save('isolation_slice_ru_' + time_str + '.npy', isolation_slice_ru)


if __name__ == '__main__':
    main()

