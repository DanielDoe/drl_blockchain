import numpy as np
from DRL_Resouce_Allocation import utils
from DRL_Resouce_Allocation.RL_brain import DeepQNetwork
from DRL_Resouce_Allocation.env import SimulationEnv
from DRL_Resouce_Allocation.physical_resource import PhysicalResource
from DRL_Resouce_Allocation import nvs
from DRL_Resouce_Allocation import netshare
import pickle


def main():
    ue_num = [60, 11, 240, 124]
    r_s = [0.23, 0.255, 0.25, 0.137]  # [0.188, 0.255, 0.216, 0.137]
    r_s_dqn = [0.169, 0.288, 0.25, 0.15]  # [0.169, 0.288, 0.217, 0.15]
    env = SimulationEnv()
    TPs = env.generate_bs(bs_num=4)
    ues = env.generate_ue(ue_num=ue_num)

    user_qos = np.array([env.rate_demands, env.delay_demands, env.lamda_k, env.packet_size])
    global_buffer_length = {}
    index = 0
    for mi in range(4):

        if mi == 0:  # Dueling DQN
            r_sk = np.ones([4, 4])  # 资源比例初始化
            for si in range(4):
                r_sk[si, :] = r_s[si]
            # user association
            association_ues, rb_size = env.ue_association(admission_ues=ues, TPs=TPs)

            # generate all ue subframe index
            data_num = 100
            slice_user_seq, max_index = env.generate_subframe_index(association_ues, lamda_k=env.lamda_k,
                                                                    data_num=data_num, mi=2)
        elif mi == 1:  # DQN
            r_sk = np.ones([4, 4])  # 资源比例初始化
            for si in range(4):
                r_sk[si, :] = r_s_dqn[si]
            # user association
            association_ues, rb_size = env.ue_association(admission_ues=ues, TPs=TPs)

            # generate all ue subframe index
            data_num = 100
            slice_user_seq, max_index = env.generate_subframe_index(association_ues, lamda_k=env.lamda_k,
                                                                    data_num=data_num, mi=2)
        elif mi == 2:  # NetShare
            # user association
            association_ues, rb_size = env.ue_association(admission_ues=ues, TPs=TPs, mi=1)

            # generate all ue subframe index
            data_num = 100
            slice_user_seq, max_index = env.generate_subframe_index(association_ues, lamda_k=env.lamda_k,
                                                                    data_num=data_num, mi=2)
            r_sk, success = netshare.slice_ra(user_qos, association_ues, env.slice_num, env.bs_num,
                                              env.sub_channel_num, env.sub_frame_num)
            r_sk *= 0.8
        else:  # NVS
            # user association
            association_ues, rb_size = env.ue_association(admission_ues=ues, TPs=TPs, mi=1)

            # generate all ue subframe index
            data_num = 100
            slice_user_seq, max_index = env.generate_subframe_index(association_ues, lamda_k=env.lamda_k,
                                                                    data_num=data_num, mi=2)
            r_sk = nvs.slice_ra(env.rate_demands, association_ues, env.slice_num, env.bs_num)
            r_sk *= 0.8

        # physical resource allocation
        pr = PhysicalResource(TPs=TPs, user_qos=env.user_qos, env=env)
        slice_sat_ratio, slice_avg_RU, slice_bs_sat, slice_bs_RU, slice_spectral_efficiency, global_slice_queue_len = \
            pr.allocate(association_ues, r_sk, max_index, rb_size)
        global_buffer_length[mi] = global_slice_queue_len
    # 保存变量
    time_str = utils.cur_time()
    print('-----------------STATISTICAL VARIABLES %s HAVE SAVED-----------------' % time_str)
    # np.save('global_buffer_length_' + time_str + '.npy', global_buffer_length)
    pickle.dump(global_buffer_length, open('./global_buffer_length_' + time_str + '.npy', 'wb'))


if __name__ == '__main__':
    main()



