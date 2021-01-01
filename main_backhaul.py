from DRL_Resouce_Allocation import nvs
from DRL_Resouce_Allocation import netshare
from DRL_Resouce_Allocation.env import SimulationEnv
from DRL_Resouce_Allocation.physical_resource import PhysicalResource
from DRL_Resouce_Allocation import utils
import numpy as np


def backhaul_change():
    env = SimulationEnv()
    TPs = env.generate_bs(bs_num=4)
    user_qos = np.array([env.rate_demands, env.delay_demands, env.lamda_k, env.packet_size])
    # statistic variables
    global_sat_ratio = np.zeros([20, 10])
    global_slice_avg_RU = np.zeros([20, 10])
    index = 0

    for bh in range(10):
        # change the backhaul of BS
        TPs['backhaul'] = np.ones(4) * bh * 1000 * 1000  # bps

        # light load and heavy load
        for ui in range(2):
            if ui == 0:
                ue_num = [20, 8, 140, 64]
            else:
                ue_num = [52, 12, 220, 112]

            # [nvs, netShare, dqn]
            temp_slice_sat_ratio = np.zeros([2, 12])
            temp_slice_avg_RU = np.zeros([2, 12])

            # 消除位置的影响
            for ci in range(2):
                ues = env.generate_ue(ue_num=ue_num)

                for mi in range(2):
                    # user association
                    association_ues, rb_size = env.ue_association(admission_ues=ues, TPs=TPs)

                    # generate all ue subframe index
                    data_num = 20
                    slice_user_seq, max_index = env.generate_subframe_index(association_ues, lamda_k=env.lamda_k,
                                                                            data_num=data_num)
                    if mi == 0:
                        r_sk = nvs.slice_ra(env.rate_demands, association_ues, env.slice_num, env.bs_num)
                    else:
                        r_sk, success = netshare.slice_ra(user_qos, association_ues, env.slice_num, env.bs_num,
                                                 env.sub_channel_num, env.sub_frame_num)

                    # physical resource allocation
                    pr = PhysicalResource(TPs=TPs, user_qos=env.user_qos, env=env)
                    slice_sat_ratio, slice_avg_RU, slice_bs_sat, slice_bs_RU = pr.allocate(association_ues, r_sk,
                                                                                           max_index, rb_size)
                    offset = mi * 4
                    temp_slice_sat_ratio[ci, offset:offset + 4] = slice_sat_ratio
                    temp_slice_avg_RU[ci, offset:offset + 4] = slice_avg_RU

            global_sat_ratio[index] = np.concatenate(([bh, np.sum(ue_num)], np.mean(temp_slice_sat_ratio, axis=0)))
            global_slice_avg_RU[index] = np.concatenate(([bh, np.sum(ue_num)], np.mean(temp_slice_avg_RU, axis=0)))
            index += 1

    time_str = utils.cur_time()
    print('-----------------STATISTICAL VARIABLES %s HAVE SAVED-----------------' % time_str)
    np.save('global_sat_ratio_' + time_str + '.npy', global_sat_ratio)
    np.save('global_slice_avg_RU_' + time_str + '.npy', global_slice_avg_RU)


if __name__ == '__main__':
    backhaul_change()

