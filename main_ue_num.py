from DRL_Resouce_Allocation import nvs
from DRL_Resouce_Allocation import netshare
from DRL_Resouce_Allocation.env import SimulationEnv
from DRL_Resouce_Allocation.physical_resource import PhysicalResource
from DRL_Resouce_Allocation import utils
import numpy as np


# 用户数量改变
def ue_num_change():
    slice1_ud = np.arange(1, 8)  # [0, 4, 4, 4, 8, 8, 8]
    ud = 0
    global_sat_ratio = np.zeros([7, 9])
    global_slice_avg_RU = np.zeros([7, 9])
    global_res_allocated = np.zeros([7, 9])
    global_res_used = np.zeros([7, 9])

    for j in range(7):
        ud += 20
        ue_num = [4 + int(np.ceil(ud * 2 / 5)), 4 + slice1_ud[j], 100 + ud, 40 + int(np.ceil(ud * 3 / 5))]
        env = SimulationEnv()
        user_qos = np.array([env.rate_demands, env.delay_demands, env.lamda_k, env.packet_size])
        # user_qos[0, 2] = 100

        print('-----------NEW EPISODE %d STARTING-------------' % j)

        # [nvs, netShare]
        temp_slice_sat_ratio = np.zeros([3, 8])
        temp_slice_avg_RU = np.zeros([3, 8])
        temp_res_allocated = np.zeros([3, 8])
        temp_res_used = np.zeros([3, 8])

        # 取消位置的影响
        for ci in range(3):
            TPs = env.generate_bs(bs_num=4)
            ues = env.generate_ue(ue_num=ue_num)

            for mi in range(2):
                # user association
                association_ues, rb_size = env.ue_association(admission_ues=ues, TPs=TPs, mi=mi)

                # generate all ue subframe index
                data_num = 20
                slice_user_seq, max_index = env.generate_subframe_index(association_ues, lamda_k=env.lamda_k,
                                                                        data_num=data_num)
                # two methods of slice resource allocation
                if mi == 0:
                    r_sk = nvs.slice_ra(env.rate_demands, association_ues, env.slice_num, env.bs_num)
                else:
                    r_sk, success = netshare.slice_ra(user_qos, association_ues, env.slice_num, env.bs_num,
                                                      env.sub_channel_num, env.sub_frame_num)

                r_sk *= 0.75  # 修改为allocated的结果
                # physical resource allocation
                pr = PhysicalResource(TPs=TPs, user_qos=env.user_qos, env=env)
                slice_sat_ratio, slice_avg_RU, slice_bs_sat, slice_bs_RU, slice_spectral_efficiency, global_slice_queue_len = \
                    pr.allocate(association_ues, r_sk.copy(), max_index, rb_size)

                r_allocated = np.sum(r_sk, axis=1) / 4  # allocated resource
                r_used = r_allocated * slice_avg_RU  # used resource

                offset = mi * 4
                temp_slice_sat_ratio[ci, offset:offset + 4] = slice_sat_ratio
                temp_slice_avg_RU[ci, offset:offset + 4] = slice_avg_RU
                temp_res_allocated[ci, offset:offset + 4] = r_allocated
                temp_res_used[ci, offset:offset + 4] = r_used

        global_sat_ratio[j] = np.concatenate(([np.sum(ue_num)], np.mean(temp_slice_sat_ratio, axis=0)))
        global_slice_avg_RU[j] = np.concatenate(([np.sum(ue_num)], np.mean(temp_slice_avg_RU, axis=0)))
        global_res_allocated[j] = np.concatenate(([np.sum(ue_num)], np.mean(temp_res_allocated, axis=0)))
        global_res_used[j] = np.concatenate(([np.sum(ue_num)], np.mean(temp_res_used, axis=0)))

    time_str = utils.cur_time()
    print('-----------------STATISTICAL VARIABLES %s HAVE SAVED-----------------' % time_str)
    np.save('global_sat_ratio_' + time_str + '.npy', global_sat_ratio)
    np.save('global_slice_avg_RU_' + time_str + '.npy', global_slice_avg_RU)
    np.save('global_res_allocated_' + time_str + '.npy', global_res_allocated)
    np.save('global_res_used_' + time_str + '.npy', global_res_used)


if __name__ == '__main__':
    ue_num_change()


