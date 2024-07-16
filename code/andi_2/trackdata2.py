from andi_datasets.models_phenom import models_phenom
import csv
# auxiliaries
import numpy as np
import matplotlib.pyplot as plt
from andi_datasets.utils_trajectories import plot_trajs
import random
import os
import argparse

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--K_low', default=-12, type=int)
    parser.add_argument('--K_high', default=6, type=int)
    parser.add_argument('--T_low', default=20, type=int)
    parser.add_argument('--T_high', default=200, type=int)
    parser.add_argument('--N_all', default=5000, type=int)
    # parser.add_argument('--K_high', default=6, type=int)
    args = parser.parse_args()
    return args


def generate_trajectories (N_all = 5000, T_min=50, T_max=200, L=1.8*128, D_max=1000, num_a_permodel=30, num_k_permodel=10000, N=100, log_range=None):
    '''
    # number of time steps per trajectory (frames)
    T_max = 200
    # number of trajectories       #长度在T_max和T_min内取值
    T_min = 200                #  全200的话就把这个也设置成200
    # Length of box (pixels)
    L = 1.8*128
    # diffusion coefficient (pixels^2 / frame)
    D_max = 1000     #diffusion coefficient从0-1000内随机取值，等间隔采样num_k_permodel   #这个可以仔细考虑一下  比赛里给的范围是0-10^6，均匀采样感觉范围太大了，一般D都不会超过100.目前的策略是0-1000均匀采样
    num_a_permodel=30   #diffusion coefficient从0-2内随机取值，等间隔采样num_a_permodel
    num_k_permodel=10000
    N_all = 5000
    N=100   #每确定一组参数（T，K，a）生成100条轨迹
    #
    '''
    K_log_range = [-5, 6] if log_range is None else log_range
    # exponents = [2 * i / num_a_permodel for i in range(0, num_a_permodel + 1)]
    # K_diffusion = [100 * i / num_k_permodel for i in range(0, num_k_permodel + 1)]
    exponents = [1.9 * i / num_a_permodel for i in range(1, num_a_permodel + 1)] + [1.9 + 0.09 * i / 6 for i in range(0, 6)]
    K_diffusion = [pow(10, K_log_range[0]) * pow(10, (K_log_range[1] - K_log_range[0]) * i / num_k_permodel) for i in range(0, num_k_permodel + 1)] # 1e-12 -- 1e6 之间取值
    T=[T_min+i for i in range(0, T_max-T_min + 1)]


    train_set= []  #  #用于存放轨迹的list    len(trajs)=N_all     第i行：   x1，x2，...,x_T; y1,y2,...,y_T; a1，a2，...,a_T; k1,k2,...,k_T    T=len(trajs[i])/4
    for i in range (int(N_all/N)):
        random_number_T = random.randint(0, T_max-T_min)
        T_1=T[random_number_T]

        random_number_a = random.randint(0, num_a_permodel)
        a_1 = exponents[random_number_a]
        random_number_a = random.randint(0, num_a_permodel)
        a_2 = exponents[random_number_a]
        random_number_a = random.randint(0, num_a_permodel)
        a_3 = exponents[random_number_a]

        random_number_k = random.randint(0, num_k_permodel)
        k_1 = K_diffusion[random_number_k]
        random_number_k = random.randint(0, num_k_permodel)
        k_2 = K_diffusion[random_number_k]
        random_number_k = random.randint(0, num_k_permodel)
        k_3 = K_diffusion[random_number_k]

        trajs_model2, labels_model2 = models_phenom().multi_state(N=N,
                                                                  L=L,
                                                                  T=T_1,
                                                                  epsilon_a=[0, 0],
                                                                  gamma_d=[1, 1],
                                                                  alphas=[[a_1, 0], [a_2, 0], [a_3, 0]],
                                                                  # Fixed alpha for each state
                                                                  Ds=[[k_1, 0], [k_2, 0], [k_3, 0]],
                                                                  # Mean and variance of each state
                                                                  M=[[0.99, 0.005, 0.005], [0.005, 0.99, 0.005],
                                                                     [0.005, 0.005, 0.99]]
                                                                  )
        #要把所有的轨迹concatenate起来
        # 使用transpose交换第一个和第二个维度
        trajs_tmp = trajs_model2.transpose(1, 0, 2)
        labels_tmp = labels_model2.transpose(1, 0, 2)
        trajs_tmp = trajs_tmp.tolist()
        labels_tmp = labels_tmp.tolist()

        for i in range (len(trajs_tmp)):
            traj_x = [item[0] for item in trajs_tmp[i]]
            traj_y = [item[1] for item in trajs_tmp[i]]
            label_a= [item[0] for item in labels_tmp[i]]
            label_k = [item[1] for item in labels_tmp[i]]
            label_state = [item[2] for item in labels_tmp[i]]
            traj_i = traj_x + traj_y +label_a + label_k + label_state
            train_set.append(traj_i)
    return train_set


args = get_args_parser()
print(args)

# number of time steps per trajectory (frames)
T_max = args.T_high
# number of trajectories       #长度在T_max和T_min内取值
T_min = args.T_low               #  全200的话就把这个也设置成200 

# 20-50 5w*1.1 50-100 20w*1.1 100-200 40w*1.1

# Length of box (pixels)
L = 1.8*128
# diffusion coefficient (pixels^2 / frame)
D_max = 1000     #diffusion coefficient从0-1000内随机取值，等间隔采样num_k_permodel   #这个可以仔细考虑一下  比赛里给的范围是0-10^6，均匀采样感觉范围太大了，一般D都不会超过100.目前的策略是0-1000均匀采样
num_a_permodel=30   #diffusion coefficient从0-2内随机取值，等间隔采样num_a_permodel
num_k_permodel=100000
N=1000   #每确定一组参数（T，K，a）生成100条轨迹


N_all_train = args.N_all
# N_all_test = 5e5
per_num = 5e3
ratio = 0.1
idx = 0

N_train = N_all_train
N_test = int(N_train * ratio)
train_set=generate_trajectories(N_train + N_test, T_min, T_max, L, D_max ,num_a_permodel, num_k_permodel,N, [args.K_low, args.K_high])

rand_idx = random.sample([i for i in range(0, N_train - 1)], N_test)
# print(rand_idx)
# print(len(train_set))
test_set = []
for i in rand_idx:
    # print('processing:', i, end=',')
    test_set.append(train_set[i])
train_idx = list(range(len(train_set)))
for i in rand_idx:
    # print('remove:', i, end=',')
    train_idx.remove(i)
train_set_new = []
for i in train_idx:
    train_set_new.append(train_set[i])
print(len(train_set_new))

# root = '../../data/'
root = '/data1/jiangy/andi_data/0626_multi_state'


# np.save(os.path.join(root, 'train', 'train_{}.npy'.format(idx)))
# np.save(os.path.join(root, 'test', 'test_{}.npy'.format(idx)))
with open(os.path.join(root, 'train{}_Nall_{}_T{}_{}_K{}_{}.csv'.format(idx, args.N_all, args.T_low, args.T_high, args.K_low, args.K_high)), 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    ttt = train_set_new
    writer.writerows(ttt)

with open(os.path.join(root, 'test{}_Nall_{}_T{}_{}_K{}_{}.csv'.format(idx, args.N_all, args.T_low, args.T_high, args.K_low, args.K_high)), 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    ttt = test_set
    writer.writerows(ttt)

idx = idx + per_num

# with open(
#         '{}/2d_train_set_{0}_Tmin_{1}_Tmax{2}_Dmax{3}.csv'.format(root, N_all_train, T_min, T_max, D_max),
#         'w', encoding='utf-8', newline='') as f:
#     writer = csv.writer(f)
#     # writer.writerow(['valid_loss','valid_task1_acc','valid_task2_acc'])
#     ttt = train_set_new
#     writer.writerows(ttt)


# with open(
#         '{}/2d_test_set_{0}_Tmin_{1}_Tmax{2}_Dmax{3}.csv'.format(root, N_all_test, T_min, T_max, D_max),
#         'w', encoding='utf-8', newline='') as f:
#     writer = csv.writer(f)
#     # writer.writerow(['valid_loss','valid_task1_acc','valid_task2_acc'])
#     ttt = test_set
#     writer.writerows(ttt)

# N_all_valid = 500
# N=10
# valid_set = generate_trajectories(N_all_valid, T_min, T_max, L, D_max, num_a_permodel, num_k_permodel, N)

# with open(
#         './datasets/andi_set/N5e4_1e2/2d_valid_set_{0}_Tmin_{1}_Tmax{2}_Dmax{3}.csv'.format(N_all_valid, T_min, T_max, D_max),
#         'w', encoding='utf-8', newline='') as f:
#     writer = csv.writer(f)
#     # writer.writerow(['valid_loss','valid_task1_acc','valid_task2_acc'])
#     ttt = valid_set
#     writer.writerows(ttt)

# N_all_test = 500
# N=10
# test_set = generate_trajectories(N_all_test, T_min, T_max, L, D_max, num_a_permodel, num_k_permodel, N)

# with open(
#         './datasets/andi_set/N5e4_1e2/2d_test_set_{0}_Tmin_{1}_Tmax{2}_Dmax{3}.csv'.format(N_all_test, T_min, T_max, D_max),
#         'w', encoding='utf-8', newline='') as f:
#     writer = csv.writer(f)
#     # writer.writerow(['valid_loss','valid_task1_acc','valid_task2_acc'])
#     ttt = test_set
#     writer.writerows(ttt)

a=0










#todo:分类网络的数据集我还得想想参数分布怎么设置 还没写好
'''

#分割以后，要进行segment的识别  长度 10-200  包含4个类别 第一个类别 0 iMobile 1 confine 2 diffusion 3 direct  每种都有N条
number_compartments = 50
radius_compartments = 10
compartments_center = models_phenom._distribute_circular_compartments(Nc = number_compartments,
                                                                      r = radius_compartments,
                                                                      L = L # size of the environment
                                                                      )

trajs_model5, labels_model5 = models_phenom().confinement(N = N,
                                                          L = L,
                                                          Ds = [1500*D, 50*D],
                                                          comp_center = compartments_center,
                                                          r = radius_compartments,
                                                          trans = 0.2 # Boundary transmittance
                                                           )

'''

