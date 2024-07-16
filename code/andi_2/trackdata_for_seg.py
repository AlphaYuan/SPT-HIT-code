from andi_datasets.models_phenom import models_phenom
import csv
# auxiliaries
import numpy as np
import matplotlib.pyplot as plt
from andi_datasets.utils_trajectories import plot_trajs
import random
import math


def find_inflection_point(arr_a, arr_k):
    inflection_points = []
    for i in range(1, len(arr_a)):
        if (arr_a[i] != arr_a[i - 1]) or (arr_k[i] != arr_k[i - 1]):
            inflection_points.append(i)
    if inflection_points:
        return inflection_points, len(inflection_points)
    else:
        return [0], 0


def generate_trajectories(N_all=500000, T_min=50, T_max=200, L=1.8 * 128, D_min=-3, D_max=3, num_a_permodel=30, N=100,
                          dataset='train'):
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

    exponents = [2 * i / num_a_permodel for i in range(0, num_a_permodel + 1)]
    # K_diffusion = [100*i / num_k_permodel for i in range(0, num_k_permodel + 1)]
    T = [T_min + i for i in range(0, T_max - T_min + 1)]

    train_set = []  # #用于存放轨迹的list    len(trajs)=N_all     第i行：   x1，x2，...,x_T; y1,y2,...,y_T; a1，a2，...,a_T; k1,k2,...,k_T    T=len(trajs[i])/4
    test_set = []
    count1t = 0
    count2t = 0
    for i in range(int(N_all / N)):
        random_number_T = random.randint(0, T_max - T_min)
        T_1 = T[random_number_T]

        random_number_a = random.randint(0, num_a_permodel)
        a_1 = exponents[random_number_a]
        random_number_a = random.randint(0, num_a_permodel)
        a_2 = exponents[random_number_a]
        random_number_a = random.randint(0, num_a_permodel)
        a_3 = exponents[random_number_a]

        # random_number_k = random.randint(0, num_k_permodel)
        k_1 = 10 ** random.uniform(D_min, D_max)
        # random_number_k = random.randint(0, num_k_permodel)
        k_2 = 10 ** random.uniform(D_min, D_max)
        # random_number_k = random.randint(0, num_k_permodel)
        k_3 = 10 ** random.uniform(D_min, D_max)

        trajs_model2, labels_model2 = models_phenom().multi_state(N=N,
                                                                  L=L,
                                                                  T=T_1,
                                                                  epsilon_a=[0, 0],
                                                                  gamma_d=[1, 1],
                                                                  alphas=[[a_1, 0.01], [a_2, 0.01], [a_3, 0.01]],
                                                                  # Fixed alpha for each state
                                                                  Ds=[[k_1, 0.1], [k_2, 0.1], [k_3, 0.1]],
                                                                  # Mean and variance of each state
                                                                  M=[[0.99, 0.005, 0.005], [0.005, 0.99, 0.005],
                                                                     [0.005, 0.005, 0.99]]
                                                                  )
        # 要把所有的轨迹concatenate起来
        # 使用transpose交换第一个和第二个维度
        trajs_tmp = trajs_model2.transpose(1, 0, 2)
        labels_tmp = labels_model2.transpose(1, 0, 2)
        trajs_tmp = trajs_tmp.tolist()
        labels_tmp = labels_tmp.tolist()

        if dataset != 'test':

            for i in range(len(trajs_tmp)):
                traj_x = [item[0] for item in trajs_tmp[i]]


                traj_y = [item[1] for item in trajs_tmp[i]]


                label_a = [item[0] for item in labels_tmp[i]]
                #label_k = [math.log10(item[1]) for item in labels_tmp[i]]
                label_k = [item[1] for item in labels_tmp[i]]

                traj_i = traj_x + traj_y + label_a + label_k + label_k

                # train_set.append(traj_i)

                label_klog = [math.log10(item[1]) for item in labels_tmp[i]]

                count1 = len([x for x in label_klog if x < -4])
                count2 = len([x for x in label_klog if x > -4])

                count1t = count1t+count1
                count2t = count2t+count2

                train_set.append(traj_i)

    print(count1t)
    print(count2t)
    return train_set


'''   else:
            label_cp_list=[]
            label_ncp_list=[]
            for i in range(len(trajs_tmp)):
                traj_x = [item[0] for item in trajs_tmp[i]]
                traj_y = [item[1] for item in trajs_tmp[i]]
                label_a = [item[0] for item in labels_tmp[i]]
                label_k = [math.log10(item[1]) for item in labels_tmp[i]]

                traj_i = traj_x + traj_y + label_a + label_k

                label_cp, label_ncp = find_inflection_point(label_a, label_k)



                test_set.append(traj_i)'''

# number of time steps per trajectory (frames)
T_max = 200
# number of trajectories       #长度在T_max和T_min内取值
T_min = 50  # 全200的话就把这个也设置成200
# Length of box (pixels)
L = 1.8 * 128
# diffusion coefficient (pixels^2 / frame)
D_max = 0  # diffusion coefficient从0-1000内随机取值，等间隔采样num_k_permodel   #这个可以仔细考虑一下  比赛里给的范围是0-10^6，均匀采样感觉范围太大了，一般D都不会超过100.目前的策略是0-1000均匀采样
D_min = -6
num_a_permodel = 30  # diffusion coefficient从0-2内随机取值，等间隔采样num_a_permodel
num_k_permodel = 10000
N = 100  # 每确定一组参数（T，K，a）生成100条轨迹

N_all_train = 8000
train_set = generate_trajectories(N_all_train, T_min, T_max, L, D_min, D_max, num_a_permodel, N, 'train')

'''trajs_model2, labels_model2 = models_phenom().multi_state(N=1,
                                                          L=L,
                                                          T=50,
                                                          epsilon_a=[0, 0],
                                                          gamma_d=[1, 1],
                                                          alphas=[[0.1, 0.01], [0.2, 0.01], [0.3, 0.01]],
                                                          # Fixed alpha for each state
                                                          Ds=[[0.0001, 0.1], [0.0001, 0.1], [0.0001, 0.1]],
                                                          # Mean and variance of each state
                                                          M=[[0.99, 0.005, 0.005], [0.005, 0.99, 0.005],
                                                             [0.005, 0.005, 0.99]]
                                                          )
labels_tmp = labels_model2.transpose(1, 0, 2)
labels_tmp = labels_tmp.tolist()'''



with open(
        '../../data/2d_train_set_{0}_Tmin_{1}_Tmax{2}_Dmax{3}.csv'.format(N_all_train, T_min, T_max,
                                                                                       D_max),
        'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    # writer.writerow(['valid_loss','valid_task1_acc','valid_task2_acc'])
    ttt = train_set
    writer.writerows(ttt)

N_all_valid = 400
valid_set = generate_trajectories(N_all_valid, T_min, T_max, L, D_min, D_max, num_a_permodel, N, 'train')

with open(
        '../../data/2d_valid_set_{0}_Tmin_{1}_Tmax{2}_Dmax{3}.csv'.format(N_all_valid, T_min, T_max,
                                                                                       D_max),
        'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    # writer.writerow(['valid_loss','valid_task1_acc','valid_task2_acc'])
    ttt = valid_set
    writer.writerows(ttt)






N_all_test = 1200
'''test_set,_ = generate_trajectories(N_all_test, T_min, T_max, L ,D_min, D_max, num_a_permodel,  N,dataset='test')

with open(
        './dataset/andi_set/2d_test_set_{0}_Tmin_{1}_Tmax{2}_Dmax{3}_diff.csv'.format(N_all_test, T_min, T_max, D_max),
        'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    # writer.writerow(['valid_loss','valid_task1_acc','valid_task2_acc'])
    ttt = test_set
    writer.writerows(ttt)'''

a = 0

# todo:分类网络的数据集我还得想想参数分布怎么设置 还没写好
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
                                                          trans = 0.2 # Boundary transmi
'''