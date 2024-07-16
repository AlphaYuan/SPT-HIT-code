import numpy as np
import ruptures as rpt
import os
from andi_datasets.models_phenom import models_phenom
from andi_datasets.utils_trajectories import plot_trajs
from andi_datasets.utils_challenge import label_continuous_to_list
from test_for_submit import merge_changepoints
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from copy import deepcopy
import csv


preds_all = np.load('../../challenge_results/daR/track_2_all_preds_all.npy', allow_pickle=True)
pad_all = np.load('../../challenge_results/daR/track_2_all_pad_all.npy', allow_pickle=True)

num_exp=12
exp_range = range (num_exp)
num_fov=30
fov_range = range (num_fov)

a_var = [[[] for j in fov_range] for i in exp_range]
k_var = [[[] for j in fov_range] for i in exp_range]
all_len = []
all_log_k = []

for i in exp_range:
    for j in fov_range:
        pred = preds_all[i][j]
        tmp_a = []
        tmp_k = []
        tmp_cp = []
        tmp_cp_cnt = []
        for idx in range(pred.shape[0]):
            count_true = np.sum(pad_all[i][j][idx])
            print(count_true)
            all_len.append(count_true)
            all_log_k += pred[idx, :count_true, 1].tolist()
            var_a = np.var(pred[idx, :count_true, 0])
            var_k = np.var(pred[idx, :count_true, 1])
            tmp_a.append(var_a)
            tmp_k.append(var_k)
        a_var[i][j] = tmp_a
        k_var[i][j] = tmp_k

len(all_log_k)

percentile = 0.8 * 100
# pen_map = {-7: 20, -6: 10, -5: 1, -4: 0.6, -3: 0.3, -2: 0.1, -1: 0.05}
pen_map = {-11:80, -10: 80, -9: 60, -8: 30, -7: 20, -6: 10, -5: 5, -4: 2, -3: 1, -2: 0.5, -1: 0.2, 0: 0.1}
ideal_cp = [1, 0.5, 0.1, 1, 1, 20, 20, 1, 20, 1, 1, 1, 5]
# print('exp\tidea\tpercen_a\t\tpen_a\tpercen_k\t\tpen_k')
min_percen = 10
offset = -1
for exp in exp_range:
    percen_a = np.percentile(np.array(sum(a_var[exp], [])), percentile)
    percen_k = np.percentile(np.array(sum(k_var[exp], [])), percentile)
    # print(exp, percen_a, percen_k)
    # min_percen = min(min_percen, round(np.log(percen_a)), round(np.log(percen_k)))
    # print(exp, ideal_cp[exp], percen_a, round(np.log(percen_a)), pen_map[round(np.log(percen_a))+offset], percen_k, round(np.log(percen_k)), pen_map[round(np.log(percen_k))+offset], sep='\t')
# print(min_percen)
    

pen_list = []
offset
for exp in exp_range:
    percen_a = np.percentile(np.array(sum(a_var[exp], [])), percentile)
    percen_k = np.percentile(np.array(sum(k_var[exp], [])), percentile)
    pen_list.append((pen_map[round(np.log(percen_a))+offset], pen_map[round(np.log(percen_k))+offset]))
print(pen_list)

pen_list[2] = (1, 1)
pen_list[3] = (1, 1)
pen_list[5] = (1, 1)
pen_list[6] = (2, 2)
pen_list[8] = (1, 1)
pen_list[9] = (0.5, 0.5)

cnt = [0, 0, 0, 0]
cp_num = []

filep_index = None

print(pen_list)
for exp in exp_range:
    pen_a = pen_list[exp][0]
    pen_k = pen_list[exp][1]
    cp_tmp = []
    for fov in fov_range:
        # filep_index = '/data3/fxc/AnDiChallenge/dataset/public_data_challenge_v0/track_1/exp_{}/convert_trajs_fov_index_{}.csv'.format(exp, fov)
        # with open(filep_index, 'r') as fp:
        #     data2 = csv.reader(fp)
        #     index_vip = []
        #     for i in data2:
        #         index_vip.append(int(i[0]))

        results = []
        print("Exp {} Fov {}".format(exp, fov))
        pred = preds_all[exp][fov]
        # print(pred.shape, pad_all[exp][fov].shape)
        for idx in range(pred.shape[0]):
            # print(idx, end='\t')
            count_true = np.sum(pad_all[exp][fov][idx])
            pre_a = pred[idx, :count_true, 0].tolist()
            pre_k = pred[idx, :count_true, 1].tolist()
            # if preds_all_stage2 is not None and exp in [11]:
            #     pre_state = np.argmax(preds_all_stage2[exp][fov][idx, :count_true, 4:], axis=-1).tolist()
            # if preds_all_stage2_c is not None and exp in [8,9,10]:
            #     pre_state = np.argmax(preds_all_stage2_c[exp][fov][idx, :count_true, 4:], axis=-1).tolist()
            model_a = rpt.KernelCPD(kernel="linear", min_size=3, jump=15).fit(np.array(pre_a))
            # 检测变点
            # breakpoints = model_a.predict(pen=30)
            breakpoints_a = model_a.predict(pen=pen_a)
            model_k = rpt.KernelCPD(kernel="linear", min_size=3, jump=15).fit(np.array(pre_k))
            breakpoints_k = model_k.predict(pen=pen_k)
            # breakpoints = sorted(list(set(breakpoints_a).union(set(breakpoints_k))))
            breakpoints = merge_changepoints(breakpoints_a, breakpoints_k)
            print(breakpoints)
            cp_tmp.append(len(breakpoints)-1)

            segments_a = [pre_a[i:j] for i, j in zip([0] + breakpoints, breakpoints + [None])]
            segments_k = [pre_k[i:j] for i, j in zip([0] + breakpoints, breakpoints + [None])]
            # if (preds_all_stage2 is not None and exp in [11]) or (preds_all_stage2_c is not None and exp in [8,9,10]):
            #     segments_state = [pre_state[i:j] for i, j in zip([0] + breakpoints, breakpoints + [None])]

            tmp = []
            tmp.append(idx)

            for j in range(len(breakpoints)):
                aver_k=sum(segments_k[j]) / len(segments_k[j])
                aver_k=10 ** aver_k
                # aver_k=np.exp(aver_k).item()
                # tmp.append(aver_k)
                aver_a = sum(segments_a[j]) / len(segments_a[j])
                # tmp.append(aver_a)
                aver_state=2
                # if exp in [11] and preds_all_stage2 is not None:
                #     aver_state = round(sum(segments_state[j]) / len(segments_state[j]))
                #     if aver_state == 1:
                #         aver_state = 2
                #     if aver_state == 3 and aver_a <= 1:
                #         aver_state = 2
                #     if aver_state == 3 and aver_a < 1.9 and aver_a > 1:
                #         aver_a = 1.9
                #     if aver_a > 1.5:
                #         aver_state = 3
                #         aver_a = 1.9
                # if exp in [8,9,10] and preds_all_stage2_c is not None:
                #     aver_state = round(sum(segments_state[j]) / len(segments_state[j]))
                #     if aver_state == 3:
                #         aver_state = 2
                #     if aver_state == 1:
                #         if (aver_a > 1) or (exp in [9] and (aver_a > 0.5 or aver_k > 0.5)):
                #             aver_state = 2
                # if exp in [2, 3]:
                #     if aver_k < 0.02 and aver_a < 0.05:
                #         aver_state = 0
                # else: 
                if exp == 11:
                    if aver_a > 1:
                        aver_state = 3
                        aver_a = 1.9
                if aver_a < 2e-3: # 5e-3
                    # print('alpha={} -> 0, state={} -> 0'.format(aver_a, aver_state))
                    aver_a = 0
                    aver_state = 0
                elif aver_a < 0.02: # 0.05
                    # print('alpha={} < 0.05, state={} -> 1'.format(aver_a, aver_state))
                    # aver_a = 0
                    aver_state = 1
                if aver_a > 1.88:
                    # print('alpha={} > 1.88, state={} -> 3'.format(aver_a, aver_state))
                    aver_state = 3
                    if aver_a > 1.99:
                        aver_a = 1.99
                if aver_k < 1e-11:
                    # print('K={} -> 0, state={} -> 0, alpha={}'.format(aver_k, aver_state, aver_a))
                    aver_k = 0
                    aver_state = 0
                cnt[aver_state] += 1
                tmp.append(aver_k)
                tmp.append(aver_a)
                tmp.append(aver_state)
                tmp.append(breakpoints[j])

            results.append(tmp)
        
        save_dir = '../../challenge_results/track_2/exp_{}'.format(exp)
        # save_dir = '../../results/daR_new/track_2_test_0707/exp_{}'.format(exp)
        os.makedirs(save_dir, exist_ok=True)

        file = open(save_dir + '/fov_{}.txt'.format(fov), 'w')

        # 定义一个数组

        # 将数组逐行写入txt文件，用逗号隔开
        for item1 in results:
            for i,item2 in enumerate(item1):
                if i != len(item1)-1:
                    file.write(str(item2) + ',')
                else:
                    file.write(str(item2))
            file.write('\n')
        # 关闭文件
        file.close()
    cp_num.append(cp_tmp)
