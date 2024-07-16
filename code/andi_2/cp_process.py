# %%
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

# %%
# preds_all = np.load('../../challenge_results/0710/oyX/track_1_vip_preds_all.npy', allow_pickle=True)
# pad_all = np.load('../../challenge_results/0710/oyX/track_1_vip_pad_all.npy', allow_pickle=True)
# preds_all = np.load('../../challenge_results/0705/7av/track_1_vip_preds_all.npy', allow_pickle=True)
# pad_all = np.load('../../challenge_results/0705/7av/track_1_vip_pad_all.npy', allow_pickle=True)
# preds_all = np.load('../../challenge_results/0701/daR_0701/track1_all_preds_all.npy', allow_pickle=True)
# pad_all = np.load('../../challenge_results/0701/daR_0701/track1_all_pad_all.npy', allow_pickle=True)
# preds_all = np.load('../../challenge_results/0701/daR_0701/track1_vip_preds_all.npy', allow_pickle=True)
# pad_all = np.load('../../challenge_results/0701/daR_0701/track1_vip_pad_all.npy', allow_pickle=True)

# preds_all = np.load('../../challenge_results/0705/daR/track_2_all_preds_all.npy', allow_pickle=True)
# pad_all = np.load('../../challenge_results/0705/daR/track_2_all_pad_all.npy', allow_pickle=True)

preds_all_stage2 = None
preds_all_stage2_c = None
# preds_all_stage2 = np.load('../../challenge_results/0711/nSQ/track_1_all_preds_all.npy', allow_pickle=True)
# preds_all_stage2_c = np.load('../../challenge_results/0712/URg/track_1_all_preds_all.npy', allow_pickle=True)

# %%

num_exp=12
exp_range = range (num_exp)
num_fov=30
fov_range = range (num_fov)

def get_var(preds_all, pad_all):
    # a_mean = [[[] for j in fov_range] for i in exp_range]
    a_var = [[[] for j in fov_range] for i in exp_range]
    # k_mean = [[[] for j in fov_range] for i in exp_range]
    k_var = [[[] for j in fov_range] for i in exp_range]
    # cp_num = [[[] for j in fov_range] for i in exp_range]
    # cps = [[[] for j in fov_range] for i in exp_range]
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
                # print(count_true)
                all_len.append(count_true)
                all_log_k += pred[idx, :count_true, 1].tolist()
                var_a = np.var(pred[idx, :count_true, 0])
                var_k = np.var(pred[idx, :count_true, 1])
                tmp_a.append(var_a)
                tmp_k.append(var_k)
                # model2 = rpt.KernelCPD(kernel="linear", min_size=3, jump=15).fit(np.array(pred[idx, :count_true, 0]))
                # # 检测变点
                # breakpoints_a = model2.predict(pen=10)
                # model_k = rpt.KernelCPD(kernel="linear", min_size=3, jump=15).fit(np.array(pred[idx, :count_true, 1]))
                # breakpoints_k = model_k.predict(pen=10)
                # # breakpoints = sorted(list(set(breakpoints_a).union(set(breakpoints_k))))
                # breakpoints = merge_changepoints(breakpoints_a, breakpoints_k)
                # tmp_cp.append(breakpoints)
                # tmp_cp_cnt.append(len(breakpoints))
            a_var[i][j] = tmp_a
            k_var[i][j] = tmp_k
            # cps[i][j] = tmp_cp
            # cp_num[i][j] = tmp_cp_cnt
    return a_var, k_var, all_log_k

# %%
preds_all = np.load('../../challenge_results/0705/7av/track_1_vip_preds_all.npy', allow_pickle=True)
pad_all = np.load('../../challenge_results/0705/7av/track_1_vip_pad_all.npy', allow_pickle=True)
# preds_all_t1_vip = deepcopy(preds_all)
# pad_all_t1_vip = deepcopy(pad_all)
# a_var_t1_vip, k_var_t1_vip, all_log_k_t1_vip = get_var(preds_all_t1_vip, pad_all_t1_vip)

# preds_all = np.load('../../challenge_results/0701/daR_0701/track1_all_preds_all.npy', allow_pickle=True)
# pad_all = np.load('../../challenge_results/0701/daR_0701/track1_all_pad_all.npy', allow_pickle=True)
# preds_all_t1_all = deepcopy(preds_all)
# pad_all_t1_all = deepcopy(pad_all)
a_var, k_var, all_log_k = get_var(preds_all, pad_all)

preds_all_t2_all = np.load('../../challenge_results/0705/daR/track_2_all_preds_all.npy', allow_pickle=True)
pad_all_t2_all = np.load('../../challenge_results/0705/daR/track_2_all_pad_all.npy', allow_pickle=True)
# preds_all_t2_all = deepcopy(preds_all)
# pad_all_t2_all = deepcopy(pad_all)
a_var_t2_all, k_var_t2_all, all_log_k_t2_all = get_var(preds_all_t2_all, pad_all_t2_all)


import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def histogram_matching(source, template):
    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()
    
    # 获取源数据和模板数据的直方图
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True, return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)
    
    # 计算累积分布函数
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]
    
    # 使用插值方法将源数据映射到目标数据
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
    
    return interp_t_values[bin_idx].reshape(oldshape)

# 示例数据
# N, T = 100, 100
source = np.array(all_log_k) #np.random.normal(loc=0, scale=1, size=(N, T, 2))
template = np.array(all_log_k_t2_all) #np.random.normal(loc=5, scale=2, size=(N, T, 2))

# 将源数据的直方图匹配到模板数据
all_log_k_matched = histogram_matching(source, template)


# %%
preds_all = deepcopy(preds_all)
pad_all = deepcopy(pad_all)
# all_log_k = deepcopy(all_logk_t1_vip)
all_log_k = deepcopy(all_log_k_matched)
tmp = []
cnt = 0
for i in exp_range:
    for j in fov_range:
        pred = preds_all[i][j]
        for idx in range(pred.shape[0]):
            count_true = np.sum(pad_all[i][j][idx])
            print(count_true)
            pred[idx, :count_true, 1] = np.array(all_log_k[cnt:cnt+count_true]).reshape(-1, )
            cnt += count_true
            tmp += pred[idx, :count_true, 1].tolist()

# a_var_t1_vip, k_var_t1_vip, all_log_k_t1_vip = get_var(preds_all_t1_vip, pad_all_t1_vip)

# %%
a_var = deepcopy(a_var_t2_all)
k_var = deepcopy(k_var_t2_all)
# a_var, k_var, _ = get_var(preds_all, pad_all)
percentile = 0.8 * 100
# pen_map = {-7: 20, -6: 10, -5: 1, -4: 0.6, -3: 0.3, -2: 0.1, -1: 0.05}
pen_map = {-11:80, -10: 80, -9: 60, -8: 30, -7: 20, -6: 10, -5: 5, -4: 2, -3: 1, -2: 0.5, -1: 0.2, 0: 0.1}
ideal_cp = [1, 0.5, 0.1, 1, 1, 20, 20, 1, 20, 1, 1, 1, 5]
print('exp\tidea\tpercen_a\t\tpen_a\tpercen_k\t\tpen_k')
min_percen = 10
offset = -1
for exp in exp_range:
    percen_a = np.percentile(np.array(sum(a_var[exp], [])), percentile)
    percen_k = np.percentile(np.array(sum(k_var[exp], [])), percentile)
    # print(exp, percen_a, percen_k)
    # min_percen = min(min_percen, round(np.log(percen_a)), round(np.log(percen_k)))
    print(exp, ideal_cp[exp], percen_a, round(np.log(percen_a)), pen_map[round(np.log(percen_a))+offset], percen_k, round(np.log(percen_k)), pen_map[round(np.log(percen_k))+offset], sep='\t')
# print(min_percen)
    
import csv

pen_list = []
offset
for exp in exp_range:
    percen_a = np.percentile(np.array(sum(a_var[exp], [])), percentile)
    percen_k = np.percentile(np.array(sum(k_var[exp], [])), percentile)
    pen_list.append((pen_map[round(np.log(percen_a))+offset], pen_map[round(np.log(percen_k))+offset]))
print(pen_list)

# %%
from copy import deepcopy
pen_list_save = deepcopy(pen_list)

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
    pen_a = pen_list[exp][0] * 2
    pen_k = pen_list[exp][1] * 4
    cp_tmp = []
    for fov in fov_range:
        filep_index = '/data3/fxc/AnDiChallenge/dataset/public_data_challenge_v0/track_1/exp_{}/convert_trajs_fov_index_{}.csv'.format(exp, fov)
        with open(filep_index, 'r') as fp:
            data2 = csv.reader(fp)
            index_vip = []
            for i in data2:
                index_vip.append(int(i[0]))

        results = []
        # print("Exp {} Fov {}".format(exp, fov))
        pred = preds_all[exp][fov]
        # print(pred.shape, pad_all[exp][fov].shape)
        for idx in range(pred.shape[0]):
            # print(idx, end='\t')
            count_true = np.sum(pad_all[exp][fov][idx])
            pre_a = pred[idx, :count_true, 0].tolist()
            pre_k = pred[idx, :count_true, 1].tolist()
            model_a = rpt.KernelCPD(kernel="linear", min_size=3, jump=15).fit(np.array(pre_a))
            # 检测变点
            # breakpoints = model_a.predict(pen=30)
            breakpoints_a = model_a.predict(pen=pen_a)
            model_k = rpt.KernelCPD(kernel="linear", min_size=3, jump=15).fit(np.array(pre_k))
            breakpoints_k = model_k.predict(pen=pen_k)
            # breakpoints = sorted(list(set(breakpoints_a).union(set(breakpoints_k))))
            breakpoints = merge_changepoints(breakpoints_a, breakpoints_k)
            # print(breakpoints)
            cp_tmp.append(len(breakpoints)-1)

            segments_a = [pre_a[i:j] for i, j in zip([0] + breakpoints, breakpoints + [None])]
            segments_k = [pre_k[i:j] for i, j in zip([0] + breakpoints, breakpoints + [None])]

            tmp = []
            if filep_index:
                tmp.append(int(index_vip[idx]))
            else:
                tmp.append(idx)

            for j in range(len(breakpoints)):
                aver_k=sum(segments_k[j]) / len(segments_k[j])
                aver_k=10 ** aver_k
                # aver_k=np.exp(aver_k).item()
                # tmp.append(aver_k)
                aver_a = sum(segments_a[j]) / len(segments_a[j])
                # tmp.append(aver_a)
                aver_state=2
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
        
        save_dir = '../../challenge_results/0713/track_1_vip/exp_{}'.format(exp)
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


# %%
print(cnt)
for i, item in enumerate(cp_num):
    print(i, sum(item))
exit(0)
# %%
print(cnt)
for i, item in enumerate(cp_num):
    print(i, sum(item))

# %%
# cnt[0], cnt[1], cnt[2] - cnt[0] - cnt[1] - cnt[3], cnt[3]
print(cnt)
for i, item in enumerate(cp_num):
    print(i, sum(item))

# %%


# %%
preds_all = np.load('../../challenge_results/0705/daR/track_2_all_preds_all.npy', allow_pickle=True)
pad_all = np.load('../../challenge_results/0705/daR/track_2_all_pad_all.npy', allow_pickle=True)
# preds_all = np.load('../../challenge_results/0710/oyX/track_1_all_preds_all.npy', allow_pickle=True)
# pad_all = np.load('../../challenge_results/0710/oyX/track_1_all_pad_all.npy', allow_pickle=True)
root = '../../challenge_results/0711/daR_new/track_2'
# preds_all = np.load('../../challenge_results/0708/daR/track_1_all_preds_all.npy', allow_pickle=True)
# pad_all = np.load('../../challenge_results/0708/daR/track_1_all_pad_all.npy', allow_pickle=True)
# root = '../../challenge_results/0708/daR/track_1_all'
# K_map = {0: (1, 1), 1: (1, 1), 2: (2, 2), 3: (3, 4), 4: (2, 2), 5: (1, 3), 6: (1, 2), 7: (3, 2), 8: (2, 2), 9: (2, 2), 10: (3, 3), 11: (2, 2)} # 10: (2, 2) ? 
# preds_all = np.load('../../challenge_results/0706/daR/track_1_all_preds_all.npy', allow_pickle=True)
# pad_all = np.load('../../challenge_results/0706/daR/track_1_all_pad_all.npy', allow_pickle=True)
# root = '../../challenge_results/0707/daR_new/track_1_all'
# K_map = {0: (2, 2), 1: (2, 2), 2: (2, 2), 3: (4, 4), 4: (2, 2), 5: (3, 3), 6: (2, 2), 7: (2, 2), 8: (2, 2), 9: (2, 2), 10: (3, 3), 11: (2, 2)}
# preds_all = np.load('../../challenge_results/7av/track_1_all_preds_all.npy', allow_pickle=True)
# pad_all = np.load('../../challenge_results/7av/track_1_all_pad_all.npy', allow_pickle=True)
# root = '../../challenge_results/7av_new/track_1_all'

import csv
import imp
import ensemble_task
imp.reload(ensemble_task)



#### set param

K_max_a = 5
K_max_k = 4
progressive_explore = False

preds_all_new = preds_all
num_exp=12
exp_range = range(num_exp)
num_fov=30
fov_range = range(num_fov)
a_all = [[] for i in exp_range]
k_all = [[] for i in exp_range]
for i in exp_range:
    for j in fov_range:
        pred = preds_all[i][j]
        for idx in range(pred.shape[0]):
            count_true = np.sum(pad_all[i][j][idx])
            if False:
            # if i in [10, 11]:
                t = pred[idx, :count_true, 0]
                t[t > 1] = 1.9
                a_all[i] += t.tolist()
            else:
                a_all[i] += pred[idx, :count_true, 0].tolist()
            k_all[i] += (10 ** pred[idx, :count_true, 1]).tolist()
print(len(a_all[0]), len(k_all[0]))

def get_weight(assigned_mu, assigned_sigma):
    # weight = np.zeros((np.unique(assigned_mu).shape[0], 1))
    weight_map = {}
    time_sum = 0
    for i in range(assigned_mu.shape[0]):
        key = '{}_{}_{}_{}'.format(assigned_mu[i][0], assigned_mu[i][1], assigned_sigma[i][0], assigned_sigma[i][1],)
        if key not in weight_map:
            weight_map[key] = 0
        weight_map[key] += 1
        time_sum += 1
    print(weight_map)
    results = []
    for key, value in weight_map.items():
        keys = key.split('_')
        # print(keys)
        ratio = value / time_sum
        flag = False
        # if ratio < 0.05:
        #     for idx, item in enumerate(results):
        #         if keys[0] == str(item[0]):
        #             if int(np.log10(float(keys[1]))) == int(np.log10(float(keys[3]))):
        #                 print(results[idx][4])
        #                 results[idx][4] += ratio
        #                 flag = True
        #                 break
        #         if keys[1] == str(item[2]):
        #             if int(np.log10(float(keys[0]))) == int(np.log10(float(keys[2]))):
        #                 print(results[idx][4])
        #                 results[idx][4] += ratio
        #                 flag = True
        #                 break
        #     if flag:
        #         print("Merge: ", [float(keys[0]), float(keys[2]), float(keys[1]), float(keys[3]), ratio], 'to', results[idx])
        #     continue
        if not flag:
            results.append([float(keys[0]), float(keys[2]), float(keys[1]), float(keys[3]), ratio])
    return results

exp_range = range(num_exp)
for exp in exp_range:
    # K_max_a, K_max_k = K_map[exp]
    print('Processing: ', exp)
    # fov_res = preds_all_new[exp]
    alpha = np.array(a_all[exp]).reshape(-1, 1)
    K = np.array(k_all[exp]).reshape(-1, 1)
    assigned_mu_alpha, assigned_sigma_alpha = ensemble_task.gmm_fit(alpha, K_max=K_max_a, progressive_explore=progressive_explore)
    assigned_mu_K, assigned_sigma_K = ensemble_task.gmm_fit(K, K_max=K_max_k, progressive_explore=progressive_explore)
    # joint = np.concatenate((alpha, K), axis=-1)
    # assigned_mu, assigned_sigma = gmm_fit(joint, K_max=5)
    cat_mu = np.concatenate((assigned_mu_alpha.reshape(-1,1), assigned_mu_K.reshape(-1,1)), axis=-1)
    cat_sigma = np.concatenate((assigned_sigma_alpha.reshape(-1,1), assigned_sigma_K.reshape(-1,1)), axis=-1)
    ensemble_results = get_weight(cat_mu, cat_sigma)
    print("num: {}".format(len(ensemble_results)))
    file_path = os.path.join(root, 'exp_{}'.format(exp), 'ensemble_labels.txt')
    # with open(file_path, 'w') as f:
    #     f.write('model: multi_state; num_state: {}\n'.format(len(ensemble_results)))
    #     writer = csv.writer(f, delimiter=';')
    #     # writer.writerows(ensemble_results)
    #     writer.writerows(list(map(list, zip(*ensemble_results)))) # transpose]
    os.makedirs(os.path.join(root, 'ensemble', 'exp_{}'.format(exp)), exist_ok=True)
    file_path = os.path.join(root, 'ensemble', 'exp_{}'.format(exp), 'ensemble_labels.txt')
    with open(file_path, 'w') as f:
        f.write('model: multi_state; num_state: {}\n'.format(len(ensemble_results)))
        writer = csv.writer(f, delimiter=';')
        # writer.writerows(ensemble_results)
        writer.writerows(list(map(list, zip(*ensemble_results)))) # transpose


# %%
import numpy as np
from sklearn.mixture import BayesianGaussianMixture
import matplotlib.pyplot as plt

preds_all = np.load('../../challenge_results/0705/daR/track_2_all_preds_all.npy', allow_pickle=True)
pad_all = np.load('../../challenge_results/0705/daR/track_2_all_pad_all.npy', allow_pickle=True)
root = '../../challenge_results/0711/daR_new/track_2'

preds_all_new = preds_all
num_exp=12
exp_range = range(num_exp)
num_fov=30
fov_range = range(num_fov)
a_all = [[] for i in exp_range]
k_all = [[] for i in exp_range]
for i in exp_range:
    for j in fov_range:
        pred = preds_all[i][j]
        for idx in range(pred.shape[0]):
            count_true = np.sum(pad_all[i][j][idx])
            if False:
            # if i in [10, 11]:
                t = pred[idx, :count_true, 0]
                t[t > 1] = 1.9
                a_all[i] += t.tolist()
            else:
                a_all[i] += pred[idx, :count_true, 0].tolist()
            k_all[i] += (10 ** pred[idx, :count_true, 1]).tolist()
print(len(a_all[0]), len(k_all[0]))

def bgmm(a):
    # 使用贝叶斯高斯混合模型，设定较大的最大成分数量
    max_components = 8  # 可以根据需要调整
    bgmm_a = BayesianGaussianMixture(n_components=max_components, random_state=42)
    bgmm_a.fit(a.reshape(-1, 1))

    # 获取拟合的参数
    means_a = bgmm_a.means_.flatten()
    covariances_a = bgmm_a.covariances_.flatten()
    weights_a = bgmm_a.weights_

    # 过滤掉权重非常小的成分
    threshold = 1e-3
    valid_components_a = weights_a > threshold
    filtered_means_a = means_a[valid_components_a]
    filtered_covariances_a = covariances_a[valid_components_a]
    filtered_weights_a = weights_a[valid_components_a]

    print('means: ', filtered_means_a)
    print('vars: ', filtered_covariances_a)
    print('weights: ', filtered_weights_a)

    posterior_probs_a = bgmm_a.predict_proba(a.reshape(-1, 1))

    estimated_means_a = np.dot(posterior_probs_a, means_a)
    estimated_variances_a = np.dot(posterior_probs_a, covariances_a)
    return estimated_means_a, estimated_variances_a

exp_range = range(num_exp)
for exp in exp_range:
    # K_max_a, K_max_k = K_map[exp]
    print('Processing: ', exp)
    # fov_res = preds_all_new[exp]
    alpha = np.array(a_all[exp]).reshape(-1, 1)
    K = np.array(k_all[exp]).reshape(-1, 1)
    assigned_mu_alpha, assigned_sigma_alpha = bgmm(alpha)
    assigned_mu_K, assigned_sigma_K = bgmm(K)
    # joint = np.concatenate((alpha, K), axis=-1)
    # assigned_mu, assigned_sigma = gmm_fit(joint, K_max=5)
    cat_mu = np.concatenate((assigned_mu_alpha.reshape(-1,1), assigned_mu_K.reshape(-1,1)), axis=-1)
    cat_sigma = np.concatenate((assigned_sigma_alpha.reshape(-1,1), assigned_sigma_K.reshape(-1,1)), axis=-1)
    # ensemble_results = get_weight(cat_mu, cat_sigma)

# %%
1.40970578+np.sqrt(0.05582661)

# %%
10 ** np.array([-0.58182036, -1.82769999, -0.94636441]), 10 ** np.array([0.0225418,  0.0070492,  0.08187755])

# %%
cp_count = [[[] for j in fov_range] for i in exp_range]
a_np = []
k_np = []
cp_np = []
lab_exp = []
for i in exp_range:
    for j in fov_range:
        tmp = []
        for k in range(len(a_var[i][j])):
            tmp.append(len(cp_num[i][j][k]))
            # print(i, j, k, len(cp_num[i][j][k]), a_var[i][j][k], 10 ** k_var[i][j][k])
            a_np.append(a_var[i][j][k])
            k_np.append(k_var[i][j][k])
            cp_np.append(len(cp_num[i][j][k]))
            lab_exp.append(i)
        cp_count[i][j] = tmp
a_np = np.array(a_np)
k_np = np.array(k_np)
cp_np = np.array(cp_np)
lab_exp = np.array(lab_exp)

# %%
lab_exp

# %%
plt.close()
cmap = cm.get_cmap('jet', len(np.unique(lab_exp)))
fig = plt.figure()
plt.scatter(a_np[:], cp_np[:], c=lab_exp[:], cmap=cmap)
# plt.plot([1,2,3], [4,54,6])
plt.legend()
plt.show()

# %%
tmp_breakpoints = [6,7,11,14,15,38,39,40]

breakpoints = []
i = 1
pre = tmp_breakpoints[0]
while i < (len(tmp_breakpoints) - 1):
    tmp = pre
    print(i, tmp)
    if tmp_breakpoints[i + 1] - pre <= 3:
        print('merge3: ', pre, tmp_breakpoints[i:i+2])
        pre = round((pre + tmp_breakpoints[i] + tmp_breakpoints[i+1]) / 3)
        i = i + 1
    elif tmp_breakpoints[i] - pre <= 3:
        print('merge2: ', pre, tmp_breakpoints[i:i+1])
        pre = round((pre + tmp_breakpoints[i]) / 2)
        i = i + 1
    else:
        pre = tmp_breakpoints[i]
        i = i + 1
    breakpoints.append(pre)
print(breakpoints)

# %%
def merge_close_points_average(sorted_list):
    if not sorted_list:
        return []

    merged_points = []
    current_merge = [sorted_list[0]]

    for i in range(1, len(sorted_list)):
        if sorted_list[i] - current_merge[-1] < 3:
            current_merge.append(sorted_list[i])
        else:
            # Calculate average of current_merge
            average_value = round(sum(current_merge) / len(current_merge))
            merged_points.append(average_value)
            current_merge = [sorted_list[i]]

    # Append the average of the last group
    average_value = round(sum(current_merge) / len(current_merge))
    merged_points.append(average_value)

    return merged_points

merge_close_points_average([1])

# %%
t0 = 0
k1, a1, s1, t1, k2, a2, s2, t2 = 0.040797585480740306,0.7658367201685905,2,40,0.02535920410211726,0.08442279733717442,2,200
# k2, a2, s2, t2 = 0.029472049022845566,0.6737451150303795,2,114
print("{},{},{}".format((k1 * (t1 - t0) + k2 * (t2 - t1)) / (t2 - t0), (a1 * (t1 - t0) + a2 * (t2 - t1)) / (t2 - t0), t2))

# %%
import numpy as np

# 已知预测的样本数量
predicted_counts = {
    0: 3700,
    1: 1812,
    2: 61000,
    3: 5600
}
target_f1_score = 0.91234
target_jsc = 0.645

# Best match found: 0类=4800, 1类=2800, 2类=50000, 3类=7800
# Achieved F1 Score: 0.9133333333333333

# predicted_counts = {
#     0: 100,
#     1: 0,
#     2: 63000,
#     3: 7200
# }
# target_f1_score = 0.9134
# target_jsc = 0.645
# Best match found: 0类=4800, 1类=2800, 2类=50000, 3类=7800

predicted_counts = {
    0: 100,
    1: 340,
    2: 69000,
    3: 956
}
target_f1_score = 0.9058
target_jsc = 0.632
# 0类=4800, 1类=2800, 2类=53000, 3类=7800


# 定义搜索范围
search_range = {
    0: range(0, 20000, 1000),
    1: range(0, 20000, 1000),
    2: range(50000, 90000, 1000),
    3: range(2000, 50000, 1000)
}

def calculate_f1_score(TP, FP, FN):
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

best_match = None
best_f1_score = 0
best_distribution = None

# 搜索每种样本数量的组合
for a in search_range[0]:
    for b in search_range[1]:
        for c in search_range[2]:
            for d in search_range[3]:
                actual_counts = {0: a, 1: b, 2: c, 3: d}
                total_predicted = sum(predicted_counts.values())
                total_actual = sum(actual_counts.values())
                
                matched_points = int(target_jsc * (total_predicted + total_actual) / (1 + target_jsc))
                
                # 假设匹配点的比例与实际点的比例相同
                matched_2 = int(matched_points * (predicted_counts[2] / total_predicted))
                TP = min(matched_2, c)
                FP = predicted_counts[2] - TP
                FN = c - TP

                f1_score = calculate_f1_score(TP, FP, FN)
                if abs(f1_score - target_f1_score) < abs(best_f1_score - target_f1_score):
                    best_f1_score = f1_score
                    best_distribution = (a, b, c, d)

print(f"Best match found: 0类={best_distribution[0]}, 1类={best_distribution[1]}, 2类={best_distribution[2]}, 3类={best_distribution[3]}")
print(f"Achieved F1 Score: {best_f1_score}")




