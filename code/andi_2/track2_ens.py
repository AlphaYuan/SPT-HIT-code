import numpy as np
import csv
import os
import ensemble_task


preds_all = np.load('../../challenge_results/daR/track_2_all_preds_all.npy', allow_pickle=True)
pad_all = np.load('../../challenge_results/daR/track_2_all_pad_all.npy', allow_pickle=True)
# preds_all = np.load('../../challenge_results/0710/oyX/track_1_all_preds_all.npy', allow_pickle=True)
# pad_all = np.load('../../challenge_results/0710/oyX/track_1_all_pad_all.npy', allow_pickle=True)
root = '../../challenge_results/track_2'
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
    with open(file_path, 'w') as f:
        f.write('model: multi_state; num_state: {}\n'.format(len(ensemble_results)))
        writer = csv.writer(f, delimiter=';')
        # writer.writerows(ensemble_results)
        writer.writerows(list(map(list, zip(*ensemble_results)))) # transpose]
    # os.makedirs(os.path.join(root, 'ensemble', 'exp_{}'.format(exp)), exist_ok=True)
    # file_path = os.path.join(root, 'ensemble', 'exp_{}'.format(exp), 'ensemble_labels.txt')
    # with open(file_path, 'w') as f:
    #     f.write('model: multi_state; num_state: {}\n'.format(len(ensemble_results)))
    #     writer = csv.writer(f, delimiter=';')
    #     # writer.writerows(ensemble_results)
    #     writer.writerows(list(map(list, zip(*ensemble_results)))) # transpose
