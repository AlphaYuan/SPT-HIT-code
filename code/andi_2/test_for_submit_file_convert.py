import csv

import logging

import os
import pickle
import numpy as np

import torch
# from test_for_submit import test_save
import test_for_submit
from multimodel_test_for_submit import test_save, load

def read_data(exp_path,csv_path):
    with open(exp_path+csv_path, 'r') as fp:  #
        data = list(csv.reader(fp))  #
        data_set1 = []
        for i in range(1,len(data),1):
            t = []
            for j in range(len(data[i])):
                t.append(float(data[i][j]))
            data_set1.append(t)

    train_set=[]
    id=0
    tmp_x = []
    tmp_y = []
    tmp_a = []
    tmp_k = []
    tmp_s = []
    for i in range(len(data_set1)):
        if data_set1[i][0] != id:
            traj_i = tmp_x + tmp_y + tmp_a + tmp_k + tmp_s
            train_set.append(traj_i)
            tmp_x = []
            tmp_y = []
            tmp_a = []
            tmp_k = []
            tmp_s = []
            id += 1
            tmp_x.append(data_set1[i][2])
            tmp_y.append(data_set1[i][3])
            tmp_a.append(0)
            tmp_k.append(10)
            tmp_s.append(0)
        else:
            tmp_x.append(data_set1[i][2])
            tmp_y.append(data_set1[i][3])
            tmp_a.append(0)
            tmp_k.append(10)
            tmp_s.append(0)
    traj_i = tmp_x + tmp_y + tmp_a + tmp_k + tmp_s
    train_set.append(traj_i)

    return train_set




logger = logging.getLogger('__main__')

NEG_METRICS = {'loss'}  # metrics for which "better" is less

val_times = {"total_time": 0, "count": 0}




# with open('/data4/jiangy/AnDiChallenge/andi_solver/outputs_new/test_andi_validation/track_2_new_ConvLSTM_enc_dec_16_32CP_all_rmse0.8.pkl', 'rb') as f:
#     CP_all_load = pickle.load(f)
num_exp=12
exp_range = range (num_exp)
# exp_range = [0, 1, 2, 3, 4, 7, 9, 10, 11] # 11没必要
# exp_range = [0, 1, 2, 3, 4, 7, 9, 10] # 3 7说不定可以再增加一些
# exp_range = [11]
# exp_range = [8]
num_fov=30
fov_range = range (num_fov)

preds_all = [[[] for j in fov_range] for i in exp_range]
pads_all = [[[] for j in fov_range] for i in exp_range]

for i in exp_range:
    for j in fov_range:
        public_data_path = '../../dataset/public_data_challenge_v0/track_2/exp_{}/'.format(i)  # make sure the folder has this name or change it
        csv_data_path = 'trajs_fov_{}.csv'.format(j)
        test_set=read_data(exp_path=public_data_path,csv_path=csv_data_path)
        convert_csv_data_path = 'convert_trajs_fov_{}.csv'.format(j)
        with open(os.path.join(public_data_path,convert_csv_data_path), 'w',
                  encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            ttt = test_set
            writer.writerows(ttt)
        # print(CP_all_load[i][j])

        device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
        print(f"[Info]: Use {device} now!")
        # addre = 'output/_2024-05-28_23-32-33_iSV/'
        # addre = 'output/_2024-05-29_01-27-50_QQJ/'
        # addre = 'output/_2024-05-30_01-04-12_H8L/'  # transmodel2 hiddenlayer=8
        # addre = 'output/_2024-05-30_21-53-33_Oj2/'  # transmodel2 hiddenlayer=8
        # addre = 'output/_2024-05-31_12-45-56_9Hp/'  # transmodel2 hiddenlayer=8

        # addre = 'output/_2024-06-18_00-13-50_Xkl/'  # model_0508 GatedLSTM
        # addre = 'output/_2024-06-18_20-55-47_Lm5/'  # transmodel2
        # addre = 'output/_2024-06-19_14-48-36_cyk/'  # GatedLSTM 0602
        # addre = '/data1/jiangy/andi_tcu/code/andi_2/output/_2024-06-20_09-16-54_21G/' # GatedLSTM no motion
        # addre = 'output/_2024-06-20_16-42-04_7E4/'  # Fusion pred_state
        # addre = 'output/_2024-06-20_17-12-36_Rvn/'  # Fusion no_state
        # addre = 'output/_2024-06-20_23-59-05_M8w/'  # Fusion mae_a

        # addre = 'output/_2024-06-21_12-38-58_B6D/'  # Fusion all_model
        # addre = 'output/_2024-06-21_19-15-21_NQG/'  # Fusion allmodel
        # addre = 'output/_2024-06-21_23-59-46_ptQ/'
        # # addre = 'output/_2024-06-21_20-04-46_c9P/'  # Fusion confinement
        # addre = 'output/_2024-06-22_09-10-58_UYK/'   # Fusion allmodel new
        # addre = 'output/_2024-06-22_22-15-08_8wO/'  # Fusion allpart
        # addre_state = 'output/_2024-06-22_22-14-30_emK/'    # Fusion allpart state

        # # addre = '/data1/jiangy/andi_tcu/code/andi_2/output/_2024-06-23_12-47-07_I4B/'
        # # addre = '/data1/jiangy/andi_tcu/code/andi_2/output/_2024-06-23_23-20-53_huj/'
        # addre = '/data1/jiangy/andi_tcu/code/andi_2/output/_2024-s-25_6rl/'
        # addre = '/data1/jiangy/andi_tcu/code/andi_2/output/_2024-06-23_23-20-53_huj/'
        # addre = '/data1/jiangy/andi_tcu/code/andi_2/output/_2024-06-24_19-53-51_c3G/'
        # addre = '/data1/jiangy/andi_tcu/code/andi_2/output/_2024-06-24_19-57-59_R07/'

        # addre = '/data1/jiangy/andi_tcu/code/andi_2/output/_2024-06-27_23-09-42_4BJ/' # Fusion 0626

        addre = '/data1/jiangy/andi_tcu/code/andi_2/output/_2024-06-28_12-56-39_daR/'

        # addre = '/data1/jiangy/andi_tcu/code/andi_2/challenge_output/_2024-07-09_15-09-11_oyX/'
        # addre = '/data1/jiangy/andi_tcu/code/andi_2/output/_2024-06-27_23-08-53_wBl/'
        # addre = '/data1/jiangy/andi_tcu/code/andi_2/output/_2024-06-28_10-20-50_5vV/'
        # addre = '/data1/jiangy/andi_tcu/code/andi_2/output/_2024-06-27_16-06-02_INf/'
        # addre = '/data1/jiangy/andi_tcu/code/andi_2/output/_2024-06-27_16-02-02_COK/'
        # addre = '/data1/jiangy/andi_tcu/code/andi_2/output/_2024-06-27_17-24-20_g6M/'
        # addre = '/data1/jiangy/andi_tcu/code/andi_2/output/_2024-06-25_10-56-34_Qrv/'
        # # addre = '/data1/jiangy/andi_tcu/code/andi_2/output/_2024-06-25_10-56-24_VF6/'
        # addre = '/data1/jiangy/andi_tcu/code/andi_2/output/_2024-06-25_16-14-17_6Gb/'
        # addre = 'output/_2024-06-25_16-11-45_7IT/'
        # addre = '/data1/jiangy/andi_tcu/code/andi_2/output/_2024-06-25_23-11-19_1Vj/'
        # addre = '/data1/jiangy/andi_tcu/code/andi_2/output/_2024-06-25_23-22-22_7LA/'
        # addre = '/data1/jiangy/andi_tcu/code/andi_2/output/_2024-06-26_08-16-51_i3L/'
        addre_state = None
        addre_cp = None#'/data1/jiangy/andi_tcu/code/andi_2/output/_2024-06-27_20-21-34_qwO/'
        
        public_data_path = '../../dataset/public_data_challenge_v0/track_2/exp_{}/'.format(i)  # make sure the folder has this name or change it
        convert_csv_data_path = 'convert_trajs_fov_{}.csv'.format(j)
        print("Exp {} Fov {}".format(i, j))
        save_dir='../../challenge_results/{}/track_2/exp_{}'.format(addre[-4:-1], i)
        preds, pad = test_for_submit.test_save(device, addre, public_data_path+convert_csv_data_path, save_dir, exp=i, fov=j, addre_state=addre_state, addre_cp=addre_cp)
        preds_all[i][j] = preds
        pads_all[i][j] = pad
    
save_predsall = np.asarray(preds_all, dtype=object)
save_padsall = np.asarray(pads_all, dtype=object)
np.save('../../challenge_results/{}/track_2_all_preds_all.npy'.format(addre[-4:-1]), save_predsall)
np.save('../../challenge_results/{}/track_2_all_pad_all.npy'.format(addre[-4:-1]), save_padsall)

        # 分长度预测
#         addre = ['/data1/jiangy/andi_tcu/code/andi_2/output/_2024-06-26_12-41-21_WWU/', '/data1/jiangy/andi_tcu/code/andi_2/output/_2024-06-26_12-42-50_Hwi/', '/data1/jiangy/andi_tcu/code/andi_2/output/_2024-06-26_12-46-07_Osu/']
#         # addre = ['/data1/jiangy/andi_tcu/code/andi_2/output/_2024-06-26_16-54-40_0Kr/', '/data1/jiangy/andi_tcu/code/andi_2/output/_2024-06-26_23-45-55_OOJ/', '/data1/jiangy/andi_tcu/code/andi_2/output/_2024-06-26_20-44-24_5OC/'] # loss mse
#         # addre = ['/data1/jiangy/andi_tcu/code/andi_2/output/_2024-06-26_23-47-14_rAV/', '/data1/jiangy/andi_tcu/code/andi_2/output/_2024-06-27_06-48-19_XfT/', '/data1/jiangy/andi_tcu/code/andi_2/output/_2024-06-27_03-30-36_ekC/'] # Loss L1_alpha
#         model_50, config_50, data_50 = load(addre[0], public_data_path+convert_csv_data_path, device)
#         model_100, config_100, data_100 = load(addre[1], public_data_path+convert_csv_data_path, device)
#         model_200, config_200, data_200 = load(addre[2], public_data_path+convert_csv_data_path, device)
#         models = [model_50, model_100, model_200]
#         configs = [config_50, config_100, config_200]
#         datas = [data_50, data_100, data_200]
    #     break
    # break
# for i in range (num_exp):
#     for j in range (num_fov):
#         print("Exp {} Fov {}".format(i, j))
#         public_data_path = '/data4/jiangy/AnDiChallenge/dataset/public_data_validation_v1/track_2/exp_{}/'.format(
#             i)  # make sure the folder has this name or change it
#         convert_csv_data_path = 'convert_trajs_fov_{}.csv'.format(j)
#         save_dir='../../results/{}/track_2/exp_{}'.format(addre[-1][-4:-1], i)
#         test_save(models, configs, datas, device, addre, public_data_path+convert_csv_data_path, save_dir,  fov=j, addre_state=addre_state)



a=0
