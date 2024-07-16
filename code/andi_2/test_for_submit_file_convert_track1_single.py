import csv

import logging
import numpy as np
import os

import torch
from test_for_submit_track1_single import test_save

def read_data(exp_path,csv_path):
    with open(exp_path+csv_path, 'r') as fp:  #
        data = list(csv.reader(fp))  #
        data_set1 = []
        for i in range(len(data)):
            t = []
            del data[i][-1]    #删除最后一个空格字符串
            for j in range(len(data[i])):
                t.append(float(data[i][j]))
            data_set1.append(t)

    train_set=[]
    vip_index=[]

    for i in range(len(data_set1)):
        tmp=data_set1[i]
        vip_index.append(int(tmp[-1]))
        del tmp[-1]
        track_len=int(len(tmp)/2)

        desired_length = 20
        if track_len < desired_length:
            tmpx = tmp[:track_len]
            lastx= [tmpx[-1]]
            tmpx += lastx * (desired_length - track_len)
            tmpy = tmp[track_len:]
            lasty = [tmpy[-1]]
            tmpy += lasty * (desired_length - track_len)
            tmp = tmpx + tmpy
            track_len =int(len(tmp)/2)

        '''if track_len == 3:
            tmpx=tmp[:3]
            tmpx+=[tmpx[-1]]
            tmpy = tmp[3:]
            tmpy += [tmpy[-1]]
            tmp=tmpx+tmpy
            track_len+=1
        if track_len == 2:
            tmpx=tmp[:2]
            tmpx+=[tmpx[-1]]
            tmpx += [tmpx[0]]
            tmpy = tmp[2:]
            tmpy += [tmpy[-1]]
            tmpy += [tmpy[0]]
            tmp=tmpx+tmpy
            track_len+=2'''

        tmp += [0] * track_len
        tmp += [10] * track_len
        tmp += [0] * track_len
        train_set.append(tmp)

    return train_set,vip_index




logger = logging.getLogger('__main__')

NEG_METRICS = {'loss'}  # metrics for which "better" is less

val_times = {"total_time": 0, "count": 0}


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


# task = 'all'
task = 'vip'


num_exp=12
num_fov=30
for i in range (num_exp):
    for j in range (num_fov):
        # VIP
        if task == 'vip':
            public_data_path = '../../dataset/track_1/exp_{}/'.format(i)  # make sure the folder has this name or change it
            csv_data_path = 'vip_tracks_fov_{}.csv'.format(j)
            test_set,vip_index=read_data(exp_path=public_data_path,csv_path=csv_data_path)
            convert_csv_data_path = 'convert_trajs_fov_{}.csv'.format(j)
            convert_index_csv_data_path = 'convert_trajs_fov_index_{}.csv'.format(j)

            with open(os.path.join(public_data_path,convert_csv_data_path), 'w',
                    encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                ttt = test_set
                writer.writerows(ttt)
            with open(os.path.join(public_data_path,convert_index_csv_data_path), 'w', encoding='utf-8',newline='') as f:
                writer = csv.writer(f)
                for item in vip_index:
                    writer.writerow([item])

        # # ALL
        if task == 'all':
            public_data_path = '../../dataset/track_1/exp_{}/'.format(i)  # make sure the folder has this name or change it
            csv_data_path = 'all_tracks_fov_{}.csv'.format(j)
            test_set,vip_index=read_data(exp_path=public_data_path,csv_path=csv_data_path)
            convert_csv_data_path = 'convert_all_trajs_fov_{}.csv'.format(j)
            convert_index_csv_data_path = 'convert_all_trajs_fov_index_{}.csv'.format(j)

            with open(os.path.join(public_data_path,convert_csv_data_path), 'w',
                    encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                ttt = test_set
                writer.writerows(ttt)
            with open(os.path.join(public_data_path,convert_index_csv_data_path), 'w', encoding='utf-8',newline='') as f:
                writer = csv.writer(f)
                for item in vip_index:
                    writer.writerow([item])


        device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
        print(f"[Info]: Use {device} now!")
        addre = 'output/_2024-05-28_23-32-33_iSV/'
        addre = 'output/_2024-05-29_01-27-50_QQJ/'
        addre = 'output/_2024-05-30_01-04-12_H8L/'  # transmodel2 hiddenlayer=8
        addre = 'output/_2024-05-30_21-53-33_Oj2/'  # transmodel2 hiddenlayer=8
        addre = 'output/_2024-05-31_12-45-56_9Hp/'  # transmodel2 hiddenlayer=8

        addre = 'output/_2024-06-21_23-59-46_ptQ/'
        addre = '/data1/jiangy/andi_tcu/code/andi_2/output/_2024-06-25_10-56-24_VF6/'
        addre = 'output/_2024-06-25_16-11-45_7IT/'
        addre = '/data1/jiangy/andi_tcu/code/andi_2/output/_2024-06-24_19-57-59_R07/'
        addre = '/data1/jiangy/andi_tcu/code/andi_2/output/_2024-06-28_12-56-39_daR/'
        
        # addre = '/data1/jiangy/andi_tcu/code/andi_2/challenge_output/_2024-07-09_15-09-11_oyX/'

        public_data_path = '../../dataset/track_1/exp_{}/'.format(
            i)  # make sure the folder has this name or change it
        # VIP
        if task == 'vip':
            # addre = '/data1/jiangy/andi_tcu/code/andi_2/output/_2024-06-28_12-56-39_daR/'
            addre = 'output/_2024-06-20_11-36-43_7av/'
            convert_csv_data_path = 'convert_trajs_fov_{}.csv'.format(j)
            convert_index_csv_data_path = 'convert_trajs_fov_index_{}.csv'.format(j)
            save_dir='../../challenge_results/{}/track_1_vip/exp_{}'.format(addre[-4:-1], i)

        # ALL
        if task == 'all':
            addre = 'output/_2024-06-28_12-56-39_daR/'
            convert_csv_data_path = 'convert_all_trajs_fov_{}.csv'.format(j)
            convert_index_csv_data_path = 'convert_all_trajs_fov_index_{}.csv'.format(j)
            save_dir='../../challenge_results/{}/track_1_all/exp_{}'.format(addre[-4:-1], i)

        print(f"Exp {i}, Fov {j}")
        preds, pad = test_save(device, addre, public_data_path+convert_csv_data_path, save_dir, public_data_path+convert_index_csv_data_path , fov=j)
        preds_all[i][j] = preds
        pads_all[i][j] = pad
    
save_predsall = np.asarray(preds_all, dtype=object)
save_padsall = np.asarray(pads_all, dtype=object)
if task == 'all':
    np.save('../../challenge_results/{}/track_1_all_preds_all.npy'.format(addre[-4:-1]), save_predsall)
    np.save('../../challenge_results/{}/track_1_all_pad_all.npy'.format(addre[-4:-1]), save_padsall)

if task == 'vip':
    np.save('../../challenge_results/{}/track_1_vip_preds_all.npy'.format(addre[-4:-1]), save_predsall)
    np.save('../../challenge_results/{}/track_1_vip_pad_all.npy'.format(addre[-4:-1]), save_padsall)
a=0
