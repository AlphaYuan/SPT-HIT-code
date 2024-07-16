import numpy as np
# import andi
import csv
import matplotlib as mpl
from copy import deepcopy
# ANDI = andi.andi_datasets()
from torch.nn import functional as F
from scipy.spatial.distance import mahalanobis
import torch
from utils1 import prepare_dataset
import utils1
from sklearn.metrics import roc_curve, auc,precision_recall_curve

# Project modules
from options import Options

# 3rd party packages
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Project modules
from options import Options
from torch import nn
import logging
import sys
# import real_llla
import matplotlib.pyplot as plt
import os
import traceback
import json
from datetime import datetime
import string
import random
from utils import utils, analysis
from collections import OrderedDict
import time
import pickle
from functools import partial
import ruptures as rpt  # 导入ruptures
import ipdb
import torch
from torch.utils.data import DataLoader
import numpy as np
import sklearn

# Project modules
from options import Options

from utils import utils

from running1 import setup, pipeline_factory, validate, check_progress, NEG_METRICS
from datasets.data import data_factory

from transmodel_fusion import model_factory
from losscompute import get_loss_module
from optimizers import get_optimizer
from utils import utils, analysis
import numpy as np
import pandas as pd
import seaborn as sns
from torch.autograd import Variable
import model_0508

logger = logging.getLogger('__main__')

NEG_METRICS = {'loss'}  # metrics for which "better" is less

val_times = {"total_time": 0, "count": 0}

plt.switch_backend('agg')

def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.style.use('ggplot')
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend(loc="upper right")
    plt.savefig(name,bbox_inches='tight')


def setup(args):
    """Prepare training session: read configuration from file (takes precedence), create directories.
    Input:
        args: arguments object from argparse
    Returns:
        config: configuration dictionary
    """

    config = args.__dict__  # configuration dictionary

    if args.config_filepath is not None:
        logger.info("Reading configuration ...")
        try:  # dictionary containing the entire configuration settings in a hierarchical fashion
            config.update(utils.load_config(args.config_filepath))
        except:
            logger.critical("Failed to load configuration file. Check JSON syntax and verify that files exist")
            traceback.print_exc()
            sys.exit(1)



    return config


class NoFussCrossEntropyLoss(torch.nn.CrossEntropyLoss):
    """
    pytorch's CrossEntropyLoss is fussy: 1) needs Long (int64) targets only, and 2) only 1D.
    This function satisfies these requirements
    """

    def forward(self, inp, target):
        return F.cross_entropy(inp, target.long().squeeze(), weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)


class RMSLELoss(nn.Module):
    def __init__(self, loss='MSE'):
        super().__init__()
        self.loss = nn.MSELoss()
        if loss == 'MAE':
            self.loss = nn.L1Loss()

    def forward(self, pred, actual):
        # return torch.sqrt(F.mse_loss(torch.log(pred + 1), torch.log(actual + 1)))
        # if torch.min(pred) < 0:
        #     return self.loss(torch.log(torch.clip(pred, 0) + 1), torch.log(actual + 1))
        return self.loss(torch.log(pred + 1), torch.log(actual + 1))

def get_criterion(loss='MAE'):
    if loss == 'MAE':
        return nn.L1Loss()
    elif loss == 'MSE':
        return nn.MSELoss()
    elif loss == 'MSLE':
        return RMSLELoss()
    elif loss == 'MALE':
        return RMSLELoss('MAE')
    

def merge_changepoints(cp_a, cp_k):
    set_cp_a = set(cp_a)
    set_cp_k = set(cp_k)
    merge = sorted(list(set_cp_a.union(set_cp_k)))
    for i in range(len(merge) - 1):
        if (merge[i + 1] - merge[i]) < 3:
            if merge[i] in set_cp_k:
                set_cp_k.remove(merge[i])
            elif merge[i + 1] in set_cp_k:
                set_cp_k.remove(merge[i + 1])
    return sorted(list(set_cp_a.union(set_cp_k)))

                



def test_save(device,addre,filep_valid_id,save_dir,exp,fov,addre_state=None,addre_cp=None,CP=None,eval_metrics=False):
    ################################################################################################################################
    myseed = 42069  # set a random seed for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(myseed)
    torch.manual_seed(myseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)


    with open(os.path.join(addre, 'configuration.json')) as fp:
        # json.dump(config, fp, indent=4, sort_keys=True)
        config = json.load(fp)

    tlen = config['tlen']
    tnum = config['tnum']
    # dim = 3
    dim = config['dimension']
    config['test_pattern'] = True
    config['load_model'] = addre + 'checkpoints/model_best_seg.pth'
    # config['load_model'] = addre + 'checkpoints/model_100_seg.pth'
    # config['load_model'] = addre + 'checkpoints/model_last_seg.pth'
    # config['load_model'] = 'output/_2022-12-30_21-35-30_g4L/checkpoints/model_best.pth'
    config['change_output'] = False
    config['batch_size'] = 3000 if not eval_metrics else 300
    # config['cnnencoderkernel']=3
    # config['cnnencoderhidden']=32
    # config['cnnencoderdilation']=5
    
    # config['max_seq_len'] = 500





    #########################################################################################################################
    # LOAD validation dataset for MA


    with open(filep_valid_id, 'r') as fp:
        data = list(csv.reader(fp))
        valid_set1 = []
        for i in range(len(data)):
            t = []
            for j in range(len(data[i])):
                t.append(float(data[i][j]))
            valid_set1.append(t)

    test_set1 = np.asarray(valid_set1)

    data_class = data_factory[config['data_class']]
    my_data = data_class(test_set1, dim=dim, n_proc=config['n_proc'],
                         limit_size=config['limit_size'], config=config)  # (sequencelenth*sample_num,feat_dim)

    #################################################################################################################################################
    ##Load model
    #################################################################################################################################################
    # Create model
    logger.info("Creating model ...")

    model = model_factory(config, my_data)

    if config['freeze']:
        for name, param in model.named_parameters():
            if name.startswith('output_layer'):
                param.requires_grad = True
            else:
                param.requires_grad = False

    # logger.info("Model:\n{}".format(model))
    logger.info("Total number of parameters: {}".format(utils.count_parameters(model)))
    logger.info("Trainable parameters: {}".format(utils.count_parameters(model, trainable=True)))

    # Initialize optimizer
    # config['l2_reg']
    # config['global_reg']
    if config['global_reg']:
        weight_decay = config['l2_reg']
        output_reg = None
    else:
        weight_decay = 0
        output_reg = config['l2_reg']
    # config['optimizer']选择优化器

    optim_class = get_optimizer(config['optimizer'])

    # optimizer = RAdam(model.parameters(), lr=config['lr'], weight_decay=weight_decay)
    optimizer = optim_class(model.parameters(), lr=config['lr'], weight_decay=weight_decay)

    start_epoch = 0
    lr_step = 0  # current step index of `lr_step`  默认值 config['lr_step']=[10000], config['lr_fractor']=[0.1]
    lr = config['lr']  # current learning step
    # Load model and optimizer state
    # if args.load_model:
    model, optimizer, start_epoch = utils.load_model(model, config['load_model'], optimizer, config['resume'],
                                                     config['change_output'],
                                                     config['lr'],
                                                     config['lr_step'],
                                                     config['lr_factor'])
    # 加载损失函数计算器
    # model = torch.nn.DataParallel(model, device_ids=gpus, output_device=gpus[0])
    model = model.to(device)

    # if addre_state:
    def load_other_model(addre_state):
        with open(os.path.join(addre_state, 'configuration.json')) as fp:
            # json.dump(config, fp, indent=4, sort_keys=True)
            config_state = json.load(fp)
        # dim = config_state['dimension']
        config_state['test_pattern'] = True
        config_state['load_model'] = addre_state + 'checkpoints/model_best_seg.pth'
        config_state['change_output'] = False
        config_state['batch_size'] = 3000
        config_state['cnnencoderkernel']=3
        config_state['cnnencoderhidden']=32
        config_state['cnnencoderdilation']=5
        model_state = model_factory(config_state, my_data)
        model_state = utils.load_model(model_state, config_state['load_model'], None, config_state['resume'],
                                                    config_state['change_output'],
                                                    config_state['lr'],
                                                    config_state['lr_step'],
                                                    config_state['lr_factor'])
        model_state = model_state.to(device)
        return model_state
    if addre_state:
        model_state = load_other_model(addre_state)
    if addre_cp:
        model_cp = load_other_model(addre_cp)

    loss_module = get_loss_module(config)
    metric = nn.L1Loss()

    dataset_class, collate_fn, runner_class, _ = pipeline_factory(config)


    # Create dataset
    def track_data(config, filename):
        with open(filename, 'r') as fp:  # 使用with写法文件读写完后会自动关闭，释放资源   其中‘r’表示对文件的操作为“只读”
            data = list(csv.reader(fp))  # 将csv文件存到一个列表data里面   data[1]表示csv文件中的第二行
            train_set1 = []
            for i in range(len(data)):
                t = []
                for j in range(len(data[i])):
                    t.append(float(data[i][j]))
                train_set1.append(t)

        train_set1 = np.asarray(train_set1)

        data_class = data_factory['Trackdata']
        my_data = data_class(train_set1, dim=config['dimension'], n_proc=-1,
                             limit_size=None, config=config)  # (sequencelenth*sample_num,feat_dim)
        train_indices = my_data.all_IDs

        # Note: currently a validation set must exist, either with `val_pattern` or `val_ratio`
        # Using a `val_pattern` means that `val_ratio` == 0 and `test_ratio` == 0

        logger.info("{} samples will be used for testing".format(len(train_indices)))

        train_dataset = dataset_class(my_data, train_indices)

        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=config['batch_size'],
                                  shuffle=False, #shuffle=True,
                                  num_workers=config['num_workers'],
                                  pin_memory=True,
                                  collate_fn=lambda x: collate_fn(x, max_len=model.max_len))
        print('max_len: ', model.max_len, my_data.max_seq_len)
        return train_dataset, train_loader, train_indices


    test_dataset, test_loader, test_indices = track_data(config, filep_valid_id)



    ###############################################################################################################################
    #test for ID recognition
    ###############################################################################################################################
    per_batch = {'target_masks': [], 'targets': [], 'predictions': [],
                 'metrics': [], 'IDs': []}
    all_target_prediction = {'targets': [],  'predictions': [],  'metrics': []}
    total_samples = 0
    epoch_loss = 0
    epoch_metrics = OrderedDict()
    epoch_num = None
    analyzer = analysis.Analyzer(print_conf_mat=True)
    model = model.eval()

    mae_metric_a = 0.
    mae_metric_k = 0.

    preds = []
    trues = []
    inputx = []
    pad=[]
    cps = []

    epoch_a_mae = 0  # total loss of epoch
    epoch_k_mae = 0  # total loss of epoch
    epoch_a_mse = 0  # total loss of epoch
    epoch_k_mse = 0  # total loss of epoch
    total_samples = 0  # total samples in epoch


    for i, batch in enumerate(test_loader):

        X, targets,  padding_masks, IDs = batch
        targets = targets.to(device)
        padding_masks = padding_masks.to(device)  # 0s: ignore
        # regression: (batch_size, num_labels); classification: (batch_size, num_classes) of logits
        ################################################################################################################################
        # 输出logits

        # predictions_m, predictions_a = model(X.to(device), padding_masks)

        predictions, _, _= model(X.to(device), padding_masks)
        if addre_state:
            pred_state, _, _ = model_state(X.to(device), padding_masks)
            predictions = torch.cat((predictions, pred_state), dim=-1)
        if addre_cp:
            pred_cp, _, _ = model_cp(X.to(device), padding_masks)
            predictions = torch.cat((predictions, pred_cp), dim=-1)

        # cp_output = model_cp(predictions[:, :, :2] if config['model'] == 'GatedLSTM' else predictions)
        # cps.append(cp_output.detach().cpu().numpy())

        #mae_a = metric(predictions[:,:,0], targets[:,:,0])
        #mae_k = metric(predictions[:,:,1], targets[:,:,1])

        lossmae=get_criterion('MAE')
        lossmse=get_criterion('MSE')
        lossmsle=get_criterion('MSLE')
        lossmale = get_criterion('MALE')

        a_mae = lossmae(predictions[:, :, 0], targets[:, :, 0])
        a_mse = lossmse(predictions[:, :, 0], targets[:, :, 0])
        #msle = lossmsle(predictions[:, :, 0], targets[:, :, 0])
        #a_male = lossmale(predictions[:, :, 0], targets[:, :, 0])

        k_mae = lossmae(predictions[:, :, 1], targets[:, :, 1])
        k_mse = lossmse(predictions[:, :, 1], targets[:, :, 1])
        #k_msle = lossmsle(predictions[:, :, 1], targets[:, :, 1])
        #k_male = lossmale(predictions[:, :, 1], targets[:, :, 1])
        #batch_loss = torch.sum(loss)
        #mean_loss = batch_loss / len(loss)  # mean loss (over active elements) used for optimization

        with torch.no_grad():
            # total_samples += len(loss)
            # total_samples += len(loss_a)
            total_samples += 1
            epoch_a_mae += a_mae.item()
            epoch_k_mae += k_mae.item()
            epoch_a_mse += a_mse.item()
            epoch_k_mse += a_mse.item()

        #mae_metric_a += mae_a.data
        #mae_metric_k += mae_k.data

        predictions= predictions.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()
        padding_masks=padding_masks.detach().cpu().numpy()
        preds.append(predictions)
        trues.append(targets)
        pad.append(padding_masks)
        inputx.append(X.detach().cpu().numpy())

        if i % 10 == 0:
            input = X.detach().cpu().numpy()
            for k in range(2):
                gt = np.array(targets[k, :, 0])
                pd = np.array(predictions[k, :, 0])
                visual(gt, pd, os.path.join(addre, str(i) + '_' + str(k) + 'a.pdf'))
                gtk = np.array(targets[k, :, 1])
                pdk = np.array(predictions[k, :, 1])
                visual(gtk, pdk, os.path.join(addre, str(i) + '_' + str(k) + 'k.pdf'))


    epoch_a_mae = epoch_a_mae / total_samples
    epoch_k_mae = epoch_k_mae / total_samples
    epoch_a_mse = epoch_a_mse / total_samples
    epoch_k_mse = epoch_k_mse / total_samples
    #mae_metric_a /= (i + 1)
    #mae_metric_k /= (i + 1)
    logger.info("epoch_a_mae: {0}, epoch_k_mae:{1}".format(epoch_a_mae,epoch_k_mae))
    logger.info("epoch_a_mse: {0}, epoch_k_mse:{1}".format(epoch_a_mse, epoch_k_mse))
    #print('a:'+str(mae_metric_a))
    #print('k:' + mae_metric_k)
    if eval_metrics == True:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'eval.csv'), 'a') as f:
            f.write("{},{},{},{},{}\n".format(filep_valid_id, epoch_a_mse, epoch_a_mae, epoch_k_mse, epoch_k_mae))
        return

    a=4
    
    pen_a = 10
    pen_k = 10
    # if exp in [0, 1, 4, 9, 10]:
    #     pen_a = 1
    #     pen_k = 1
    # elif exp in [3, 7]:
    #     pen_a = 0.1
    #     pen_k = 0.1
    # elif exp in [2]:
    #     pen_a = 0.5
    #     pen_k = 0.5
    # elif exp in [11]:
    #     pen_a = 1
    #     pen_k = 1
    assert len(preds) == 1
    total_num=len(preds[0])
    results=[]
    for i in range(total_num):
        label_a = preds[0][i][:, 0]
        label_k = preds[0][i][:, 1]
        if addre_state:
            label_state = np.argmax(preds[0][i][:, 2:2+4], axis=-1)
        padding = pad[0][i]
        count_true = sum(1 for elem in padding if elem)
        pre_a = label_a[:count_true].tolist()
        pre_k = label_k[:count_true].tolist()
        if addre_state:
            pre_state = label_state[:count_true].tolist()
        if addre_cp:
            label_cp = np.argmax(preds[0][i][:, -2:], axis=-1)
            # print(label_cp.shape, np.sum(label_cp[:count_true]))
            pre_cp = np.argwhere(label_cp[:count_true] > 0)
            # print(pre_cp)
            if pre_cp.shape[0] == 0:
                tmp_breakpoints = []
            else:
                tmp_breakpoints = pre_cp[:, 0].tolist()
            breakpoints = []
            for i in range(len(tmp_breakpoints) - 1):
                
                breakpoints.append(i)
            if count_true not in breakpoints:
                breakpoints.append(count_true)
        else:
            model2 = rpt.KernelCPD(kernel="linear", min_size=3, jump=15).fit(np.array(pre_a))
            # 检测变点
            breakpoints_a = model2.predict(pen=pen_a)
            model_k = rpt.KernelCPD(kernel="linear", min_size=3, jump=15).fit(np.array(pre_k))
            breakpoints_k = model_k.predict(pen=pen_k)
            # breakpoints = sorted(list(set(breakpoints_a).union(set(breakpoints_k))))
            breakpoints = merge_changepoints(breakpoints_a, breakpoints_k)
        print(breakpoints)
        # print(cps[0][i], cps[0][i].shape)
        # breakpoints = np.argwhere(np.argmax(cps[0][i][:count_true], axis=-1) > 0).squeeze(-1) + 1
        # breakpoints = breakpoints.tolist()
        # print([0] + breakpoints, breakpoints + [None])

        # 分割数组
        segments_a = [pre_a[i:j] for i, j in zip([0] + breakpoints, breakpoints + [None])]
        segments_k = [pre_k[i:j] for i, j in zip([0] + breakpoints, breakpoints + [None])]
        if addre_state:
            segments_state = [pre_state[i:j] for i, j in zip([0] + breakpoints, breakpoints + [None])]

        #todo:记录这一行信息
        tmp = []
        tmp.append(i)

        for j in range(len(breakpoints)):
            aver_k=sum(segments_k[j]) / len(segments_k[j])
            aver_k=10 ** aver_k
            # aver_k=np.exp(aver_k).item()
            # tmp.append(aver_k)
            aver_a = sum(segments_a[j]) / len(segments_a[j])
            # tmp.append(aver_a)
            aver_state=2
            if addre_state:
                aver_state = sum(segments_state[j]) / len(segments_state[j])
                aver_state = round(aver_state)
            if aver_a >= 1.88:
                print('alpha={} > 1.9, state={} -> 3'.format(aver_a, aver_state))
                aver_state = 3
                if aver_a > 1.99:
                    aver_a = 1.99
            elif aver_a < 1e-3:
                print('alpha={} -> 0,'.format(aver_a))
                aver_a = 0
                # aver_state = 0
            # elif aver_a < 0.05:
            #     print('alpha < 0.05, state={} -> 1'.format(aver_state))
            #     # aver_a = 0
            #     aver_state = 1
            if aver_k < 5e-3:
                if aver_k < 1e-12:
                    aver_k = 0
                # if aver_a < 0.5:
                #     print('K={}, state={} -> 0'.format(aver_k, aver_state))
                #     aver_state = 0
            tmp.append(aver_k)
            tmp.append(aver_a)
            tmp.append(aver_state)
            tmp.append(breakpoints[j])

        results.append(tmp)

    # 判断文件夹是否存在
    if not os.path.exists(save_dir):
        # 如果不存在则创建文件夹
        os.makedirs(save_dir, exist_ok=True)

    file = open(save_dir+'/fov_{}.txt'.format(fov), 'w')

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
    return preds[0], pad[0]



if __name__ == '__main__':
    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    addre_list = [
        # '/data1/jiangy/andi_tcu/code/andi_2/challenge_output/_2024-07-09_15-09-11_oyX/', # all_part
        # '/data1/jiangy/andi_tcu/code/andi_2/output/_2024-07-10_00-36-18_NEY/', # all part pipeline
        # '/data1/jiangy/andi_tcu/code/andi_2/challenge_output/_2024-07-09_15-10-16_iNz/', # sample
        # '/data1/jiangy/andi_tcu/code/andi_2/output/_2024-06-28_12-56-39_daR/',
        '/data1/jiangy/andi_tcu/code/andi_2/challenge_output/_2024-07-11_21-27-43_GmJ/',
    ]
    valid_file_list = [
            '/data1/jiangy/andi_data/0529/test/merge_3e5_new.csv',
            '/data1/jiangy/andi_data/0621_merge/test/merge_sample.csv',
            '/data1/jiangy/andi_data/0629_merge/test/merge_sample.csv',           
        ]
    for addre in addre_list:
        for valid_file in valid_file_list:
            save_dir = addre # '../../results/eval/{}_{}_{}_{}'.format(addre[0][-4:-1], addre[1][-4:-1], addre[2][-4:-1], addre[3][-4:-1])
            test_save(device, addre, valid_file, save_dir, exp=0, fov=0, eval_metrics=True)