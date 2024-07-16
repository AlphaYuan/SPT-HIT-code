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
import real_llla
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

from transmodel2 import model_factory
from losscompute import get_loss_module
from optimizers import get_optimizer
from utils import utils, analysis
import numpy as np
import pandas as pd
import seaborn as sns
from torch.autograd import Variable

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


if __name__ == '__main__':
    ################################################################################################################################
    #setting
    args = Options().parse()  # `argsparse` object
    config = setup(args)  # configuration dictionary'''

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    print(f"[Info]: Use {device} now!")

    myseed = 42069  # set a random seed for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(myseed)
    torch.manual_seed(myseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)



    addre='output/_2024-05-28_23-32-33_iSV/'
    addre = 'output/_2024-05-29_01-27-50_QQJ/'
    addre = 'output/_2024-05-30_01-04-12_H8L/'    #transmodel2 hiddenlayer=8
    addre = 'output/_2024-05-30_21-53-33_Oj2/'  # transmodel2 hiddenlayer=8
    addre = 'output/_2024-05-31_12-45-56_9Hp/'  # transmodel2 hiddenlayer=8





    with open(os.path.join(addre, 'configuration.json')) as fp:
        # json.dump(config, fp, indent=4, sort_keys=True)
        config = json.load(fp)

    tlen = config['tlen']
    tnum = config['tnum']
    # dim = 3
    dim = config['dimension']
    config['test_pattern'] = True
    config['load_model'] = addre + 'checkpoints/model_best_seg.pth'
    # config['load_model'] = 'output/_2022-12-30_21-35-30_g4L/checkpoints/model_best.pth'
    config['change_output'] = False
    config['batch_size'] = 3000
    filep_valid_id = config['valid_address']
    filep_valid_id = '../../data/test0_Nall_12000_T20_200_K-12_6.csv'
    filep_valid_id = '../../data/test0_Nall_5000_T20_200_K-12_6.csv'





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

    logger.info("Model:\n{}".format(model))
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

    loss_module = get_loss_module(config)
    metric = nn.L1Loss()

    dataset_class, collate_fn, runner_class = pipeline_factory(config)


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
                                  shuffle=True,
                                  num_workers=config['num_workers'],
                                  pin_memory=True,
                                  collate_fn=lambda x: collate_fn(x, max_len=model.max_len))

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


    a=4

    total_num=len(preds[0])
    results=[]
    for i in range(total_num):
        label_a = preds[0][i][:, 0]
        label_k = preds[0][i][:, 1]
        padding = pad[0][i]
        count_true = sum(1 for elem in padding if elem)
        pre_a = label_a[:count_true].tolist()
        pre_k = label_k[:count_true].tolist()
        model2 = rpt.KernelCPD(kernel="linear", min_size=10, jump=15).fit(np.array(pre_a))
        # 检测变点
        breakpoints = model2.predict(pen=30)
        # 分割数组
        segments_a = [pre_a[i:j] for i, j in zip([0] + breakpoints, breakpoints + [None])]
        segments_k = [pre_k[i:j] for i, j in zip([0] + breakpoints, breakpoints + [None])]

        #todo:记录这一行信息
        tmp = []
        tmp.append(i)

        for j in range(len(breakpoints)):
            aver_k=sum(segments_k[j]) / len(segments_k[j])
            aver_k=10 ** aver_k
            tmp.append(aver_k)
            aver_a = sum(segments_a[j]) / len(segments_a[j])
            tmp.append(aver_a)
            state=2
            tmp.append(2)
            tmp.append(breakpoints[j])

        results.append(tmp)

    file = open('output.txt', 'w')

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



