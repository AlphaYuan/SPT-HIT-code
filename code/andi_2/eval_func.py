import numpy as np
import ruptures as rpt
from scipy.optimize import linear_sum_assignment

import torch

def evaluate_cp_prediction(true, pred, changepoint_threshold=10):
    "Evaluates the change point prediction."
    true_positive = 0
    false_positive = max(len(pred) - len(true), 0)
    false_negative = max(len(true) - len(pred), 0)
    squared_error = []
    all_squared_error = []
    
    assignment, cost = assign_changepoints(true, pred)
    for idx in assignment:
        difference = np.abs(true[idx[0]] - pred[idx[1]])
        difference_square = difference**2
        all_squared_error.append(difference_square)
        if difference < changepoint_threshold:
            true_positive += 1
            squared_error.append(difference_square)
        else:
            false_positive += 1
            false_negative += 1
            
    return {'squared_error': squared_error, 
            'tp': true_positive, 
            'fp': false_positive, 
            'fn': false_negative, 
            'assignment': assignment, 
            'all_squared_error': all_squared_error,
            'cost': cost}

def assign_changepoints(true, pred):
    "Matches predicted and true changepoints solving a linear sum assignment problem."
    cost = np.zeros((len(true), len(pred)))
    for i, t in enumerate(true):
        cost[i, :] = np.abs(t-pred)
    return np.array(linear_sum_assignment(cost)).T, cost

def jaccard_index(true_positive, false_positive, false_negative):
    "Computes the Jaccard index a.k.a. Tanimoto index."
    return true_positive/(true_positive + false_positive + false_negative)

def f1_score(true_positive, false_positive, false_negative):
    "Computes the Jaccard index a.k.a. Tanimoto index."
    return 2*true_positive/(2*true_positive + false_positive + false_negative)

def metric_CP(gt, pred):
    res_list = []
    rmse_list = []
    # jsc_list = []
    tp_list = []
    fp_list = []
    fn_list = []
    assign_list = []
    # pred[:, 0, :] = 1
    # pred[:, -1, :] = 1
    for idx in range(pred['alpha'].shape[0]):
        true = torch.where(gt[idx] == 1)[0].cpu().numpy() + 1
        # p = torch.where(pred[idx] > 0.5)[0].cpu().numpy()
        p = get_CP(pred['alpha'][idx].detach().cpu().numpy(), pred['D'][idx].detach().cpu().numpy())
        res = evaluate_cp_prediction(true, p, changepoint_threshold=10)
        res_list.append(res)
        rmse_list.append(np.mean(res['squared_error']))
        tp_list.append(res['tp'])
        fp_list.append(res['fp'])
        fn_list.append(res['fn'])
        # print(res['assignment'])
        assign = np.array([true[res['assignment'][:,0]], p[res['assignment'][:,1]]]).T
        assign_list.append(assign)
        # assign_list.append(np.concatenate((np.array([[0,0]]), assign), axis=0))
        # jsc = jaccard_index(res['tp'], res['fp'], res['fn'])
        # jsc_list.append(jsc)
    return res_list, rmse_list, tp_list, fp_list, fn_list, assign_list


def metric_alpha(gt, pred, assign_list):
    mae = []
    for i in range(pred.shape[0]):
        seg_mae = []
        for j in range(1, len(assign_list[i])):
            # print(assign_list[i][j-1], assign_list[i][j])
            seg_gt = gt[i][assign_list[i][j-1][0]:assign_list[i][j][0]]
            seg_pred = pred[i][assign_list[i][j-1][1]:assign_list[i][j][1]]
            if len(seg_pred) == 0 or len(seg_gt) == 0:
                # print('continue')
                continue
            seg_mean_abs = np.abs(np.mean(seg_pred) - np.mean(seg_gt))
            seg_mae.append(seg_mean_abs)
        if len(seg_mae) != 0:
            mae.append(np.mean(seg_mae))
        # break
    # print(mae)
    return np.mean(mae)

def metric_D(gt, pred, assign_list):
    msle = []
    for i in range(pred.shape[0]):
        seg_msle = []
        for j in range(1, len(assign_list[i])):
            # print(assign_list[i][j-1], assign_list[i][j])
            seg_gt = gt[i][assign_list[i][j-1][0]:assign_list[i][j][0]]
            seg_pred = pred[i][assign_list[i][j-1][1]:assign_list[i][j][1]]
            if len(seg_pred) == 0 or len(seg_gt) == 0:
                # print('continue')
                continue
            seg_log = (np.log(np.mean(seg_pred) + 1) - np.log(np.mean(seg_gt) + 1)) ** 2
            seg_msle.append(seg_log)
        if len(seg_msle) != 0:
            msle.append(np.mean(seg_msle))
        # break
    # print(msle)
    return np.mean(msle)


def metric_state(gt, pred, assign_list):
    tp_list = []
    fp_list = []
    fn_list = []
    for i in range(pred.shape[0]):
        true_positive = 0
        false_positive = 0
        false_negative = 0
        for j in range(1, len(assign_list[i])):
            # print(assign_list[i][j-1], assign_list[i][j])
            seg_gt = gt[i][assign_list[i][j-1][0]:assign_list[i][j][0]]
            seg_pred = np.argmax(pred[i][assign_list[i][j-1][1]:assign_list[i][j][1], :], axis=1)
            # print(np.mean(seg_gt), np.mean(seg_pred))
            if len(seg_pred) == 0 or len(seg_gt) == 0:
                # print('continue')
                continue
            if np.mean(seg_gt) == np.mean(seg_pred):
                true_positive += 1
            else:
                false_positive += 1
                false_negative += 1

        tp_list.append(true_positive)
        fp_list.append(false_positive)
        fn_list.append(false_negative)
        # break
    f1 = f1_score(np.sum(tp_list), np.sum(fp_list), np.sum(fn_list))
    # print(f1)
    return f1


def get_CP(seq_alpha, seq_D):
    # assert seq_alpha.shape[0] == 200
    # algo = rpt.Pelt(model='rbf').fit(seq_D)
    # result = algo.predict(pen=10)
    # return result

    CP_list = []
    for i in range(1, seq_alpha.shape[0]):
        if np.abs((seq_alpha[i] - seq_alpha[i-1]) / max(seq_alpha[i], seq_alpha[i-1])) > 0.2:# or np.abs((seq_D[i] - seq_D[i-1]) / max(seq_D[i], seq_D[i-1])) > 0.3:
            if len(CP_list) != 0 and (i - CP_list[-1]) < 3:
                continue
            CP_list.append(i)
    CP_list.append(seq_alpha.shape[0])
    # if (int(seq_alpha.shape[0]) - 1) not in CP_list:
    #     CP_list.append(seq_alpha.shape[0])
    # if 200 not in CP_list:
    #     print(CP_list)
    return np.array(CP_list)


def split_all(pred, CPs=None):
    alpha = []
    D = []
    state = []
    CP = []
    for i in range(pred['alpha'].shape[0]):
        seg_alpha = []
        seg_D = []
        seg_state = []
        seg_CP = get_CP(pred['alpha'][i], pred['D'][i]) if CPs is None else CPs[i] #np.argwhere(pred['CP'][i] > 0.5)[:, 1]
        assign_list = seg_CP
        # print(assign_list)
        # print(seg_CP)
        if assign_list[0] != 0:
            # print('j=0')
            j=0
            seg_alpha.append(np.mean(pred['alpha'][i][0:assign_list[j]]))
            seg_D.append(np.mean(pred['D'][i][0:assign_list[j]]))
            seg_state.append(np.mean(np.argmax(pred['state'][i][0:assign_list[j], :], axis=-1)))
            # seg_CP.append(assign_list[i][j][1])

        for j in range(1, len(assign_list)):
            # print(assign_list[j-1], assign_list[j])
            seg_alpha.append(np.mean(pred['alpha'][i][assign_list[j-1]:assign_list[j]]))
            seg_D.append(np.mean(pred['D'][i][assign_list[j-1]:assign_list[j]]))
            seg_state.append(np.mean(np.argmax(pred['state'][i][assign_list[j-1]:assign_list[j], :], axis=-1)))
            # seg_CP.append(assign_list[i][j][1])
        alpha.append(seg_alpha)
        D.append(seg_D)
        state.append(seg_state)
        CP.append(seg_CP)
        # break
    # print(mae)
    return alpha, D, state, CP


def split_single_traj(pred, CPs=None):
    alpha = []
    D = []
    state = []
    CP = []
    # for i in range(pred['alpha'].shape[0]):
    seg_alpha = []
    seg_D = []
    seg_state = []
    seg_CP = get_CP(pred['alpha'], pred['D']) if CPs is None else CPs #np.argwhere(pred['CP'][i] > 0.5)[:, 1]
    assign_list = seg_CP
    # print(assign_list[0], assign_list[0] == 0, type(assign_list))
    # print(assign_list)
    # print(seg_CP)
    if assign_list[0] != 0:
        # print('j=0')
        j=0
        seg_alpha.append(np.mean(pred['alpha'][0:assign_list[j]]))
        seg_D.append(np.mean(pred['D'][0:assign_list[j]]))
        seg_state.append(np.mean(np.argmax(pred['state'][0:assign_list[j], :], axis=-1)))
        # seg_CP.append(assign_list[i][j][1])

    for j in range(1, len(assign_list)):
        # print(assign_list[j-1], assign_list[j])
        seg_alpha.append(np.mean(pred['alpha'][assign_list[j-1]:assign_list[j]]))
        seg_D.append(np.mean(pred['D'][assign_list[j-1]:assign_list[j]]))
        seg_state.append(np.mean(np.argmax(pred['state'][assign_list[j-1]:assign_list[j], :], axis=-1)))
        # seg_CP.append(assign_list[i][j][1])
    alpha.append(seg_alpha)
    D.append(seg_D)
    state.append(seg_state)
    CP.append(seg_CP)
        # break
    # print(mae)
    return alpha, D, state, CP