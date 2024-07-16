from andi_datasets.utils_trajectories import plot_trajs
import random
import os
import argparse
import csv


with open('../../../../jiangy/andi_data/0529/train/merge.csv', 'r') as fp:
    data = list(csv.reader(fp))  #
    valid_set1 = []
    for i in range(len(data)):
        t = []
        for j in range(len(data[i])):
            t.append(float(data[i][j]))
        valid_set1.append(t)



'../../../../jiangy/project/andi_challenge/andi_solver/datasets/andi_set/0529/train/merge.csv', 'r'

valid_set1 = np.asarray(valid_set1)