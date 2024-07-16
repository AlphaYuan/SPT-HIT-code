import os
import numpy as np
import random

# 采样合并之后的文件

listt = []
new_lines = []
per_num = 300000
with open('/data1/jiangy/andi_data/0529/train/merge.csv', 'r') as f:
    lines = f.readlines()
    print(len(lines))
    # new_lines = random.sample(lines, per_num) # sample train
    # new_lines = random.sample(lines, 30000) # sample test
    # new_lines = lines
    # rand_idx = random.sample([i for i in range(0, len(lines) - 1)], 300000)
    # print('sample completed!')
    # for idx, line in enumerate(lines):
    #     if idx in rand_idx:
    #         new_lines.append(line)
    random.shuffle(lines)
    print('start write')
    for i in range(int(len(lines) / per_num)):
        new_lines = lines[i*per_num:(i+1)*per_num]
        print('processing part: {}, num: {}'.format(i, len(new_lines)))
        with open('/data1/jiangy/andi_data/0529/train/merge_sample_part{}.csv'.format(i), 'w') as f2:
            f2.writelines(new_lines)
# print('start write')
# with open('/data1/jiangy/andi_data/0626_multi_state/train/merge_sample.csv', 'w') as f2:
#     f2.writelines(new_lines)

# 合并不同model的轨迹

# root = './datasets/andi_set/0621_merge/train'
# os.makedirs(root, exist_ok=True)
# files = ['/newdata/jiangy/project/andi_challenge/andi_solver/datasets/andi_set/0621_confinement/train/merge.csv', '/newdata/jiangy/project/andi_challenge/andi_solver/datasets/andi_set/0621_dimerization/train/merge.csv', '/newdata/jiangy/project/andi_challenge/andi_solver/datasets/andi_set/0621_immobile_traps/train/merge.csv', '/newdata/jiangy/project/andi_challenge/andi_solver/datasets/andi_set/0602/train/merge.csv']
# root = './datasets/andi_set/0621_merge/test'
# os.makedirs(root, exist_ok=True)
# files = ['/newdata/jiangy/project/andi_challenge/andi_solver/datasets/andi_set/0621_confinement/test/merge.csv', '/newdata/jiangy/project/andi_challenge/andi_solver/datasets/andi_set/0621_dimerization/test/merge.csv', '/newdata/jiangy/project/andi_challenge/andi_solver/datasets/andi_set/0621_immobile_traps/test/merge.csv', '/newdata/jiangy/project/andi_challenge/andi_solver/datasets/andi_set/0602/test/merge.csv']
# print(files)

# for file in files:
#     print(file)
#     with open(file, 'r') as f:
#         lines = f.readlines()
#         print(lines[-1][-1])
#         with open(os.path.join(root, 'merge.csv'), 'a') as f2:
#             f2.writelines(lines)
#             if lines[-1][-1] != '\n':
#                 print('not ended')
#                 f2.write('\n')

# 合并不同长度的原始数据csv

# root = '/data1/jiangy/andi_data/0626_multi_state/test'
# files = os.listdir(root)
# print(files)

# for file in files:
#     print(file)
#     with open(os.path.join(root, file), 'r') as f:
#         lines = f.readlines()
#         print(lines[-1][-1])
#         with open(os.path.join(root, 'merge.csv'), 'a') as f2:
#             f2.writelines(lines)
#             if lines[-1][-1] != '\n':
#                 print('not ended')
#                 f2.write('\n')