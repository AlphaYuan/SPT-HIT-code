a = [1, 2, 3, 4, 5]  # 原始列表a

b = a[::-1]  # 将列表a倒序得到列表b

new_list = a + b  # 拼接a和b形成一个新列表

print(new_list)  # 输出新列表

import numpy as np  # 导入numpy库

# 原始列表a
a = np.array([[1, 2, 3, 4, 5],
              [6, 7, 8, 9, 10]])

# 将a的每一行倒序
b = np.flip(a, axis=1)

# 拼接a和b形成一个新列表
new_list = np.concatenate((a, b), axis=1)

print(new_list)  # 输出新列表

# 原始列表a
a = [[1, 2, 3, 4, 5],
     [6, 7, 8],
     [9, 10]]

# 将a的每一个子序列倒序
b = [row[::-1] for row in a]

# 拼接a和b形成一个新列表
new_list = a + b

print(new_list)  # 输出新列表

import numpy as np  # 导入numpy库



# 原始列表a
a = [[1, 2, 3, 4, 5],
     [6, 7, 8],
     [9, 10]]

# 将a的每一个子序列倒序
b = [row[::-1] for row in a]

# 横向合并a和b
new_list = [x + y for x, y in zip(a, b)]

print(new_list)  # 输出新列表

new_list = [x + y+z for x, y,z in zip(a, b,a)]

print(new_list)  # 输出新列表
import os
# 判断文件夹是否存在
save_dir='../../results/track_2/exp_0'
if not os.path.exists(save_dir):
    # 如果不存在则创建文件夹
    os.makedirs(save_dir, exist_ok=True)

file = open(save_dir+'/fov_{}.txt'.format(0), 'w')