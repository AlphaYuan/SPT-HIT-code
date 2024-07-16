import ruptures as rpt  # 导入ruptures
import numpy as np       # 导入numpy

# 创建模拟数据
n = 1000  # 数据点数量
n_bkps = 3  # 真实变点数量
signal, bkps = rpt.pw_constant(n, n_bkps, noise_std=1)  # 生成带有变点的数据

# 创建KernelCPD对象
model = rpt.KernelCPD(kernel="linear", min_size=20, jump=15).fit(signal)

# 检测变点
breakpoints = model.predict(pen=30)

# 可视化结果
rpt.display(signal, bkps, breakpoints,computed_chg_pts_color="r")
import matplotlib.pyplot as plt
plt.show()
a=0

# 创建一个txt文件
file = open('output.txt', 'w')

# 定义一个数组
array = [1, 1.35, 2]

# 将数组逐行写入txt文件，用逗号隔开
for item in array:
    file.write(str(item) + ',\n')

# 关闭文件
file.close()

# 定义原数组和changepoint
array = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]  # 假设原数组为这个
changepoints = [3, 6, 9]

# 分割数组
segments = [array[i:j] for i, j in zip([0] + changepoints, changepoints + [None])]

print(segments)  # 输出分割后的数组
