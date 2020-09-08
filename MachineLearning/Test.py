import numpy as np
import matplotlib.pyplot as plt
import math

print(np.percentile(range(1, 101), 75))

import random
import pandas as pd
import numpy as np
from pandas import Series,DataFrame
#用随机数产生一个二维数组。分别是年龄的性别。
df=pd.DataFrame({'Age':np.random.randint(0,70,100),
                'Sex':np.random.choice(['M','F'],100),
                })
#用cut函数对于年龄进行分段分组，用bins来对年龄进行分段，左开右闭
age_groups=pd.cut(df['Age'],bins=[0,18,35,55,70,100])
# print(age_groups)
print(df.groupby(age_groups).count())
print(df.groupby(age_groups).sum())

'''
Burg算法求解5阶AR模型参数
并绘制p分别取3,4,5时的功率谱曲线
'''
u = [101, 82, 66, 35, 31, 7, 20, 92, 154, 125, 85, 68, 38, 23, 10, 24, 83, 132, 131, 118, 90, 67, 60, 47, 41, 21, 16, 6,
     4, 7, 14, 34, 45, 43, 48, 42, 28, 10, 8, 2, 0, 1, 5, 12, 14, 35, 46, 41, 30, 24, 16, 7, 4, 2, 8, 17, 36, 50, 62,
     67, 71, 48, 28, 8, 13, 57, 122, 138, 103, 86, 63, 37, 24, 11, 15, 40, 62, 98, 124, 96, 66, 64, 54, 39, 21, 7, 4,
     23, 55, 94, 96, 77, 59, 44, 47, 30, 16, 7, 37, 74]
N = len(u)
print("数据长度为:" + str(N))
k = 5  # 阶数

# 数据初始化
fO = u[:]  # 0阶前向误差
bO = u[:]  # 0阶反向误差
f = u[:]  # 用于更新的误差变量
b = u[:]
a = np.array(np.zeros((k + 1, k + 1)))  # 模型参数初始化
for i in range(k + 1):
    a[i][0] = 1
# 计算P0 1/N*sum(u*2)
P0 = 0
for i in range(N):
    P0 += u[i] ** 2
P0 /= N
print("P0:" + str(P0))
P = [P0]

# Burg 算法更新模型参数
for p in range(1, k + 1):
    Ka = 0  # 反射系数的分子
    Kb = 0  # 反射系数的分母
    for n in range(p, N):
        Ka += f[n] * b[n - 1]
        Kb = Kb + f[n] ** 2 + b[n - 1] ** 2
    K = 2 * Ka / Kb
    print("第%d阶反射系数:%f" % (p, K))
    # 更新前向误差和反向误差
    fO = f[:]
    bO = b[:]
    for n in range(p, N):
        b[n] = -K * fO[n] + bO[n - 1]
        f[n] = fO[n] - K * bO[n - 1]
    # 更新此时的模型参数
    print("第%d阶模型参数：" % p)
    for i in range(1, p + 1):
        if (i == p):
            a[p][i] = -K
        else:
            a[p][i] = a[p - 1][i] - K * a[p - 1][p - i]
        print("a%d=%f" % (i, a[p][i]))
    P.append((1 - K ** 2) * P[p - 1])
    print("第%d阶模型的平均功率：%f" % (p, P[p]))


# 计算第k阶的功率谱
def calPSD(k, l=512):
    H = np.array(np.zeros(l), dtype=complex)
    for f in range(l):
        f1 = f * 0.5 / l  # 频率值
        for i in range(1, k + 1):
            H[f] += complex(a[k][i] * np.cos(2 * np.pi * f1 * i), -a[k][i] * np.sin(2 * np.pi * f1 * i))
        H[f] += 1
        H[f] = 1 / H[f]  # 系统函数的表达式
        H[f] = 10 * math.log10(np.abs(H[f]) ** 2 * P[k])
    return H


H3 = calPSD(3)
H4 = calPSD(4)
H5 = calPSD(5)

# 绘制功率谱曲线
l = 512
plt1, = plt.plot(np.arange(0, 0.5, 0.5 / l), H3, 'r-', label="k=3")
plt2, = plt.plot(np.arange(0, 0.5, 0.5 / l), H4, 'g--', label="k=4")
plt3, = plt.plot(np.arange(0, 0.5, 0.5 / l), H5, 'b-.', label="k=5")
plt.xlabel("Frequency (Hz)")
plt.ylabel('PSD (dB/Hz)')
plt.legend([plt1, plt2, plt3], ('k=3', 'k=4', 'k=5'))
plt.title('The curve of power spectrum(p=3,4,5)')
plt.show()