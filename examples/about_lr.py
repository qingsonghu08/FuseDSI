# -*- coding: UTF-8 -*-
'''
@Project ：cluster-contrast-reid-v3
@File ：about_lr.py
@Author ：棋子
@Date ：2023/9/20 14:45
@Software: PyCharm
'''

import numpy as np

# 定义函数 f(x)
def f(x, growth_rate=0.03, max_e=-20):

    # f = np.exp(growth_rate * (x -50))
    f = np.exp(growth_rate * (x - max_e))
    return f

def draw():
    # 画出函数的折线图
    import matplotlib.pyplot as plt
    import numpy as np

    # 生成 x 值的范围
    x = np.linspace(0, 50, 100)  # 生成从0到60的100个点作为 x 值

    # 计算相应的 y 值
    y = f(x)

    # 创建折线图
    plt.plot(x, y, label='f(x)')

    # 添加标签和标题
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('f(x)')

    # 添加图例
    plt.legend()

    # 显示图形
    plt.grid(True)  # 添加网格线
    plt.show()

def custom_metric(x):
    # 控制增长速率的参数
    growth_rate = 0.01

    # 确保x在[0, 50]范围内
    # x = np.clip(x, 0, 50)

    # 使用指数函数实现变化
    f_x = (np.exp(growth_rate * x) - 1) / (np.exp(growth_rate * 50) - 1)
    # f_x = (np.exp(growth_rate * x)) / (np.exp(growth_rate * 50))

    return f_x


a1 = f(x=0)
a2 = f(x=49)
print(a1, a2)

draw()







