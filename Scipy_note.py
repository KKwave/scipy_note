# import numpy as np
# import scipy as sp
# import matplotlib as mpl
# import matplotlib.pyplot as plt

# 我们可以使用 numpy 中的 info 函数来查看函数的文档：
# np.info(optimize.fmin)

# 可以用 lookfor 来查询特定关键词相关的函数：
# np.lookfor("resize array")

# 还可以指定查找的模块：
# np.lookfor("remove path", module="os"
#
# 设置 Numpy 浮点数显示格式：
# np.set_printoptions(precision=2, suppress=True))

# 插值
# 先导入一维插值函数 interp1d：
# interp1d(x, y)
# from scipy.interpolate import interp1d
# interp1d 默认的插值方法是线性，关于线性插值的定义

# 多项式插值
# 我们可以通过 kind 参数来调节使用的插值方法，来得到不同的结果：
# nearest 最近邻插值
# zero 0阶插值
# linear 线性插值
# quadratic 二次插值
# cubic 三次插值
# 4,5,6,7 更高阶插值
# 最近邻插值：
# cp_ch4 = interp1d(data['TK'], data['Cp'], kind="nearest")

# 对于二维乃至更高维度的多项式插值：
# from scipy.interpolate import interp2d, interpnd
# 其使用方法与一维类似。

# 导入 Scipy 的统计模块：
# import scipy.stats.stats as st
# 其他统计量：height为数据
# print 'median, ', st.nanmedian(heights)    # 忽略nan值之后的中位数
# print 'mode, ', st.mode(heights)           # 众数及其出现次数
# print 'skewness, ', st.skew(heights)       # 偏度
# print 'kurtosis, ', st.kurtosis(heights)   # 峰度


# -----------------------------------------------------------------
# 常见的概率分布都可以在scipy.stats 中找到

# 正态分布
# 以正态分布为例，先导入正态分布：
# from scipy.stats import norm
# 它包含四类常用的函数：
# norm.cdf 返回对应的累计分布函数值
# norm.pdf 返回对应的概率密度函数值
# norm.rvs 产生指定参数的随机变量
# norm.fit 返回给定数据下，各参数的最大似然估计（MLE）值
# 从正态分布产生500个随机点：
# x_norm = norm.rvs(size=500)

# 导入积分函数：
# from scipy.integrate import trapz
# 通过积分，计算落在某个区间的概率大小：
# x1 = linspace(-2,2,108)
# p = trapz(norm.pdf(x1), x1)

# 其他连续分布
# from scipy.stats import lognorm, t, dweibull
# 支持与 norm 类似的操作，如概率密度函数等。

# 离散分布
# 导入离散分布：
# from scipy.stats import binom, poisson, randint
#
# 假设检验
# 导入相关的函数：
# 正态分布
# 独立双样本 t 检验，配对样本 t 检验，单样本 t 检验
# 学生 t 分布
# from scipy.stats import norm
# from scipy.stats import ttest_ind, ttest_rel, ttest_1samp
# from scipy.stats import t

# 独立样本 t 检验
# 两组参数不同的正态分布：
# n1 = norm(loc=0.3, scale=1.0)
# n2 = norm(loc=0, scale=1.0)
# 从分布中产生两组随机样本：
# n1_samples = n1.rvs(size=100)
# n2_samples = n2.rvs(size=100)
# 将两组样本混合在一起：
# samples = hstack((n1_samples, n2_samples))
# 最大似然参数估计：
# loc, scale = norm.fit(samples)
# n = norm(loc=loc, scale=scale)
# 比较：
# x = linspace(-3,3,100)
# hist([samples, n1_samples, n2_samples], normed=True)
# plot(x, n.pdf(x), 'b-')
# plot(x, n1.pdf(x), 'g-')
# plot(x, n2.pdf(x), 'r-')

# 独立双样本 t 检验的目的在于判断两组样本之间是否有显著差异：
# t_val, p = ttest_ind(n1_samples, n2_samples)
# print 't = {}'.format(t_val)
# print 'p-value = {}'.format(p)
# p 值小，说明这两个样本有显著性差异


# --------------------------------------
# 曲线拟合
# 多项式拟合
# 导入线多项式拟合工具：
# from numpy import polyfit, poly1d
# 产生数据：
# x = np.linspace(-5, 5, 100)
# y = 4 * x + 1.5
# noise_y = y + np.random.randn(y.shape[-1]) * 2.5
# 进行线性拟合，polyfit 是多项式拟合函数，线性拟合即一阶多项式：
# coeff = polyfit(x, noise_y, 1)
# print(coeff) 返回两个参数
# 还可以用 poly1d 生成一个以传入的 coeff 为参数的多项式函数：
# f = poly1d(coeff)
# p = plt.plot(x, noise_y, 'rx')
# p = plt.plot(x, f(x))

# 多项式拟合正弦函数
# 正弦函数：
# x = np.linspace(-np.pi,np.pi,100)
# y = np.sin(x)
# 用一阶到九阶多项式拟合，类似泰勒展开：
# y1 = poly1d(polyfit(x,y,1))
# y3 = poly1d(polyfit(x,y,3))
# y5 = poly1d(polyfit(x,y,5))
# y7 = poly1d(polyfit(x,y,7))
# y9 = poly1d(polyfit(x,y,9))

# 最小二乘拟合
# 导入相关的模块：
# from scipy.linalg import lstsq
# from scipy.stats import linregress
# 生成数据
# x = np.linspace(0,5,100)
# y = 0.5 * x + np.random.randn(x.shape[-1]) * 0.35
# Scipy.linalg.lstsq 最小二乘解
# 要得到 C ，可以使用 scipy.linalg.lstsq 求最小二乘解。
# 这里，我们使用 1 阶多项式即 N = 2，先将 x 扩展成 X：
# X = np.hstack((x[:,np.newaxis], np.ones((x.shape[-1],1))))
# 求解：
# C, resid, rank, s = lstsq(X, y) 对应sum squared residual，rank of the X matrix，singular values of X

# Scipy.stats.linregress 线性回归
# 对于上面的问题，还可以使用线性回归进行求解：
# slope, intercept, r_value, p_value, stderr = linregress(x, y)

# 更高级的拟合
# from scipy.optimize import leastsq
# Scipy.optimize.curve_fit
# 更高级的做法：
# from scipy.optimize import curve_fit
# 不需要定义误差函数，直接传入 function 作为参数：  （function是一个非线性函数）
# p_est, err_est = curve_fit(function, x, y_noisy)
# 第一个返回的是函数的参数，第二个返回值为各个参数的协方差矩阵

# ---------------------------------------------------------------
# 最小化函数
# minimize 函数
# 导入 scipy.optimize.minimize：
# from scipy.optimize import minimize
# result = minimize(neg_dist, 40, args=(1,))
# minimize 接受三个参数：第一个是要优化的函数，第二个是初始猜测值，第三个则是优化函数的附加参数，
# 默认 minimize 将优化函数的第一个参数作为优化变量，所以第三个参数输入的附加参数从优化函数的第二个参数开始。
# 优化方法
# minimize 函数默认根据问题是否有界或者有约束，使用 'BFGS', 'L-BFGS-B', 'SLSQP' 中的一种;
# 默认没有约束时，使用的是 BFGS 方法；改变 minimize 使用的算法，传入method参数
# 利用 callback 参数查看迭代的历史：
# result = minimize(rosen, x0, callback=xi.append)


# -------------------------------------------------------
# 积分
# 符号运算可以用 sympy 模块完成。
# 先导入 init_printing 模块方便其显示：
# from sympy import init_printing
# init_printing()
# from sympy import symbols, integrate
# import sympy
#
# 产生 x 和 y 两个符号变量，并进行运算：
# x, y = symbols('x y')
# sympy.sqrt(x ** 2 + y ** 2)
# 对于生成的符号变量 z，我们将其中的 x 利用 subs 方法替换为 3：
# z = sympy.sqrt(x ** 2 + y ** 2)
# z.subs(x, 3)  此时这个方程的x被赋值为3
# z.subs(x, 3).subs(y, 4) 都替换则直接得出结果5
#
# 还可以从 sympy.abc 中导入现成的符号变量：
# from sympy.abc import theta
# y = sympy.sin(theta) ** 2
#
# 对 y 进行积分：
# Y = integrate(y)
# 产生不定积分对象：
# Y_indef = sympy.Integral(y)
# 定积分：
# Y_def = sympy.Integral(y, (theta, 0, sympy.pi))
#
# 数值积分
# 导入贝塞尔函数：
# from scipy.special import jv
# def f(x):
#     return jv(2.5, x)

# 还有很多具体建议到教程查看

# ------------------------------------
# 积分求解
# 例子
# def dy_dt(y, t):
#     return np.sin(t)
# 积分求解：
# from scipy.integrate import odeint
# t = np.linspace(0, 2*pi, 100)
# result = odeint(dy_dt, 0, t)

# ---------------------------------------
# 稀疏矩阵
# 构建稀疏矩阵
# from scipy.sparse import *
# import numpy as np
# 创建一个空的稀疏矩阵：
# coo_matrix((2,3))
#
# sparse.find 函数
# 返回一个三元组，表示稀疏矩阵中非零元素的 (row, col, value)：
# from scipy import sparse
# row, col, val = sparse.find(C)
#
# sparse.issparse 函数
# 查看一个对象是否为稀疏矩阵：
# sparse.issparse(B)

# --------------------------------------------
# 线性代数
# numpy 和 scipy 中，负责进行线性代数部分计算的模块叫做 linalg
# import numpy as np
# import numpy.linalg
# import scipy as sp
# import scipy.linalg
# import matplotlib.pyplot as plt
# from scipy import linalg
# %matplotlib inline
#
# numpy.linalg VS scipy.linalg
# 一方面scipy.linalg 包含 numpy.linalg 中的所有函数，同时还包含了很多 numpy.linalg 中没有的函数。
# 另一方面，scipy.linalg 能够保证这些函数使用 BLAS/LAPACK 加速，而 numpy.linalg 中这些加速是可选的。
# 因此，在使用时，我们一般使用 scipy.linalg 而不是 numpy.linalg。
#
# 线性代数的基本操作对象是矩阵，而矩阵的表示方法主要有两种：numpy.matrix 和 2D numpy.ndarray
#
# numpy.matrix
# numpy.matrix 是一个矩阵类，提供了一些方便的矩阵操作：
# 支持类似 MATLAB 创建矩阵的语法
# 矩阵乘法默认用 * 号
# .I 表示逆，.T 表示转置
# 可以用 mat 或者 matrix 来产生矩阵：
# A = np.mat("[1, 2; 3, 4]")
# print repr(A)
# A = np.matrix("[1, 2; 3, 4]")
# print repr(A)
#
# 2 维 numpy.ndarray
# 虽然 numpy.matrix 有着上面的好处，但是一般不建议使用，而是用 2 维 numpy.ndarray 对象替代，这样可以避免一些不必要的困惑。
# A = np.array([[1,2], [3,4]])
#
# linalg.inv() 可以求一个可逆矩阵的逆：
# 可以使用 linalg.solve 求解方程组
# A = np.array([[1, 3, 5],
#               [2, 5, 1],
#               [2, 3, 8]])
# b = np.array([10, 8, 3])
# 这里A是X的参数，b的Y，此时x = linalg.solve(A, b)就是求X
#
# 可以用 linalg.det 计算行列式：
# A = np.array([[1, 3, 5],
#               [2, 5, 1],
#               [2, 3, 8]])
# print linalg.det(A)
#
# linalg.norm 可以计算向量或者矩阵的模：
# A = np.array([[1, 2],
#               [3, 4]])
# print linalg.norm(A)
# print linalg.norm(A,'fro') # frobenius norm 默认值
# print linalg.norm(A,1) # L1 norm 最大列和
# print linalg.norm(A,-1) # L -1 norm 最小列和
# print linalg.norm(A,np.inf) # L inf norm 最大行和
#
# 问题求解
# 在给定 y 和 A 的情况下，我们可以使用 linalg.lstsq 求解 c。
# 在给定 A 的情况下，我们可以使用 linalg.pinv 或者 linalg.pinv2 求解 A†。
# 求解最小二乘问题：
# c, resid, rank, sigma = linalg.lstsq(A, zi)
# 其中 c 的形状与 zi 一致，为最小二乘解，resid 为 zi - A c 每一列差值的二范数，rank 为矩阵 A 的秩，sigma 为矩阵 A 的奇异值。
# 广义逆
# linalg.pinv 或 linalg.pinv2 可以用来求广义逆，其区别在于前者使用求最小二乘解的算法，后者使用求奇异值的算法求解
#
# 矩阵分解
# 特征值和特征向量
# 问题求解
# linalg.eig(A)
# 返回矩阵的特征值与特征向量
# linalg.eigvals(A)
# 返回矩阵的特征值
# linalg.eig(A, B)
# 求解 Av=λBv 的问题
#
# 奇异值分解
# 问题求解
# U,s,Vh = linalg.svd(A)
# 返回 U 矩阵，奇异值 s，VH 矩阵
# Sig = linalg.diagsvd(s,M,N)
# 从奇异值恢复 Σ 矩阵

# 还有很多具体建议到教程查看


