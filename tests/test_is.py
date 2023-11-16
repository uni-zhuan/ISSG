
import numpy as np
import matplotlib.pyplot as plt

#定义了一个目标概率密度函数P(x)，这个函数是一个混合高斯分布
P = lambda x: 3 * np.exp(-x*x/2) + np.exp(-(x - 4)**2/2)
Z = 10.0261955464

#生成了一个在-10到10之间的等间隔数列x_vals，并计算对应的目标概率密度y_vals。然后绘制了P(x)的图像。
x_vals = np.linspace(-10, 10, 1000)
y_vals = P(x_vals)
plt.figure(1)
plt.plot(x_vals, y_vals, 'r', label='P(x)')
plt.legend(loc='upper right', shadow=True)
plt.show()

#定义了两个函数f(x)和g(x)，分别是x和sin(x)，这两个函数将用于计算期望值。
f_x = lambda x: x
g_x = lambda x: np.sin(x)
true_expected_fx = 10.02686647165
true_expected_gx = -1.15088010640

#定义了一个均匀分布作为建议分布Q(x)，范围是-4到8。然后绘制了P(x)、f(x)、g(x)和Q(x)的图像。
a, b = -4, 8
uniform_prob = 1./(b - a)
plt.figure(2)
plt.plot(x_vals, y_vals, 'r', label='P(x)')
plt.plot(x_vals, f_x(x_vals), 'b', label='x')
plt.plot([-10, a, a, b, b, 10], [0, 0, uniform_prob, uniform_prob, 0, 0], 'g', label='Q(x)')
plt.plot(x_vals, np.sin(x_vals), label='sin(x)')
plt.xlim(-4, 10)
plt.ylim(-1, 3.5)
plt.legend(loc='upper right', shadow=True)
plt.show()

#初始化了两个变量expected_f_x和expected_g_x，用于存储计算得到的期望值E[f(x)]和E[g(x)]。
expected_f_x = 0.
expected_g_x = 0.
n_samples = 100000
den = 0.
for i in range(n_samples):
    sample = np.random.uniform(a, b)
    importance = P(sample) / uniform_prob
    den += importance
    expected_f_x += importance * f_x(sample)
    expected_g_x += importance * g_x(sample)
expected_f_x /= den
expected_g_x /= den
expected_f_x *= Z
expected_g_x *= Z
print('E[f(x)] = %.5f, Error = %.5f' % (expected_f_x, abs(expected_f_x - true_expected_fx)))
print('E[g(x)] = %.5f, Error = %.5f' % (expected_g_x, abs(expected_g_x - true_expected_gx)))