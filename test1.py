import numpy as np

def generate_data(n_sample=100, true_beta=[2.0, 9.0, 0.5]):
    """
    n: 样本个数
    生成y=e^(2x_1)+9x_2^2+0.5x_3+epsilon样本数据
    其中x_1~N(1, 1), x_2~N(0, 1), x_3~N(0, 1), epsilon~N(0, 0.5^2)
    """
    x1 = np.random.normal(loc=1, scale=1, size=(n_sample, 1))
    x2 = np.random.normal(loc=0, scale=1, size=(n_sample, 1))
    x3 = np.random.normal(loc=0, scale=1, size=(n_sample, 1))
    epsilon = np.random.normal(loc=0, scale=0.5, size=(n_sample, 1))
    y = np.exp(true_beta[0] * x1) + true_beta[1] * x2 * x2 + true_beta[2] * x3 + epsilon
    # beta = np.array([[2.0], [9.0], [0.5]])
    # y = comput_y(beta) + epsilon
    return x1, x2, x3, y

x1, x2, x3, y = generate_data()
y = np.log(y)

beta = np.random.uniform(low=0, high=1, size=(3, 1))
y_hat = beta[0] * x1 + 2*np.log(x2) + np.log(x3) + np.log(beta[1]*beta[2])
x2[3]























