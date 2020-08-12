import numpy as np
import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from PIL import Image
import pickle
import matplotlib.pyplot as plt


def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])
    return network


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y_exp_a = exp_a / sum_exp_a

    return y_exp_a


def identify_function(x):
    return x


def forward(network, x):
    W1 = network['W1']
    W2 = network['W2']
    W3 = network['W3']

    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    return y


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


# network = init_network()
# x = np.array([1.0, 0.5])
# y = forward(network, x)
#
# print(y)
def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False, one_hot_label=False)

    return x_test, t_test


def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)

    return network


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, t.size)
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size


# def cross_entropy_error(y, t):
#     if y.ndim == 1:
#         t = t.reshape(1, t.size)
#         y = y.reshape(1, y.size)
#
#     batch_size = y.shape[0]
#     return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


def function1(x):
    return 0.01 * x ** 2 + 0.1 * x


def function2(x):
    return np.sum(x**2)


# 数值微分
def numerical_diff(f, x):
    h = 1e-4  # 0.0001
    return (f(x+h) - f(x-h)) / (2*h)


# 梯度
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)  # 生成和x形状相同的数组

    for idx in range(x.size):
        tmp_val = x[idx]
        # f(x+h)
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x-h)
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1-fxh2) / (2*h)
        x[idx] = tmp_val

    return grad


# 梯度下降
def gradient_descent(f, init_x, lr=0.1, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)
        print(self.W)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        print(y)
        loss = cross_entropy_error(y, t)

        return loss


if __name__ == '__main__':
    # x, t = get_data()
    # network = init_network()
    # accuracy_cnt = 0
    # batch_size = 1000
    #
    # for i in range(0, len(x), batch_size):
    #     x_batch = x[i:i+batch_size]
    #     y_batch = predict(network, x_batch)
    #     p = np.argmax(y_batch, axis=1)
    #     print(i)
    #     # print("p:", p, "t[i]:", t[i])
    #     # img = x[i].reshape(28, 28)
    #     # img_show(img)
    #     accuracy_cnt += np.sum(p == t[i:i+batch_size])
    #
    # print("Accuracy:", str(float(accuracy_cnt) / len(x)))

    # x = np.arange(0, 20, 0.1)
    # y = function1(x)
    # plt.plot(x, y)
    # plt.xlabel('x')
    # plt.ylabel('f(x)')
    #
    # k = numerical_diff(function1, 10)
    # b = function1(10) - k * 10
    # q = k * x + b
    # plt.plot(x, q, linestyle='--')
    # plt.show()

    net = simpleNet()

    x = np.array([0.6, 0.9])
    p = net.predict(x)
    print(p)
    print(np.argmax(p))  # 最大值的索引

    t = np.array([0, 0, 1])  # 正确标签
    print(net.loss(x, t))


    # print(numerical_gradient(function2, np.array([3.0, 4.0])))
    # print(numerical_gradient(function2, np.array([0.0, 2.0])))
    # print(numerical_gradient(function2, np.array([3.0, 0.0])))
    #
    # init_x = np.array([-3.0, 4.0])
    # print(gradient_descent(function2, init_x, lr=0.1, step_num=100))


# img = x_train[0]
# label = t_train[0]
# print(label)
# print(img.shape)
# img = img.reshape(28, 28)
# print(img.shape)
# img_show(img)