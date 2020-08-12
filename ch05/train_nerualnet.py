from two_layer_net import TwoLayerNet
from dataset.mnist import load_mnist
import numpy as np

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iters_num = 20000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.01
iter_per_epoch = max(train_size / batch_size, 1)

train_loss_list = []
train_acc_list = []
test_acc_list = []

for i in range(iters_num):
    # 随机选取batch_size大小的训练数据
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 梯度
    grads = network.gradient(x_batch, t_batch)
    # 更新
    for key in grads.keys():
        network.params[key] -= learning_rate * grads[key]
    # 损失
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)

        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(i, ": loss:", loss, "train_acc:", train_acc, "test_acc:", test_acc)
