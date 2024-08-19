from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml

import numpy as np
import pandas as pd
from lib.functions import np_log, relu, deriv_relu, deriv_softmax, softmax
from lib.model import Model
from lib.datahandler import create_batch

np.random.seed(34)

# MNISTデータの取得
digits = fetch_openml(name='mnist_784', version=1)
x_mnist = np.array(digits.data)
t_mnist = np.array(digits.target)

x_mnist = x_mnist.astype("float64") / 255.  # 値を[0, 1]に正規化する
t_mnist = np.eye(N=10)[t_mnist.astype("int32").flatten()]  # one-hotベクトルにする

x_mnist = x_mnist.reshape(x_mnist.shape[0], -1)  # 1次元に変換

# train data: 5000, valid data: 10000 test data: 10000にする
x_train_mnist, x_test_mnist, t_train_mnist, t_test_mnist =\
    train_test_split(x_mnist, t_mnist, test_size=10000)
x_train_mnist, x_valid_mnist, t_train_mnist, t_valid_mnist =\
    train_test_split(x_train_mnist, t_train_mnist, test_size=10000)

def train(model, x, t, eps=0.01):
    y = model(x)
    delta = y - t
    model.backward(delta)
    model.update(eps)
    cost = (-t * np_log(y)).sum(axis=1).mean()
    return cost

def valid(model, x, t):
    y = model(x)
    cost = (-t * np_log(y)).sum(axis=1).mean()
    return cost, y


model = Model(
    hidden_dims=[784, 100, 100, 10], 
    activation_functions=[relu, relu, softmax], 
    deriv_functions=[deriv_relu, deriv_relu, deriv_softmax])


batch_size = 128
epoch = 50
lr =0.1

Accuracy_array = np.array([])
Cost_array = np.array([])

for i in range(epoch):
    x_train_mnist, t_train_mnist = shuffle(x_train_mnist, t_train_mnist)
    x_train_batch, t_train_batch = \
        create_batch(x_train_mnist, batch_size), create_batch(t_train_mnist, batch_size)
    
    for x, t in zip(x_train_batch, t_train_batch):
        cost = train(model, x, t, eps=lr)
    cost, y_pred = valid(model, x_valid_mnist, t_valid_mnist)
    accuracy = accuracy_score(t_valid_mnist.argmax(axis=1), y_pred.argmax(axis=1))

    print(f"#####EPOCH: {i+1}")
    print(f"BP Valid[Cost: {cost:.3f}, Accuracy: {accuracy:.3f}]")
    Accuracy_array = np.append(Accuracy_array, accuracy)
    Cost_array = np.append(Cost_array, cost)

#data = pd.Series(data=Accuracy_array, index=x, name='Accuracy_BP')
x = range(epoch)
Accuracy_array = pd.Series(data=Accuracy_array, index=x, name='Accuracy_BP')
Cost_array = pd.Series(data=Cost_array, index=x, name='Cost_BP')

data = pd.concat([Accuracy_array , Cost_array], axis=1)

data.to_csv('./data/data_mlp.csv')