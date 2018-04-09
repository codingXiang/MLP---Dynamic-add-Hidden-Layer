import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class MLP(object):
    def __init__(self, dataset, lr=0.1, batch_size=200, epoch=10, momentum=0.9):
        self.lr = lr
        self.batch_size = batch_size
        self.epoch = epoch
        self.momentum = momentum
        self.dataset = dataset
        self.layer_list = []
        self.weight_list = []
        self.setup()

    def setup(self):
        self.setup_network()
        self.setup_weight()

    def setup_network(self):
        self.x = self.dataset.train_x
        self.Y = self.dataset.train_Y

        self.x_node = self.x.shape[1]
        self.y_node = self.Y.shape[1]

        self.h1_node = 200
        self.h2_node = 100
        self.h3_node = 50
        self.pre_delta_y = 0
        self.pre_delta_h1 = 0
        self.pre_delta_h2 = 0
        self.pre_delta_h3 = 0
        self.pre_delta_y_bias = 0
        self.pre_delta_h1_bias = 0
        self.pre_delta_h2_bias = 0
        self.pre_delta_h3_bias = 0

    def setup_weight(self):
        self.w1 = np.random.uniform(-1.0, 1.0, size=self.h1_node * (self.x_node + 1))
        self.w1 = self.w1.reshape(self.x_node + 1, self.h1_node)
        self.w2 = np.random.uniform(-1.0, 1.0, size=self.h2_node * (self.h1_node + 1))
        self.w2 = self.w2.reshape(self.h1_node + 1, self.h2_node)
        self.w3 = np.random.uniform(-1.0, 1.0, size=self.h3_node * (self.h2_node + 1))
        self.w3 = self.w3.reshape(self.h2_node + 1, self.h3_node)
        self.w4 = np.random.uniform(-1.0, 1.0, size=self.y_node * (self.h3_node + 1))
        self.w4 = self.w4.reshape((self.h3_node + 1, self.y_node))

    def forward(self, x):
        self.h1 = self.sigmoid((np.dot(x, self.w1[1:]) + self.w1[0]))
        self.h2 = self.sigmoid((np.dot(self.h1, self.w2[1:]) + self.w2[0]))
        self.h3 = self.sigmoid((np.dot(self.h2, self.w3[1:]) + self.w3[0]))
        self.y = self.sigmoid((np.dot(self.h3, self.w4[1:]) + self.w4[0]))

    def predict(self, x, Y):
        self.forward(x)
        self.accuracy = 0
        for i in range(0, Y.shape[0]):
            zy = np.argmax(self.y[i])
            ty = np.argmax(Y[i])
            if (zy == ty):
                self.accuracy = self.accuracy + 1
        self.accuracy = self.accuracy / Y.shape[0] * 100
        return self

    def backend(self, x, Y):
        E = (Y - self.y)
        self.mse = E.sum() / Y.shape[0]
        delta_y = E * self.sigmoid_gradient(self.y)
        delta_h3 = self.sigmoid_gradient(self.h3) * np.dot(delta_y, self.w4[1:].T)
        delta_h2 = self.sigmoid_gradient(self.h2) * np.dot(delta_h3, self.w3[1:].T)
        delta_h1 = self.sigmoid_gradient(self.h1) * np.dot(delta_h2, self.w2[1:].T)

        self.w4[1:] += self.lr * self.h3.T.dot(delta_y) + (self.pre_delta_y * self.momentum)
        self.w3[1:] += self.lr * self.h2.T.dot(delta_h3) + (self.pre_delta_h3 * self.momentum)
        self.w2[1:] += self.lr * self.h1.T.dot(delta_h2) + (self.pre_delta_h2 * self.momentum)
        self.w1[1:] += self.lr * x.T.dot(delta_h1) + (self.pre_delta_h1 * self.momentum)

        self.w4[0] += self.lr * delta_y.sum() + (self.momentum * self.pre_delta_y_bias)
        self.w3[0] += self.lr * delta_h3.sum() + (self.momentum * self.pre_delta_h3_bias)
        self.w2[0] += self.lr * delta_h2.sum() + (self.momentum * self.pre_delta_h2_bias)
        self.w1[0] += self.lr * delta_h1.sum() + (self.momentum * self.pre_delta_h1_bias)

        self.pre_delta_y = self.lr * self.h3.T.dot(delta_y)
        self.pre_delta_h3 = self.lr * self.h2.T.dot(delta_h3)
        self.pre_delta_h2 = self.lr * self.h1.T.dot(delta_h2)
        self.pre_delta_h1 = self.lr * x.T.dot(delta_h1)

        self.pre_delta_y_bias = self.lr * delta_y.sum()
        self.pre_delta_h3_bias = self.lr * delta_h3.sum()
        self.pre_delta_h2_bias = self.lr * delta_h2.sum()
        self.pre_delta_h1_bias = self.lr * delta_h1.sum()
        return E

    def fit(self):
        self.accuracy_list = []
        for _iter in range(0, self.epoch):
            for i in range(0, self.Y.shape[0], self.batch_size):
                x = self.x[i: i + self.batch_size]
                Y = self.Y[i: i + self.batch_size]
                self.forward(x)
                self.backend(x, Y)
            self.predict(self.x, self.Y)
            self.accuracy_list.append(self.accuracy)
            if (_iter % 5 == 0):
                print("epoch = {} , accuracy = {:.2f}%".format(_iter, self.accuracy))
            if (self.accuracy >= 98.0):
                print("epoch = {} , accuracy = {:.2f}%".format(_iter, self.accuracy))
                break

    def data_std(self, data):
        return (data - data.mean()) / data.std()

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_gradient(self, x):
        return x * (1 - x)