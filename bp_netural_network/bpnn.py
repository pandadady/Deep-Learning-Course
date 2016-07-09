#!/usr/bin/python
# -*- coding:  utf-8 -*
"""
This module is used to learn bp netual network algorithm
Authors: shiduo(shiduo@baidu.com)
Date:2016-7-1
"""
# Written in Python.  See http://www.python.org/
# Placed in the public domain.
# Neil Schemenauer <nas@arctrix.com>

import math
from numpy import *
random.seed(0)



def sigmoid(x):
    """
    sigmoid function，1/(1+e^-x)
    :param x:
    :return:
    """
    return 1.0 / (1.0 + math.exp(-x))


def dsigmoid(y):
    """
    sigmoid function derivatives
    :param y:
    :return:
    """
    return y * (1 - y)


class NN:
    def __init__(self, ni, nh, no):
        # number of input, hidden, and output nodes
        """
        构造神经网络
        :param ni:输入单元数量
        :param nh:隐藏单元数量
        :param no:输出单元数量
        """
        self.ni = ni + 1  # +1 是为了偏置节点
        self.nh = nh
        self.no = no

        # 激活值（输出值）
        self.ai = [1.0] * self.ni
        self.ah = [1.0] * self.nh
        self.ao = [1.0] * self.no

        # 权重矩阵
        self.wi = random.uniform(-0.2,0.2,(self.ni, self.nh))  # 输入层到隐藏层
        self.wo = random.uniform(-5.0,5.0,(self.nh, self.no))  # 隐藏层到输出层
        # 权重矩阵的上次梯度
        self.ci = zeros((self.ni, self.nh))
        self.co = zeros((self.nh, self.no))
    def runNN(self, inputs):
        """
        前向传播进行分类
        :param inputs:输入
        :return:类别
        """
        if len(inputs) != self.ni - 1:
            print 'incorrect number of inputs'

        for i in range(self.ni - 1):
            self.ai[i] = inputs[i]

        for j in range(self.nh):
            sum = 0.0
            for i in range(self.ni):
                sum += ( self.ai[i] * self.wi[i][j] )
            self.ah[j] = sigmoid(sum)

        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum += ( self.ah[j] * self.wo[j][k] )
            self.ao[k] = sigmoid(sum)

        return self.ao


    def backPropagate(self, targets, N, M):
        """
        后向传播算法
        :param targets: 实例的类别
        :param N: 本次学习率
        :param M: 上次学习率
        :return: 最终的误差平方和的一半
        """

        # 计算输出层 deltas
        # dE/dw[j][k] = (t[k] - ao[k]) * s'( SUM( w[j][k]*ah[j] ) ) * ah[j]
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = targets[k] - self.ao[k]
            output_deltas[k] = error * dsigmoid(self.ao[k])

        # 更新输出层权值
        for j in range(self.nh):
            for k in range(self.no):
                # output_deltas[k] * self.ah[j] 才是 dError/dweight[j][k]
                change = output_deltas[k] * self.ah[j]
                self.wo[j][k] += N * change + M * self.co[j][k]
                self.co[j][k] = change

        # 计算隐藏层 deltas
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error += output_deltas[k] * self.wo[j][k]
            hidden_deltas[j] = error * dsigmoid(self.ah[j])

        # 更新输入层权值
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j] * self.ai[i]
                # print 'activation',self.ai[i],'synapse',i,j,'change',change
                self.wi[i][j] += N * change + M * self.ci[i][j]
                self.ci[i][j] = change

        # 计算误差平方和
        # 1/2 是为了好看，**2 是平方
        error = 0.0
        for k in range(len(targets)):
            error = 0.5 * (targets[k] - self.ao[k]) ** 2
        return error


    def weights(self):
        """
        打印权值矩阵
        """
        print 'Input weights:'
        for i in range(self.ni):
            print self.wi[i]
        print
        print 'Output weights:'
        for j in range(self.nh):
            print self.wo[j]
        print ''

    def test(self, patterns):
        """
        测试
        :param patterns:测试数据
        """
        for p in patterns:
            inputs = p[0]
            print 'Inputs:', p[0], '-->', self.runNN(inputs), '\tTarget', p[1]

    def train(self, patterns, max_iterations=10000, N=0.8, M=0.1):
        """
        训练
        :param patterns:训练集
        :param max_iterations:最大迭代次数
        :param N:本次学习率
        :param M:上次学习率
        """
        for i in range(max_iterations):
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.runNN(inputs)
                error = self.backPropagate(targets, N, M)
            if i % 50 == 0:
                print 'Combined error', error
        


def main():
    pat = [
        [[0, 0], [1]],
        [[0, 1], [1]],
        [[1, 0], [1]],
        [[1, 2], [1]],
        [[3, 3], [1]],
        [[5, 1], [0]],
        [[1, 7], [0]],
        [[9, 1], [0]],
        [[0, 8], [0]]
    ]
    pat1 = [
        [[5, 7], [0]],
    ]
    myNN = NN(2, 2, 1)
#     myNN.weights()
    myNN.train(pat)
    myNN.test(pat1)


if __name__ == "__main__":
    main()
