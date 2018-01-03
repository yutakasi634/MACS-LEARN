#!/usr/bin/python
# -*- coding: utf-8 -*-

#
#py_learner.py
#

from sklearn.utils import shuffle
import numpy as np

def softmax_dist_func(input_array):
    #denominator of softmax function
    #input_array is {x1, x2, ........, xn}
    #return is {sum(exp(x1)), sum(exp(x2)), ...... ,sum(exp(xn))}
    return np.sum(np.exp(input_array), axis=0)

def softmax_func(connections, input_array):
    #connections is wMK = {w1K, w2K, ......, wmK} m is cluster num, 2 dim
    #wmK = {wm1, wm2,... wmk} k is node num, 1 dim
    #inputs is xK = {x1, x2, ......., xk} k is node num
    #return is {exp(w1KTxK) / sum(exp(w2KTxK)), ....., exp(wmKTxK) / sum(exp(wMKTxK))}
    inputs = np.matual(connections.T, input_array) #w1KTxK, w2KTxK, ...wmKTxK
    return np.exp(inputs) / softmax_dist_func(inputs)  #p(cluster1), p(cluster2)...p(clusterm)

def multiclass_cross_entropy(true_class, connections, inputs):    
    np.log(softmax_func(connections, inputs))
    return 0

class SigmoidNetwork:
    '''Neural network which have Sigmoid function as activation function.'''

    def __init__(self, node_num, layer_num):
        self.connections = np.rondom.uniform(low=-0.1, high=0.1, size(node_num, layer_num)).astype('float32')

    def foward_propagate(input)
        
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    def deriv_sigmoid(x):
        return sigmoid(x) * (1 - sigmoid(x))    

    
if __name__ == '__main__':

    numpy.random.seed(43)
    
    print('py_learn')
