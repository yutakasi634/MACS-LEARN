#!/usr/bin/python
# -*- coding: utf-8 -*-

#
#py_learner.py
#

from sklearn.utils import shuffle
import numpy as np

def multiclass_cross_entropy(true_class, connections, inputs):    
    np.log(softmax_func(connections, inputs))
    return 0

class SigmoidNetwork:
    '''Neural network which have Sigmoid function as activation function.'''

    # * - * - * - * 
    #   \   /   /   \
    # * - * - * - * - *
    #   /   /   \   /
    # * - * - * - * - *
    #   \   \   /   / ^
    # * - * - * - *   |
    # ^   ^   ^   ^   last layer is output layer
    # |   |   |   |
    #     layers
    
    def __init__(self, nodes_num, layers_num, classes_num, random_seed):
        self.properties = {'nodes_num': nodes_num, 'layers_num': layers_num, \
                           'classes_num': classes_num, 'random_seed': random_seed}
        self.random_generator = np.random.RandomState(self.properties['random_seed'])
        self.connections = \
            self.random_generator.uniform( \
                low=-0.1, high=0.1, size=(self.properties['layers_num'], \
                                          self.properties['nodes_num'], \
                                          self.properties['nodes_num'])) \
                                 .astype('float32')
        self.classification_connection = \
            self.random_generator.uniform( \
                low=-0.1, high=0.1, size=(self.properties['nodes_num'], \
                                          self.properties['classes_num'])).astype('float32')
        self.biases = np.zeros((self.properties['nodes_num'], \
                                self.properties['layers_num'])).astype('float32')
        
    def forward_propagate(self, input):
        # input is numpy array, 1 dim
        nodes_num = self.properties['nodes_num']
        assert input.shape == (nodes_num,), 'Invalid dimension input!!'
        outputs = np.empty((nodes_num, 0)).astype('float32')
        propagated_state = input
        for layer in range(self.properties['layers_num']):
            output = np.dot(propagated_state, self.connections[layer])
            output = self.sigmoid(output)
            rand_array = self.random_generator.rand(nodes_num)
            propagated_state = np.less(rand_array, output).astype(int)
            outputs = np.append(outputs, propagated_state.reshape(1,-1).T, axis=1)
        return outputs

    def classify(self, input):
        # input is numpy array
        # return is numpy array
        assert input.shape[0] == self.properties['nodes_num'], 'Invalid dimension input!!'
        classify_probs = self.softmax_func(self.classification_connection, input)
        return classify_probs
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def deriv_sigmoid(self, x):
        return sigmoid(x) * (1 - sigmoid(x))    

    def softmax_func(self, connections, input_array):
        #connections is wMK = {w1K, w2K, ......, wmK} m is cluster num, 2 dim, line vector
        #wmK = {wm1, wm2,... wmk} k is node num, 1 dim, column vector
        #inputs is xK = {x1, x2, ......., xk} k is node num, column vector
        #return is {exp(w1KTxK) / sum(exp(w2KTxK)), ....., exp(wmKTxK) / sum(exp(wMKTxK))}
        inputs = np.matmul(input_array, connections) #w1KTxK, w2KTxK, ...wmKTxK
        return np.exp(inputs) / self.softmax_dist_func(inputs)  #p(cluster1), p(cluster2)...p(clusterm), column vector

    def softmax_dist_func(self, input_array):
        #denominator of softmax function
        #input_array is {x1, x2, ........, xn}, column vector
        #return is {sum(exp(x1)), sum(exp(x2)), ...... ,sum(exp(xn))}, scholar
        return np.sum(np.exp(input_array), axis=0)

    def differential_in_output(self, inp, classify_probs, answer_node):
        # both inp and classify_probs are numpy array. 1 dim
        # answer_node is scholar
        # differential is 2d matrix, node_num * class_num
        # differential_by_weited_sum is numpy array, class_num elements
        differential_by_weighted_sum = classify_probs
        differential_by_weighted_sum[answer_node] -=  1
        differential = np.dot(np.matrix(inp).T, np.matrix(differential_by_weighted_sum))
        return differential, differential_by_weighted_sum

    def back_propagation(self, output_diff_by_wighted_sum, outputs):
        input2activation_func = np.einsum('ij,ijk->ki',outputs.T, self.connections) #node * layer
        deriv_activation_func = self.deriv_sigmoid(input2activation_func)
        # del u_k^{l+1} / del u_j^{l}
        deriv_inp2act_by_inp2act = \
            np.einsum('ijk,ji->ijk', self.connections, deriv_activation_func)
        #TODO multiply upper layer's deriv of error and sum
        
