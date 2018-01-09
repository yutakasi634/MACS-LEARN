#!/usr/bin/python
# -*- coding: utf-8 -*-

#
#py_learner.py
#

import numpy as np

class SigmoidNetwork:
    '''Neural network which have Sigmoid function as activation function.'''

    # input layer is not network layer. input connection is first connection matrix.
    # connections matrix's first matrix is input to first layer of network.
    # * - * - * - * - * 
    #   \   \   /   /   \
    # * - * - * - * - * - *
    #   /   /   /   \   /
    # * - * - * - * - * - *
    #   \   \   \   /   / ^
    # * - * - * - * - *   |
    # ^  ^   ^   ^   ^   last layer is output layer
    # |  |   |   |   |
    # |  (   layers  )
    # input
    def __init__(self, nodes_num, layers_num, classes_num, random_seed, epsilon):
        self.nodes_num = nodes_num
        self.layers_num = layers_num
        self.classes_num = classes_num
        self.random_seed = random_seed
        self.epsilon = epsilon
        self.random_generator = np.random.RandomState(self.random_seed)
        self.connections = \
            self.random_generator.uniform( \
                low=-0.1, high=0.1, size=(self.layers_num, \
                                          self.nodes_num, \
                                          self.nodes_num)) \
                                 .astype('float32')
        self.classification_connection = \
            self.random_generator.uniform( \
                low=-0.1, high=0.1, size=(self.nodes_num, \
                                          self.classes_num)).astype('float32')
        self.biases = np.zeros((self.nodes_num, \
                                self.layers_num)).astype('float32')

    def learning_step(self, input, answer_node):
        epsilon = self.epsilon
        outputs = self.forward_propagate(input)
        classify_probs = self.classify(outputs[:,-1])
        deriv_by_classification_connection, output_deriv_by_weighted_sum = \
            self.differential_in_output(outputs[:,-1], classify_probs, answer_node)
        derivs_err_by_connections = \
            self.back_propagation(output_deriv_by_weighted_sum, outputs)
        self.connections -= self.epsilon * derivs_err_by_connections
        self.classification_connection -= \
            self.epsilon * deriv_by_classification_connection

    def answer(self, input):
        input2classify_layer = self.forward_propagate(input)[:,-1]
        classify_probs = self.classify(input2classify_layer)
        return classify_probs
        
    def forward_propagate(self, input):
        # input is numpy array, 1 dim
        assert input.shape == (self.nodes_num,), 'Invalid dimension input!!'
        outputs = np.array([input]).astype('float32')
        propagated_state = input
        for connect in self.connections:
            output = np.dot(propagated_state, connect)
            output = self.sigmoid(output)
            rand_array = self.random_generator.rand(self.nodes_num)
            propagated_state = np.less(rand_array, output).astype(int)
            outputs = np.append(outputs, [propagated_state], axis=0)
        return outputs.T

    def classify(self, input):
        # input is numpy array
        # return is numpy array
        assert input.shape[0] == self.nodes_num, 'Invalid dimension input!!'
        classify_probs = self.softmax_func(self.classification_connection, input)
        return classify_probs

    def properties(self):
        return { 'nodes_num': self.nodes_num, 'layers_num': self.layers_num, \
                 'classes_num': self.classes_num, 'random_seed': self.random_seed, \
                 'epsilon': self.epsilon }
    
    def differential_in_output(self, inp2classify_layer, classify_probs, answer_node):
        # both inp and classify_probs are numpy array. 1 dim
        # answer_node is scholar
        # differential is 2d matrix, node_num * class_num
        # differential_by_weited_sum is numpy array, class_num elements
        differential_by_weighted_sum = classify_probs
        differential_by_weighted_sum[answer_node] -=  1
        differential = np.dot(np.matrix(inp2classify_layer).T, \
                              np.matrix(differential_by_weighted_sum))
        return differential, differential_by_weighted_sum
    
    def back_propagation(self, output_diff_by_wighted_sum, outputs):
        inp2layer = \
            np.einsum('ij,jik->kj', outputs[:,:-1], self.connections)
        deriv_act_by_inp = self.deriv_sigmoid(inp2layer)

        # back propagation to last layer to before last layer.
        deriv_inp2output_by_inp2last = \
            np.einsum('ij,i->ij', self.classification_connection ,deriv_act_by_inp[:, -1])
        deriv_err_by_weighted_sum = \
            np.dot(output_diff_by_wighted_sum, deriv_inp2output_by_inp2last.T)
        deriv_err_by_connection  = np.einsum('i,j->ij', outputs[:, -2], \
                                         deriv_err_by_weighted_sum)
        derivs_err_by_connections = np.array([deriv_err_by_connection]).astype('float32')
        # back propagation to last layer to first layer.
        deriv_inps_by_inpses = \
            np.einsum('ijk,ji->ijk', self.connections, deriv_act_by_inp)

        for (der_inps_inps, output) in zip(reversed(deriv_inps_by_inpses[:-1]), \
                                           reversed(outputs[:, :-2].T)):
            deriv_err_by_weighted_sum = \
                np.dot(der_inps_inps, deriv_err_by_weighted_sum.T)
            deriv_err_by_connection = \
                np.einsum('i,j->ij', output, deriv_err_by_weighted_sum)
            derivs_err_by_connections = \
                np.append([deriv_err_by_connection], derivs_err_by_connections, axis=0)

        return derivs_err_by_connections
    
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def deriv_sigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))    

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
