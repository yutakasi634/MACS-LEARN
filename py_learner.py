#!/usr/bin/python
# -*- coding: utf-8 -*-

#
#py_learner.py
#

import numpy as np
import math

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
    def __init__(self, nodes_nums, classes_num, random_seed,\
                 epsilon, momentum = 0):
        # nodes_nums include input layer, middle layer (not classify layer).
        self.nodes_nums = nodes_nums
        self.layers_num = len(nodes_nums)
        self.classes_num = classes_num
        self.random_seed = random_seed
        self.epsilon = epsilon
        self.momentum = momentum
        self.random_generator = np.random.RandomState(self.random_seed)
        self.connections = []
        self.momentum_connections_updated = []
        self.biases = []
        self.momentum_biases_updated = []
        for index in range(self.layers_num - 1):
            self.connection.append(
                self.random_generator \
                .uniform(low=-0.1, high=0.1, \
                         size=(self.nodes_nums[index], \
                               self.nodes_nums[index + 1])) \
                .astype('float32'))

            self.momentum_connections_updated.append(
                np.zeros((self.nodes_num[index], self.nodes_nums[index + 1])) \
                .astype('float32'))

            self.biases.append(
                self.random_generator \
                .uniform(low=-1.0, high=1,0, \
                         size=(self.nodes_num[index])) \
                astype('float32'))

            self.momentum_biases_updated.append(
                np.zeros((self.nodes_num[index])).astype('float32'))
            
        self.classification_connection = \
            self.random_generator.uniform( \
                low=-0.1, high=0.1, size=(self.nodes_nums[-1], \
                                          self.classes_num)).astype('float32')

        self.momentum_classif_connection_updated = \
            np.zeros((self.nodes_num, self.classes_num))
        self.biases = np.zeros((self.nodes_num, \
                                self.layers_num)).astype('float32')

    def learning_step(self, input, answer_node):
        outputs = self.forward_propagate(input)
        classify_probs = self.classify(outputs[:, -1])
        derivs_err_by_connections, derivs_err_by_biases, deriv_by_classification_connection = \
            self.derivs_err_by_connections(outputs, classify_probs, answer_node)
        connections_update, biases_update, classif_con_update = \
            self.calc_update(derivs_err_by_connections, derivs_err_by_biases, \
                             deriv_by_classification_connection)
        self.update_weight(connections_update, biases_update, classif_con_update)
        error = math.log(classify_probs[answer_node])
        return error

    def momentum_learning_step(self, input, answer_node):
        outputs = self.forward_propagate(input)
        classify_probs = self.classify(outputs[:, -1])
        derivs_err_by_connections, derivs_err_by_biases, deriv_by_classification_connection = \
            self.derivs_err_by_connections(outputs, classify_probs, answer_node)
        connections_update, biases_update, classif_con_update = \
            self.calc_momentum_update(derivs_err_by_connections, \
                                      derivs_err_by_biases,\
                                      deriv_by_classification_connection)
        self.update_weight(connections_update, \
                           biases_update, classif_con_update)
        error = math.log(classify_probs[answer_node])
        return error
        
    def answer(self, input):
        classify_probs = self.answer_probs(input)
        random_num = self.random_generator.rand()
        total_prob = 0.
        answer = -1
        for index, prob in enumerate(classify_probs):
            total_prob += prob
            if random_num <= total_prob:
                answer = index
                break
        return answer
        
        
    def answer_probs(self, input):
        input2classify_layer = self.forward_propagate(input)[:,-1]
        classify_probs = self.classify(input2classify_layer)
        return classify_probs
        
    def forward_propagate(self, input):
        # input is numpy array, 1 dim
        assert input.shape[0] == self.nodes_nums[0], 'Invalid dimension input!!'
        outputs = [np.array(input).astype(int)]
        propagated_state = input
        for index, connect in enumerate(self.connections):
            output = np.dot(propagated_state, connect) + self.biases[:, index]
            output = self.sigmoid(output)
            rand_array = self.random_generator.rand(self.nodes_nums[index + 1])
            propagated_state = np.less(rand_array, output).astype(int)
            outputs.append(propagated_state)
        return outputs

    def classify(self, input):
        # input is numpy array
        # return is numpy array
        assert input.shape[0] == self.nodes_nums[-1], 'Invalid dimension input!!'
        classify_probs = self.softmax_func(self.classification_connection, input)
        return classify_probs

    def properties(self):
        return { 'nodes_nums': self.nodes_nums, 'layers_num': self.layers_num, \
                 'classes_num': self.classes_num, 'random_seed': self.random_seed, \
                 'epsilon': self.epsilon }
    
    def differential_in_output(self, inp2classify_layer, classify_probs, answer_node):
        # both inp and classify_probs are numpy array. 1 dim
        # answer_node is scholar
        # differential is 2d matrix, node_num * class_num
        # differential_by_weited_sum is numpy array, class_num elements
        differential_by_weighted_sum = classify_probs.copy()
        differential_by_weighted_sum[answer_node] = \
            differential_by_weighted_sum[answer_node] - 1
        differential = np.dot(np.matrix(inp2classify_layer).T, \
                              np.matrix(differential_by_weighted_sum))
        return differential, differential_by_weighted_sum
    
    def back_propagation(self, output_diff_by_wighted_sum, outputs):
        inp2layers = \
            map(lambda output, connection: np.dot(output, connection),
                outputs[:-1], self.connections)
        deriv_act_by_inps = map(lambda inp2layer: self.deriv_sigmoid(inp2layer),
                               inp2layers)

        # back propagation to last layer to before last layer.
        deriv_inp2output_by_inp2last = \
            np.einsum('ij,i->ij', self.classification_connection ,deriv_act_by_inps[-1])
        deriv_err_by_weighted_sum = \
            np.dot(output_diff_by_wighted_sum, deriv_inp2output_by_inp2last.T)
        derivs_err_by_biases = np.array([deriv_err_by_weighted_sum]).astype('float32')
        deriv_err_by_connection  = np.einsum('i,j->ij', outputs[:, -2], \
                                         deriv_err_by_weighted_sum)
        derivs_err_by_connections = [np.array(deriv_err_by_connection).astype('float32')]

        # back propagation to before last layer to first layer.
        deriv_inps_by_inpses = \
            map(lambda connection, deriv_act_by_inp: einsum('ij,i-ij'), \
                deriv_act_by_inps[:-1])
        deriv_inps_by_inpses = \
            np.einsum('ijk,ji->ijk', self.connections, deriv_act_by_inp)

        for (der_inps_inps, output) in zip(reversed(deriv_inps_by_inpses[:-1]), \
                                           reversed(outputs[:, :-2].T)):
            deriv_err_by_weighted_sum = \
                np.dot(der_inps_inps, deriv_err_by_weighted_sum.T)
            deriv_err_by_connection = \
                np.einsum('i,j->ij', output, deriv_err_by_weighted_sum)
            derivs_err_by_biases = \
                np.append([deriv_err_by_weighted_sum], derivs_err_by_biases, axis=0)
            derivs_err_by_connections = \
                np.append([deriv_err_by_connection], derivs_err_by_connections, axis=0)

        return derivs_err_by_connections, derivs_err_by_biases.T

    def derivs_err_by_connections(self, outputs, classify_probs, answer_node):
        deriv_by_classification_connection, output_deriv_by_weighted_sum = \
            self.differential_in_output(outputs[:,-1], classify_probs, answer_node)
        derivs_err_by_connections, derivs_err_by_biases = \
            self.back_propagation(output_deriv_by_weighted_sum, outputs)
        return derivs_err_by_connections, derivs_err_by_biases, deriv_by_classification_connection

    def update_weight(self, connection_update, biases_update, classif_con_update):
        self.connections -= connection_update
        self.biases -= biases_update
        self.classification_connection -= classif_con_update

    def calc_update(self, deriv_by_connections, deriv_by_biases, \
                    deriv_by_classification_connection):
        return self.epsilon * deriv_by_connections, \
            self.epsilon * deriv_by_biases, \
            self.epsilon * deriv_by_classification_connection

    def calc_momentum_update(self, derivs_by_connections, derivs_by_biases, \
                             derivs_by_classification_connection):
        self.momentum_connections_updated = \
            self.momentum * self.momentum_connections_updated + \
            self.epsilon * derivs_by_connections

        self.momentum_biases_updated = \
            self.momentum * self.momentum_biases_updated + \
            self.epsilon * derivs_by_biases
        
        self.momentum_classif_connection_updated = \
            self.momentum * self.momentum_classif_connection_updated + \
            self.epsilon * derivs_by_classification_connection
        return self.momentum_connections_updated, self.momentum_biases_updated, \
            self.momentum_classif_connection_updated
        
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
