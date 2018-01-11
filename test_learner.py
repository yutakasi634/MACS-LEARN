#!/usr/bin/python
#-*- coding: utf-8 -*-

#
#test_learner.py
#

import numpy as np
import unittest
from py_learner import *

class TestSigmoidNetwork(unittest.TestCase):

    def setUp(self):
        self.net = SigmoidNetwork(5, 4, 3, 43, 0.1, momentum=0.9)
        self.input = np.array([x for x in range(5)])
        
class PrintProperty(TestSigmoidNetwork):
    def test_print_property(self):
        print('Network property is ')
        self.print_property()
        print('\nInput array is ', self.input)

    def print_property(self):
        properties = self.net.properties()
        print('nodes num   = ', properties['nodes_num'])
        print('layers num  = ', properties['layers_num'])
        print('classes num = ', properties['classes_num'])
        print('random seed = ', properties['random_seed'])
        print('epsilon     = ', properties['epsilon'])
        print('Initial connection matrix =\n', self.net.connections)
        print('Initial classification connection matrix = \n', \
              self.net.classification_connection)

class TestForwardPropagete(TestSigmoidNetwork):
    def test_dimension_of_outputs(self):
        outputs = self.net.forward_propagate(self.input)

        print('\nOutputs of all layer is \n', outputs)        

        self.assertEqual(outputs.shape, \
                         (self.net.nodes_num, \
                          self.net.layers_num + 1))
        
class TestClassifiedArray(TestSigmoidNetwork):        
    def test_classified_array_sum(self):
        outputs = self.net.forward_propagate(self.input)
        classify_probs = \
            self.net.classify(outputs[:, self.net.layers_num - 1])        

        print('\nClassified_array is \n', classify_probs)   

        self.assertAlmostEqual(np.sum(classify_probs), 1.0, 7)
        for prob in classify_probs:
            self.assertGreaterEqual(prob, 0)
        
class TestDifferentialInOutput(TestSigmoidNetwork):
    def test_differential_in_output(self):
        answer_node = 2
        outputs = self.net.forward_propagate(self.input)
        classify_probs = \
            self.net.classify(outputs[:, self.net.layers_num - 1])
        differential, differential_by_weighted_sum = \
            self.net.differential_in_output( \
                outputs[:, self.net.layers_num - 1], classify_probs, answer_node)

        print('\n answer node is ', answer_node)
        print('Differential of output layer is \n', differential)
        print('\nDifferential of output layer by weited sum is \n', \
            differential_by_weighted_sum)

        self.assertEqual(differential.shape, \
            (self.net.nodes_num, self.net.classes_num))
        self.assertEqual(differential_by_weighted_sum.shape, \
            (self.net.classes_num,))

class TestBackPropagation(TestSigmoidNetwork):
    def test_deriv_err_by_connections_dimension(self):
        answer_node = 2
        outputs = self.net.forward_propagate(self.input)
        classify_probs = \
            self.net.classify(outputs[:, self.net.layers_num - 1])
        differential, differential_by_weighted_sum = self.net.differential_in_output( \
            outputs[:, self.net.layers_num - 1], classify_probs, answer_node)
        deriv_err_by_connections = \
            self.net.back_propagation(differential_by_weighted_sum, outputs)

        print('\n answer node is ', answer_node)
        print('\n Differential of error by connectin is \n', deriv_err_by_connections)
        
        self.assertEqual(deriv_err_by_connections.shape, \
                         (self.net.layers_num, \
                          self.net.nodes_num, \
                          self.net.nodes_num))

class TestLearningStep(TestSigmoidNetwork):
    def test_learning_step(self):
        answer_node = 2
        total_learning_step = 1000
        total_answer_step = 10
        for step in range(total_learning_step):
            self.net.learning_step(self.input, answer_node)
        print('\nTotal learning step is ', total_learning_step)
        print('\nLearned network weight is \n', self.net.connections)
        print('\nLearned classification network weight is \n', \
              self.net.classification_connection)

        prob_mean = np.zeros((self.net.classes_num))
        answer_mean = 0
        for step in range(total_answer_step):
            prob_mean += self.net.answer_probs(self.input)
            answer_mean += self.net.answer(self.input)
        prob_mean /= total_answer_step
        answer_mean /= total_answer_step
        print('\nTotal answer step is ', total_answer_step)
        print('\nMean answer prob is ', prob_mean)
        print('\nMean answer is ', answer_mean)

class TestMomentumLearningStep(TestSigmoidNetwork):
    def test_momentum_learning_step(self):
        answer_node = 2
        total_learning_step = 1000
        total_answer_step = 10
        for step in range(total_learning_step):
            self.net.momentum_learning_step(self.input, answer_node)        

        prob_mean = np.zeros((self.net.classes_num))
        for step in range(total_answer_step):
            prob_mean += self.net.answer_probs(self.input)
        prob_mean /= total_answer_step

        self.assertGreater(prob_mean[answer_node], 0.8)
            
if __name__ == '__main__':

    unittest.main()
    
    print('\nFinish py_learn test!')
