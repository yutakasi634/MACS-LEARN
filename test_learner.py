import unittest
from py_learner import *

class TestSigmoidNetwork(unittest.TestCase):

    def setUp(self):
        self.net = SigmoidNetwork(5, 4, 3, 43)
        self.input = np.array([x for x in range(5)])
        self.outputs = self.net.forward_propagate(self.input)
        self.classify_probs = \
            self.net.classify(self.outputs[:, self.net.properties['layers_num'] - 1])
        
class PrintProperty(TestSigmoidNetwork):
    def test_print_property(self):
        print('Network property is ')
        self.print_property()
        print('\nInput array is ', self.input)
        print('\nOutputs of all layer is \n', self.outputs)
        print('\nClassified_array is \n', self.classify_probs)

    def print_property(self):
        print('nodes num   = ', self.net.properties['nodes_num'])
        print('layers num  = ', self.net.properties['layers_num'])
        print('classes num = ', self.net.properties['classes_num'])
        print('random seed = ', self.net.properties['random_seed'])

        
class TestClassifiedArray(TestSigmoidNetwork):        
    def test_classified_array_sum(self):
        self.assertAlmostEqual(np.sum(self.classify_probs), 1.0, 7)

class TestForwardPropagete(TestSigmoidNetwork):
    def test_dimension_of_outputs(self):
        self.assertEqual(self.outputs.shape, \
                         (self.net.properties['nodes_num'], \
                          self.net.properties['layers_num'] + 1))
        print('\nOutputs probs of forward propagate is \n', self.outputs)
        
class TestDifferentialInOutput(TestSigmoidNetwork):
    def test_differential_in_output(self):
        answer_node = 2
        differential, differential_by_weighted_sum = self.net.differential_in_output( \
            self.outputs[:, self.net.properties['layers_num'] - 1], \
            self.classify_probs, answer_node)
        self.assertEqual(differential.shape, \
            (self.net.properties['nodes_num'], self.net.properties['classes_num']))
        self.assertEqual(differential_by_weighted_sum.shape, \
            (self.net.properties['classes_num'],))
        print('\n answer node is ', answer_node)
        print('Differential of output layer is \n', differential)
        print('\nDifferential of output layer by weited sum is \n', \
            differential_by_weighted_sum)
        
if __name__ == '__main__':

    unittest.main()
    
    print('\nFinish py_learn test!')
