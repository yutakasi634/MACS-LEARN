import unittest
from py_learner import *

class TestSigmoidNetwork(unittest.TestCase):

    def setUp(self):
        print('Constructing Sigmoid Network for test ...\n')
        self.net = SigmoidNetwork(5, 3, 3, 43)
        print('Network property is ')
        self.print_property()
        
        self.input = np.array([x for x in range(5)])
        print('\nInput array is ', self.input)
        
        self.inter_outputs = self.net.forward_propagate(self.input)
        print('\nIntermidiate array is\n', self.inter_outputs)
        
        self.classify_probs = self.net.classify( \
            self.inter_outputs[:, self.net.properties['layers_num'] - 1])
        print('\nClassified_array is \n', self.classify_probs)

    def print_property(self):
        print('nodes num   = ', self.net.properties['nodes_num'])
        print('layers num  = ', self.net.properties['layers_num'])
        print('classes num = ', self.net.properties['classes_num'])
        print('random seed = ', self.net.properties['random_seed'])

        
class TestClassifiedArray(TestSigmoidNetwork):        
    def test_classified_array_sum(self):
        self.assertAlmostEqual(np.sum(self.classify_probs), 1.0, 7)
            
class TestDifferentialInOutput(TestSigmoidNetwork):
    def test_differential_in_output(self):
        answer_node = 2
        differential = self.net.differential_in_output( \
            self.inter_outputs[:, self.net.properties['layers_num'] - 1], \
            self.classify_probs, answer_node)
        print('\nDifferential of output layer is \n', differential)
        
if __name__ == '__main__':

    unittest.main()
    
    print('\nFinish py_learn test!')
