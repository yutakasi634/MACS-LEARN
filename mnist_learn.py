import numpy as np
import random, sys, os

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt

import py_learner

args = sys.argv

session_name = args[1]

color_states = 255.0
figure_size = (28, 28)
layer_num = 2
classes_num = 10
random_seed = 42
epsilon = 0.1

total_learning_step = 10000
total_test_step = 1000

output_dir_name = 'output'

def random_plot(datas, xnum, ynum):
    data_num = datas.shape[0]
    figures = plt.figure(figsize=(xnum, ynum))
    figures.subplots_adjust(left=0, right=1, bottom=0,
                    top=0.5, hspace=0.05, wspace=0.05)

    for i in range(xnum * ynum):
         ax = figures.add_subplot(xnum, ynum, i + 1, xticks=[], yticks=[])
         random_data = datas[random.randint(0, data_num - 1)]
         ax.imshow(random_data.reshape(figure_size[0], figure_size[1]), \
                   cmap='gray')
    plt.show()

if __name__ == '__main__':

    if not os.path.exists(output_dir_name):
        os.mkdir(output_dir_name)
    log_file_name = output_dir_name + '/' + session_name + '_log.dat'
    error_file_name = output_dir_name + '/' + session_name + '_error.dat'
    connections_file_name = output_dir_name + '/' + session_name + '_connections.toml'

    mnist = fetch_mldata('MNIST original')
    mnist_data, mnist_ans = shuffle(mnist.data, mnist.target, random_state=42)
    mnist_data = mnist_data / color_states

    train_data, test_data, train_ans, test_ans = \
        train_test_split(mnist_data, mnist_ans, test_size=0.2, \
                         random_state=42)

    #random_plot(train_data, 4, 4)

    training_data_num = train_data.shape[0]
    test_data_num = test_data.shape[0]

    errorfile = open(error_file_name, 'w')
    connectionfile = open(connections_file_name, 'w')

    network = py_learner.SigmoidNetwork(figure_size[0] * figure_size[1], \
                                        layer_num, classes_num, random_seed, \
                                        epsilon)
    net_properties = network.properties()

    with open(log_file_name, 'w') as logfile:
        logfile.write('Training data num is ' + str(training_data_num))
        logfile.write('\nTraining figure size is ' + str(figure_size))
        logfile.write('\nTotal learning step is ' + str(total_learning_step))
        logfile.write('\nNetwork middle layer num is ' + str(net_properties['layers_num']))
        logfile.write('\nNetwork nodes num is ' + str(net_properties['nodes_num']))
        logfile.write('\nNetwork classes num is ' + str(net_properties['classes_num']))
        logfile.write('\nNetwork random seed is ' + str(net_properties['random_seed']))
        logfile.write('\nNetwork learning epsilon is ' + str(net_properties['epsilon']))
    print('Learning start ...')

    with open(error_file_name, 'w') as errfile:
        for step in range(total_learning_step):
            data_index = random.randint(0, training_data_num - 1)
            error = network.momentum_learning_step(train_data[data_index], int(train_ans[data_index]))
            errfile.write(str(error) + '\n')
            if step % 100 == 0:
                print('Learning step is ', step)

    correct_count = 0
    for (test_inp, answer) in zip(test_data[0:total_test_step], test_ans[0:total_test_step]):
        ans = network.answer(test_inp)
        if ans == int(answer):
            correct_count += 1
    correct_prob = correct_count / total_test_step

    with open(log_file_name, 'a') as logfile:
        logfile.write('\nCorrect probability is ')
        logfile.write(str(correct_prob))

    with open(connections_file_name, 'w') as confile:
        confile.write('[Connections]\n')
        confile.write('intra_connections = [\n')
        for layer in network.connections:
            confile.write('[')
            for sender in layer:
                confile.write('[')
                for receiver in sender:
                    confile.write(str(receiver) + ', ')
                confile.write(']\n')
            confile.write(']\n')
        confile.write('\n]\n')

        confile.write('classification_connection = [\n')
        for sender in network.classification_connection:
            confile.write('[')
            for receiver in sender:
                confile.write(str(receiver) + ', ')
            confile.write(']\n')
        confile.write(']')
