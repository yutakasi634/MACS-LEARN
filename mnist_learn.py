import random

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt

import py_learner

color_states = 255.0
figure_size = (28, 28)
layer_num = 3
classes_num = 10
random_seed = 42
epsilon = 0.1
total_learning_step = 10000

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
    mnist = fetch_mldata('MNIST original')
    mnist_data, mnist_ans = shuffle(mnist.data, mnist.target, random_state=42)
    mnist_data = mnist_data / color_states

    train_data, test_data, train_ans, test_ans = \
        train_test_split(mnist_data, mnist_ans, test_size=0.2, \
                         random_state=42)

    #random_plot(train_data, 4, 4)

    training_data_num = train_data.shape[0]
   
    print('Training data num is ', training_data_num)
    print('Training figure size is ', figure_size)
    print('Total learning step is ', total_learning_step)
    print('Learning start ...')
    
    network = py_learner.SigmoidNetwork(figure_size[0] * figure_size[1], \
                                        layer_num, classes_num, random_seed, \
                                        epsilon)
    for step in range(total_learning_step):
        data_index = random.randint(0, training_data_num - 1)
        # print(train_data[data_index])
        # print(train_ans[data_index])
        # print(train_data[data_index].shape)
        network.learning_step(train_data[data_index], int(train_ans[data_index]))
        if step % 100 == 0:
            print('Learning step is ', step)
