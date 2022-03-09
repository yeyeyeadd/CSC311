from check_grad import check_grad
from utils import *
from logistic import *

import matplotlib.pyplot as plt
import numpy as np


def run_logistic_regression():
    # train_inputs, train_targets = load_train()
    train_inputs, train_targets = load_train_small()
    valid_inputs, valid_targets = load_valid()
    test_inputs, test_targets = load_test()

    N, M = train_inputs.shape

    #####################################################################
    # TODO:                                                             #
    # Set the hyperparameters for the learning rate, the number         #
    # of iterations, and the way in which you initialize the weights.   #
    #####################################################################
    hyperparameters = {
        "learning_rate": 2,
        "weight_regularization": 0.,
        "num_iterations": 30
    }
    weights = None
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    # Verify that your logistic function produces the right gradient.
    # diff should be very close to 0.
    run_check_grad(hyperparameters)

    # Begin learning with gradient descent
    #####################################################################
    # TODO:                                                             #
    # Modify this section to perform gradient descent, create plots,    #
    # and compute test error.                                           #
    #####################################################################
    weights = np.zeros((M + 1, 1))
    loss_tr, loss_val = [], []
    for t in range(hyperparameters["num_iterations"]):
        f, df, y = logistic(weights, train_inputs, train_targets, hyperparameters)
        weights -= hyperparameters['learning_rate'] * df
        loss_tr.append(f)
        loss_val.append(evaluate(valid_targets,
            logistic_predict(weights, valid_inputs))[0])
    print(hyperparameters)
    targets, inputs = train_targets, train_inputs
    ce, frac_correct = evaluate(targets, logistic_predict(weights, inputs))
    print("On training set, the classification error is {:.2f} with loss {:.2f}".
        format(1 - frac_correct, ce))
    targets, inputs = valid_targets, valid_inputs
    ce, frac_correct = evaluate(targets, logistic_predict(weights, inputs))
    print("On validation set, the classification error is {:.2f} with loss {:.2f}".
        format(1 - frac_correct, ce))
    targets, inputs = test_targets, test_inputs
    ce, frac_correct = evaluate(targets, logistic_predict(weights, inputs))
    print("On test set, the classification error is {:.2f} with loss {:.2f}".
        format(1 - frac_correct, ce))
    plt.plot(range(hyperparameters["num_iterations"]), loss_tr, label='train')
    plt.plot(range(hyperparameters["num_iterations"]), loss_val, label='val')
    plt.xlabel('num_iterations')
    plt.ylabel('cross entropy loss')
    plt.legend()
    # plt.savefig('q3_2c_2.png')
    plt.show()
    # test
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def run_check_grad(hyperparameters):
    """ Performs gradient check on logistic function.
    :return: None
    """
    # This creates small random data with 20 examples and
    # 10 dimensions and checks the gradient on that data.
    num_examples = 20
    num_dimensions = 10

    weights = np.random.randn(num_dimensions + 1, 1)
    data = np.random.randn(num_examples, num_dimensions)
    targets = np.random.rand(num_examples, 1)

    diff = check_grad(logistic,
                      weights,
                      0.001,
                      data,
                      targets,
                      hyperparameters)

    print("diff =", diff)


if __name__ == "__main__":
    run_logistic_regression()
