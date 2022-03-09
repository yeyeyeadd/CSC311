'''
Question 4 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''

from numpy.linalg import cond
import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt
from scipy.special import logsumexp

def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''
    means = np.zeros((10, 64))
    # Compute means
    n = train_data.shape[0]
    one_hot = np.zeros((n, 10))
    one_hot[np.arange(n), train_labels.astype(int)] = 1
    means = one_hot.T @ train_data / np.sum(one_hot, axis=0).reshape(-1, 1)
    return means

def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class
    '''
    covariances = np.zeros((10, 64, 64))
    # Compute covariances
    means = compute_mean_mles(train_data, train_labels)
    for i in range(10):
        covariances[i] = (train_data[train_labels == i] - means[i]).T  @ (train_data[train_labels == i] 
        - means[i]) / np.sum(train_labels == i) + 0.01 * np.identity(64)
    return covariances

def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array
    '''
    const = -32 * np.log(2 * np.pi) - 0.5 * np.log(np.linalg.det(covariances))
    p = np.zeros((10, digits.shape[0]))
    for i in range(10):
        p[i] = -0.5 * np.diag((digits - means[i]) @ np.linalg.inv(covariances[i]) @ (digits - means[i]).T)
    return (p + const.reshape(-1, 1)).T

def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    p_y = 0.1 * np.ones((digits.shape[0], 10))
    p_xy = generative_likelihood(digits, means, covariances)
    p_x = logsumexp(p_xy, b=p_y, axis=1).reshape(-1, 1)
    return np.log(p_y) + p_xy - p_x

def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    # Compute as described above and return
    return np.mean(cond_likelihood[np.arange(digits.shape[0]), labels.astype(int)])

def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    # Compute and return the most likely class
    return np.argmax(cond_likelihood, axis=1)

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)

    # Evaluation
    print('the average conditional log-likelihood on training set is {}'.format(
        avg_conditional_likelihood(train_data, train_labels, means, covariances)))
    print('the average conditional log-likelihood on test set is {} \n'.format(
        avg_conditional_likelihood(test_data, test_labels, means, covariances)))
    acc_train = classify_data(train_data, means, covariances) == train_labels
    acc_test = classify_data(test_data, means, covariances) == test_labels
    print('the accuracy on training set is {}'.format(np.mean(acc_train)))
    print('the accuracy on training set is {}'.format(np.mean(acc_test)))
    plt.figure(figsize=(10, 6))
    for i in range(10):
        w, v = np.linalg.eig(covariances[i])
        plt.subplot(2, 5, i + 1)
        plt.title(f'class {i}')
        plt.imshow(v[:, np.argmax(w)].reshape(8, 8), cmap='gray')
    plt.tight_layout()
    plt.savefig('eigenvector.jpg')
    plt.show()

if __name__ == '__main__':
    main()
