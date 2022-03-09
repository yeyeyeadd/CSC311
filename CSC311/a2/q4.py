# -*- coding: utf-8 -*-
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import scipy
from sklearn.datasets import load_boston
np.random.seed(0)
from sklearn.model_selection import train_test_split

# load boston housing prices dataset
boston = load_boston()
x = boston['data']
N = x.shape[0]
x = np.concatenate((np.ones((506,1)),x),axis=1) #add constant one feature - no bias needed
d = x.shape[1]
y = boston['target']
idx = np.random.permutation(range(N))

#helper function
def l2(A,B):
    '''
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    '''
    A_norm = (A**2).sum(axis=1).reshape(A.shape[0],1)
    B_norm = (B**2).sum(axis=1).reshape(1,B.shape[0])
    dist = A_norm+B_norm-2*A.dot(B.transpose())
    return dist



#to implement
def LRLS(test_datum,x_train,y_train, tau,lam=1e-5):
    '''
    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    '''
    ## TODO
    N, d = x_train.shape
    L2 = l2(test_datum.reshape(1, -1), x_train) # (1, N)
    deno = np.sum(np.exp(-1 * L2 / (2 * tau ** 2)))
    A = np.zeros((N, N))
    np.fill_diagonal(A, np.exp(-1 * L2 / (2 * tau ** 2)) / deno)
    w = np.linalg.solve(x_train.T @ A @ x_train + lam * np.identity(d),
        x_train.T @ A @ y_train)
    return test_datum.T @ w
    ## TODO




def run_validation(x,y,taus,val_frac):
    '''
    Input: x is the N x d design matrix
           y is the N x 1 targets vector
           taus is a vector of tau values to evaluate
           val_frac is the fraction of examples to use as validation data
    output is
           a vector of training losses, one for each tau value
           a vector of validation losses, one for each tau value
    '''
    ## TODO
    M = int(val_frac * N)
    # x, x_val, y, y_val = train_test_split(x, y, test_size=0.3)
    x_val = x[idx, :][: M]
    y_val = y[idx][: M]
    x = x[idx, :][M: ]
    y = y[idx][M: ]
    tr_loss, val_loss = [], []
    for tau in taus:
        loss = 0
        for i in range(x_val.shape[0]):
            y_predict = LRLS(x_val[i], x, y, tau)
            loss += (y_predict - y_val[i]) ** 2
        val_loss.append(loss / x_val.shape[0])
        loss = 0
        for i in range(x_val.shape[0]):
            y_predict = LRLS(x[i], x, y, tau)
            loss += (y_predict - y[i]) ** 2
        tr_loss.append(loss / x.shape[0])
    tr_loss = np.array(tr_loss)
    val_loss = np.array(val_loss)
    return tr_loss, val_loss
    ## TODO


if __name__ == "__main__":
    # In this excersice we fixed lambda (hard coded to 1e-5) and only set tau value.
    # Feel free to play with lambda as well if you wish
    taus = np.logspace(1.0,3.0,200)
    train_losses, test_losses = run_validation(x,y,taus,val_frac=0.3)
    # plt.semilogx(taus, train_losses, label='train')
    plt.semilogx(taus, test_losses, label='validation')
    plt.xlabel('tau')
    plt.ylabel('loss')
    plt.legend()
    # plt.savefig('q4.png')
    plt.show()
