
import random
import numpy as np



def compute_mse(y, tx, w):
    """Calculate the loss using either MSE.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """ 
    pred = np.dot(tx, w)
    return (1/(2*len(tx))) *  np.sum((y - pred)** 2)



def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of GD
    """

    weights = initial_w
    loss = compute_mse(y, tx, weights)
    
    for n_iter in range(max_iters):

        error = y -  tx.dot(weights)
        gradient = -1/len(tx) * np.dot(tx.T, error)  # compute gradient
        loss = compute_mse(y, tx, weights)
        
        weights = weights - gamma*gradient  # update weights

    return loss, weights




def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD).

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of SGD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of SGD
    """

    weights = initial_w
    loss = compute_mse(y, tx, weights)

    for n_iter in range(max_iters):
        n = random.randint(0,len(tx))
        gradient = tx[n] * (y[n] - tx[n] @ weights) # compute gradient
        loss = compute_mse(y, tx, weights)
        
        weights = weights - gamma*gradient # update weights


    return weights, loss



def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.

    """

    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    mse = compute_mse(y, tx, w)
    
    return w, mse


def ridge_regression(y, tx, lambda_):
    """implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.

    """

    N, D = tx.shape
    w = np.linalg.solve(np.dot(tx.T, tx)  + np.identity(D) * (lambda_*(2*N)),  np.dot(tx.T, y))
    mse = compute_mse(y, tx, w)
    
    return w, mse


def sigmoid(t):
    """Apply the logistic function."""
    return 1.0 / (1 + np.exp(-t))


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Perform logistic regression using gradient descent.
    
    Parameters:
    y: np.array
        The target values
    tx: np.array
        The data matrix (each row is a data point)
    initial_w: np.array
        Initial weights
    max_iters: int
        Maximum number of iterations for gradient descent
    gamma: float
        Learning rate

    Returns:
    w: np.array
        Optimized weights after training
    """
    
    w = initial_w
    pred = sigmoid(tx.dot(w))
    loss = (1/(2*len(tx))) *  np.sum((y - pred)** 2)
    
    for iter in range(max_iters):
        # compute the gradient
        grad =  tx.T.dot(pred - y)

        # update w through the negative gradient direction
        w = w - gamma * grad

        pred = sigmoid(tx.dot(w))
        loss = (1/(2*len(tx))) *  np.sum((y - pred)** 2)

        
    return w, loss



def reg_logistic_regression(y, tx, initial_w, gamma, max_iters, lambda_):
    """
    Perform regularized logistic regression using gradient descent.
    
    Parameters:
    y: np.array
        The target values
    tx: np.array
        The data matrix (each row is a data point)
    initial_w: np.array
        Initial weights
    max_iters: int
        Maximum number of iterations for gradient descent
    gamma: float
        Learning rate
    lambda_: float
        Regularization strength for L2 penalty

    Returns:
    w: np.array
        Optimized weights after training
    """
    
    w = initial_w

    pred = sigmoid(tx.dot(w))  # initial predictions
    loss = (1/(2*len(tx))) *  np.sum((y - pred)** 2)

    
    for iter in range(max_iters):

        # compute the gradient
        grad = tx.T.dot(pred - y) + 2 * lambda_ * w

        # update w through the negative gradient direction
        w = w - gamma * grad

        pred = np.dot(tx, w)
        loss = (1/(2*len(tx))) *  np.sum((y - pred)** 2)
        
    return w, loss

