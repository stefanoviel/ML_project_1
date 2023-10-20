import random
import numpy as np
from implementations import *

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]



def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8
    you will have 80% of your data set dedicated to training
    and the rest dedicated to testing. If ratio times the number of samples is not round
    you can use np.floor. Also check the documentation for np.random.permutation,
    it could be useful.

    Args:
        x: numpy array of shape (N,), N is the number of samples.
        y: numpy array of shape (N,).
        ratio: scalar in [0,1]
        seed: integer.

    Returns:
        x_tr: numpy array containing the train data.
        x_te: numpy array containing the test data.
        y_tr: numpy array containing the train labels.
        y_te: numpy array containing the test labels.

    >>> split_data(np.arange(13), np.arange(13), 0.8, 1)
    (array([ 2,  3,  4, 10,  1,  6,  0,  7, 12,  9]), array([ 8, 11,  5]), array([ 2,  3,  4, 10,  1,  6,  0,  7, 12,  9]), array([ 8, 11,  5]))
    """
    # set seed
    np.random.seed(seed)
    
    x_shuffle, y_shuffle = unison_shuffled_copies(x, y)
    split_pos = round(len(x_shuffle)*ratio)
    
    return x_shuffle[: split_pos], x_shuffle[split_pos:], y_shuffle[:split_pos], y_shuffle[split_pos:],


def predict_logistic(X, w): 
    return (sigmoid(X @ w) >= 0.5).flatten()

def normalize(data):
    """
    Normalize the input data to have mean 0 and standard deviation 1.

    Parameters:
    - data: numpy array of shape (m, n) where m is the number of samples and n is the number of features.

    Returns:
    - normalized_data: numpy array of shape (m, n) with normalized values.
    """
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    normalized_data = (data - mean) / (std + 10e-300)

    return normalized_data



def k_fold_cross_validation(X, y, model, k, model_params):
    """
    Perform k-fold cross-validation.

    Parameters:
    - X: features, numpy array of shape (num_samples, num_features)
    - y: targets, numpy array of shape (num_samples, )
    - model: a classifier having fit and predict methods
    - k: number of folds
    - model_params: dictionary with values of model paramters

    Returns:
    - mean_accuracy: the average accuracy over the k-folds
    """
    num_samples = X.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    fold_size = num_samples // k
    accuracies = []

    for i in range(k):
        # Split data into train and test for this fold
        test_indices = indices[i * fold_size: (i + 1) * fold_size]
        train_indices = np.setdiff1d(indices, test_indices)
        
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        
        # Fit model and predict
        w, loss = model(y_train, X_train, **model_params)
        
        y_pred = predict_logistic(X_test, w)
        
        # Calculate accuracy for this fold and append to accuracies list
        accuracy = np.mean(y_pred == y_test)
        accuracies.append(accuracy)
    
    # Calculate mean accuracy over all k-folds
    mean_accuracy = np.mean(accuracies)
    
    return mean_accuracy




def hyperparameter_tuning(X, y, model, lambdas, gammas, model_params,  k=5):
    """
    Tune hyperparameter using k-fold cross-validation.

    Parameters:
    - X: features
    - y: targets
    - model_class: a class of the model that accepts the hyperparameter in its constructor
    - param_name: name of the hyperparameter to be tuned
    - param_values: list of values for the hyperparameter
    - k: number of folds for cross-validation

    Returns:
    - best_param_value: the value of the hyperparameter that gives the best cross-validation accuracy
    """
    best_accuracy = 0
    best_param_lambda = None
    best_param_gamma = None
    
    for gamma in gammas: 
        for lambda_ in lambdas: 

            model_params['lambda_'] = lambda_
            model_params['gamma'] = gamma
            # model = model(X, y, k, **model_params)  # Construct model with the hyperparameter
            accuracy = k_fold_cross_validation(X, y, model, k, model_params)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_param_lambda = lambda_
                best_param_gamma = gamma
                
            print(f" lambda= {lambda_} gamma= {gamma} , CV accuracy = {accuracy:.4f}")
        
    return best_param_lambda, best_param_gamma

 