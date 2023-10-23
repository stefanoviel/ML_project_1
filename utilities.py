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


def predict_logistic(X, w, threshold): 
    return (sigmoid(X @ w) >= threshold).flatten()

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



def k_fold_cross_validation(X, y, model, k, model_params, threshold):
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
    f1_scores = []

    for i in range(k):
        # Split data into train and test for this fold
        test_indices = indices[i * fold_size: (i + 1) * fold_size]
        train_indices = np.setdiff1d(indices, test_indices)
        
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        
        # Fit model and predict
        w, loss = model(y_train, X_train, **model_params)
        
        y_pred = predict_logistic(X_test, w, threshold)
        
        # Calculate accuracy for this fold and append to accuracies list
        accuracy = np.mean(y_pred == y_test)
        f1 = compute_f1(y_test, y_pred)
        accuracies.append(accuracy)
        f1_scores.append(f1)
    
    # Calculate mean accuracy over all k-folds
    mean_accuracy = np.mean(accuracies)
    f1_score = np.mean(f1_scores)
    
    return mean_accuracy, f1_score




def hyperparameter_tuning(X, y, model, lambdas, gammas, thresholds, model_params,  k=5):
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
    best_f1_score = 0
    best_param_lambda = None
    best_param_gamma = None
    best_param_threshold = None
    
    for gamma in gammas: 
        for lambda_ in lambdas: 
            for threshold in thresholds: 


                model_params['lambda_'] = lambda_
                model_params['gamma'] = gamma
                accuracy, f1_score = k_fold_cross_validation(X, y, model, k, model_params, threshold)
                
                if f1_score > best_f1_score and accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_f1_score = f1_score
                    best_param_lambda = lambda_
                    best_param_gamma = gamma
                    best_param_threshold = threshold
                    
                print(f" lambda= {lambda_}, gamma= {gamma}, threshold = {threshold} CV accuracy = {accuracy:.4f}, f1_score = {f1_score:.4f}")
        
    return best_param_lambda, best_param_gamma, best_param_threshold

 
def compute_f1(y_true, y_pred):
    # Ensure the two arrays are of the same length
    assert y_true.shape[0] == y_pred.shape[0], "Mismatched length between y_true and y_pred."

    # True Positives
    TP = np.sum(np.logical_and(y_pred == 1, y_true == 1))

    # False Positives
    FP = np.sum(np.logical_and(y_pred == 1, y_true == 0))

    # False Negatives
    FN = np.sum(np.logical_and(y_pred == 0, y_true == 1))

    # Precision and Recall
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0

    # F1 Score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return f1


def balance_dataset(x_train, y_train):
    indices_0 = np.where(y_train == 0)[0] # MAJORITY CLASS
    indices_1 = np.where(y_train == 1)[0] # MINORITY CLASS

    count_0 = len(indices_0)
    count_1 = len(indices_1)

    target_count = (count_0 + count_1) // 2  # find a middle ground to avoid extremely oversampling the minority class
    #target_count = max(count_0, count_1)
    #target_count = min(count_0, count_1)

    oversampled_indices = np.random.choice(indices_1, size=target_count, replace=True)
    oversampled_X = x_train[oversampled_indices]
    oversampled_y = y_train[oversampled_indices]

    undersampled_indices = np.random.choice(indices_0, size=target_count, replace=False)
    undersampled_X = x_train[undersampled_indices]
    undersampled_y = y_train[undersampled_indices]

    balanced_X = np.vstack((undersampled_X, oversampled_X))
    balanced_y = np.hstack((undersampled_y, oversampled_y))

    balanced_data = np.column_stack((balanced_X, balanced_y))
    np.random.shuffle(balanced_data)
    shuffled_balanced_X = balanced_data[:, :-1]
    shuffled_balanced_y = balanced_data[:, -1]

    return shuffled_balanced_X, shuffled_balanced_y


def clean_X_0(data):
    data = data[:, 1:]  
    data[np.isnan(data)] = 0  
    data = normalize(data)  
    ones_column = np.ones((data.shape[0], 1))
    return np.hstack((ones_column, data))

def clean_X_mean(data):
    data = data[:, 1:]  
    column_means = np.nanmean(data, axis=0)
    data[np.isnan(data)] = np.take(column_means, np.where(np.isnan(data))[1])
    data = normalize(data)  
    ones_column = np.ones((data.shape[0], 1))
    return np.hstack((ones_column, data))

def clean_X_median(x_data):
    x_data = x_data[:, 1:]  
    column_medians = np.nanmedian(x_data, axis=0)
    x_data[np.isnan(x_data)] = np.take(column_medians, np.where(np.isnan(x_data))[1])
    x_data = normalize(x_data)  
    return x_data

def clean_Y(y_data): 
    y_data = y_data[:, 1]  # remove ids
    y_data[y_data == -1] = 0  # set -1 to 0 
    return y_data


def columns_to_remove(x_data,t):
    remove = np.array([])

    for j in range(x_data.shape[1]):   
        # focus: columns with too many nan values
        num_nan = np.sum(np.isnan(x_data[:,j]))
        per_nan = num_nan / x_data.shape[0]
        
        # focus: columns with too many 77/99 values
        num_2729 = np.sum(x_data[:,j] == 77) + np.sum(x_data[:,j] == 99)
        per_2729 = num_2729 / x_data.shape[0]
        
        # focus: columns with too many 77/99 values
        num_3739 = np.sum(x_data[:,j] == 777) + np.sum(x_data[:,j] == 999)
        per_3739 = num_3739 / x_data.shape[0]

        # focus: columns with too many 7777/9999 values
        num_4749 = np.sum(x_data[:,j] == 7777) + np.sum(x_data[:,j] == 9999)
        per_4749 = num_4749 / x_data.shape[0]

        # focus: columns with too many 777777/999999 values
        num_6769 = np.sum(x_data[:,j] == 777777) + np.sum(x_data[:,j] == 999999)
        per_6769 = num_6769 / x_data.shape[0]

        # find out which features to remove
        num_remove = num_nan + num_2729 + num_3739 + num_4749 + num_6769
        per_remove = num_remove / x_data.shape[0]
        if ( per_remove >= t ):
            remove = np.append(remove,j)
            
    return remove.astype(int)

def reduced_data(x_data, t=0.6):
    filtered_data = np.delete(x_data, columns_to_remove(x_data, t), 1)
    return filtered_data