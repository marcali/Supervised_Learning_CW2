from dataclasses import dataclass
from typing import NamedTuple
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from helpers import *
from collections import Counter
from itertools import combinations

#polynomial kernel
def polynomial_kernel(p, q, d):
    return (1 + (p@q.T)) ** d

def gaussian_kernel_matrix(X1, X2, sigma):
    # It looks a little weird cause its vectorised
    sq_dists = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
    return np.exp(-sigma * sq_dists)

#convering the data to -1 and 1
def mysign(data):
    return np.where(data <= 0.0, -1.0, 1.0)

def create_summary_table(mean_train_errors, mean_std_train_errors, mean_test_errors, mean_std_test_errors, range, parameter_name='Polynomial degree '):
    # Create a summary table
    columns = ['parameter', 'Mean Train Error', 'Std Train Error', 'Mean Test Error', 'Std Test Error']

    # Take an average over all the runs
    kernel_perceptron = pd.concat((
        pd.DataFrame(np.round(mean_train_errors,5), columns=[columns[1]]),
        pd.DataFrame(np.round(mean_std_train_errors,5), columns=[columns[2]]),
        pd.DataFrame(np.round(mean_test_errors,5), columns=[columns[3]]),
        pd.DataFrame(np.round(mean_std_test_errors,5), columns=[columns[4]])), axis=1
    )
    kernel_perceptron.set_index(parameter_name + pd.Series(range).astype(str), inplace=True)
    kernel_perceptron.index.name = 'parameter'
    #kernel_perceptron = kernel_perceptron.style.format("{:.5f}")

    return kernel_perceptron

def calculate_confusion_matrix(y_true, y_pred):
    classes = np.unique(y_true)
    matrix = np.zeros((len(classes), len(classes)))

    for true_class in classes:
        for pred_class in classes:
            if true_class != pred_class:  # Exclude diagonal elements
                matrix[true_class, pred_class] = np.sum((y_true == true_class) & (y_pred == pred_class)) / np.sum(y_true == true_class)

    return matrix

def calculate_confusion_matrix_std(confusion_matrices):
    return np.std(confusion_matrices, axis=0)

def plot_confusion_matrix(confusion_matrices):
    std_dev_matrix = calculate_confusion_matrix_std(confusion_matrices)
    mean_confusion_matrix = confusion_matrices.mean(axis=0)
    combined_matrix = np.vectorize(lambda mean, std: f"{mean*100:.2f}% \n± \n{std*100:.2f}%")(mean_confusion_matrix, std_dev_matrix)

    plt.figure(figsize=(12, 10), dpi=80)
    sns.set(style="whitegrid", font_scale=1.1)
    ax = sns.heatmap(mean_confusion_matrix, annot=combined_matrix, fmt="", cmap='coolwarm', cbar=True, square=True, annot_kws={"size": 12}, xticklabels=np.arange(10), yticklabels=np.arange(10))
    plt.title("Error Rate and Standard Deviation", fontsize=19)
    plt.xlabel('Predicted Label', fontsize=19)
    plt.ylabel('True Label', fontsize=19)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.show()
    plt.savefig('confusion_matrix.png')

def calculate_mean_star_param_and_mean_test_error_with_std(test_errors, stars_param, param):
    mean_cross_validation_test_error_with_star_param = np.mean(test_errors, axis=0)
    mean_cross_validation_test_error_std_with_star_param = np.std(test_errors, axis=0)
    mean_cross_validation_star_param = np.mean(stars_param)
    mean_cross_validation_std_star_param = np.std(stars_param)
    print("Mean Test Error {param}: ", mean_cross_validation_test_error_with_star_param, "±", mean_cross_validation_test_error_std_with_star_param)
    print("Mean Best {param}", mean_cross_validation_star_param, "±", mean_cross_validation_std_star_param)

    part_2_results = {
            f"Mean Best {param}": mean_cross_validation_star_param,
            f"Mean Best {param} standard deviation": mean_cross_validation_std_star_param,
            "Mean Test Error": mean_cross_validation_test_error_with_star_param,
            "Test Error Standard Deviation": mean_cross_validation_test_error_std_with_star_param
        }

    df_results_2 = pd.DataFrame.from_dict(part_2_results, orient='index')
    df_results_2.reset_index(inplace=True)
    
    return df_results_2, mean_cross_validation_star_param

def plot_test_error(values, mean_test_errors_1vs1, optimizable_param_name, kernel_name):
    plt.figure(figsize=(10, 5), dpi=100)
    plt.plot(values, mean_test_errors_1vs1*100, label='Test Error', marker='o', color='red')
    plt.title(f'Mean Test Error Rates vs Kernel {kernel_name} Parameters', fontsize=18)
    plt.xlabel(f'{optimizable_param_name}', fontsize=18)
    plt.ylabel('Error Rate (%)', fontsize=18)
    plt.legend(fontsize=18)
    plt.tight_layout()
    plt.show()

@dataclass
class FoldData:
    fold_index: int
    start_index: int
    stop_index: int
    train_data: np.array
    train_labels: np.array
    test_data: np.array
    test_labels: np.array
    
def create_folds_data(X_train, Y_train, number_of_folds: int) -> list[FoldData]:
    """
    Create a list of FoldData helper objects which contain all the data needed to define a fold.

    :param X_train: training data
    :param Y_train: training labels
    :param number_of_folds: number of folds to split the data into
    :return: list of FoldData objects
    """
    fold_size = X_train.shape[0] / number_of_folds

    fold_data = []
    for i_fold in range(number_of_folds):
        # Define the fold start/stop as the range of the test data in the fold
        fold_start = int(i_fold * fold_size)
        fold_stop = int((i_fold + 1) * fold_size) if i_fold != number_of_folds - 1 else X_train.shape[0]

        fold_test_data = X_train[fold_start:fold_stop, :]
        fold_test_labels = Y_train[fold_start:fold_stop]

        # The training data is all the non-test data
        fold_train_data = np.vstack((X_train[:fold_start, :], X_train[fold_stop:, :]))
        fold_train_labels = np.concatenate((Y_train[:fold_start], Y_train[fold_stop:]))

        fold_data.append(
            FoldData(
                fold_index=i_fold,
                start_index=fold_start,
                stop_index=fold_stop,
                train_data=fold_train_data,
                train_labels=fold_train_labels,
                test_data=fold_test_data,
                test_labels=fold_test_labels,
            )
        )

    return fold_data

def get_d_star_from_cross_validation_error(fold_test_prediction_error: np.array, ds):
    error_over_folds = fold_test_prediction_error[:, :].mean(axis=0)  # average over the folds
    index_min_error_flat = np.argmin(error_over_folds)  # get the index of the minimum cross validation error
    d_star_index = np.unravel_index(
        index_min_error_flat, error_over_folds.shape
    )
    d_star = ds[d_star_index[0]]
    return d_star, d_star_index[0]

#splitting data and labels
def split_into_data_and_labels(data):
    y = data[:,0].astype(int)
    #convert to -1 and 1 here instead of in the loop
    x = data[:, 1:]
    return x, y

### QUESTION 4

def plot_label_counts(label_counts):
    plt.figure(figsize=(10,6))
    plt.bar(label_counts.keys(), label_counts.values())
    plt.xlabel('Labels')
    plt.ylabel('Count')
    plt.title('Count of each label')
    plt.show()
    

### Question 6
def calculate_accuracy(actual, predicted):
    return np.mean(actual == predicted)


def softmax(x):
    # Subtracting the max can help with numerical stability
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


def make_probabilistic_predictions(kappas):
    """
    How does this function work?
    - We'll take in our kappas (confidence vectors)
    - We'll convert them to probabilities using softmax
    - Then we can sample from the probabilities
    """
    predictions = []
    for kappa in kappas:
        # Convert kappa to probabilities using softmax
        kappa = np.expand_dims(kappa, axis=0)  # Ensure kappa is a 2D array
        probabilities = softmax(kappa)
        
        # Now we can sample from the probabilities
        predictions.append(np.random.choice(len(probabilities[0]), p=probabilities[0]))
    return np.array(predictions)