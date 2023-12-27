from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from matplotlib import pyplot as plt

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

def get_d_star_from_cross_validation_error(fold_test_prediction_error: np.array, ds) -> tuple[float, float, int, int]:
    error_over_folds = fold_test_prediction_error[:, :].mean(axis=0)  # average over the folds
    index_min_error_flat = np.argmin(error_over_folds)  # get the index of the minimum cross validation error
    d_star_index = np.unravel_index(
        index_min_error_flat, error_over_folds.shape
    )
    d_star = ds[d_star_index[0]]
    return d_star, d_star_index[0]