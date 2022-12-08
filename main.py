import numpy as np
import pandas as pd

def cross_validation(complete_data, cur_features, feat_to_add):
    """
    This function performs cross-validation on the given data to determine the accuracy of a particular subset of features.
    It returns the accuracy as a decimal value between 0 and 1.

    Args:
        complete_data: The complete dataset to be used for cross-validation, including all features and labels.
        cur_features: The subset of features to be considered for cross-validation.
        feat_to_add: The additional feature to be considered for cross-validation (if any).

    Returns:
        The accuracy of the given subset of features as a decimal value between 0 and 1.
    """

    # Use NumPy's copy() function to deep copy the complete_data array
    data = np.copy(complete_data)

    # Set all features not in cur_features or feat_to_add to 0
    for i in range(1, complete_data.shape[1]):
        if i not in cur_features and i != feat_to_add:
            data[:, i] = 0

    num_correct = 0
    num_rows = data.shape[0]

    # Iterate through each row in the dataset
    for i in range(num_rows):
        objs_to_classify = data[i, 1:]
        label = data[i, 0]

        # Set initial values for nearest neighbor
        nn_dist = nn_loc = 10000.0
        nn_lbl = 0.0

        # Iterate through each row in the dataset again to find the nearest neighbor
        for j in range(num_rows):
            if i != j:
                dist = np.sqrt(sum(pow(objs_to_classify - data[j, 1:], 2)))
                if dist <= nn_dist:
                    nn_dist = dist
                    nn_loc = j
                    nl_lbl = data[nn_loc, 0]

        # Check if the nearest neighbor is the correct label
        if label == nl_lbl:
            num_correct += 1

    # Calculate the accuracy
    accuracy = num_correct / num_rows
    return accuracy


def backward_elimination(data):
    # set of features selected
    current_features = set(range(1, data.shape[1]))

    # dictionary to track subsets of feature performances
    feature_performances = {}
    # stores accuracies at each iteration
    accuracies = set()
    # stores the accuracy with all features
    overall_accuracy = 0

    num_cols = data.shape[1]
    # iterate through each level
    for i in range(num_cols - 1):
        # store feature to add and best accuracy so far
        feature_to_remove = None
        best_accuracy = 0

        print('Evaluating level {} of the search tree'.format(i + 1))
        # iterate through each feature
        for j in range(num_cols - 1):
            if j + 1 in current_features:
                print('\tConsidering removing feature {}'.format(j + 1))

                # create a copy of the current features and remove the feature being considered
                sub_features = current_features.copy()
                sub_features.remove(j + 1)
                accuracy = cross_validation(data, sub_features, 0)

                if i == 0 and j == 0:
                    overall_accuracy = accuracy

                # update best accuracy if improvement exists
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    print('Updated accuracy:', best_accuracy)
                    feature_to_remove = j + 1

                # keeps track of accuracies
                accuracies.add(best_accuracy)

        # remove the feature that was selected to be removed
        current_features.remove(feature_to_remove)

        # keep track of feature sets and corresponding accuracies
        if round(best_accuracy, 3) not in feature_performances.keys():
            feature_performances[round(best_accuracy, 3)] = current_features.copy()

        print('Feature {} removed on level {}\n'.format(feature_to_remove, i + 1))

    print('Best features: ', feature_performances[round(max(accuracies), 3)])
    print('Best accuracy: ', round(max(accuracies), 3))
    print('Accuracy with all features:', overall_accuracy)

    return accuracies, feature_performances



def forward_selection(data):
    # set of features selected
    current_features = set()

    # dictionary to track subsets of feature performances
    feature_performances = {}
    accuracies = set()

    num_cols = data.shape[1]
    # iterate through each level
    for i in range(num_cols - 1):
        # store feature to add and best accuracy so far
        feature_to_add = None
        best_accuracy = 0

        print('Evaluating level {} of the search tree'.format(i + 1))
        # iterate through each feature
        for j in range(num_cols - 1):
            if j + 1 not in current_features:
                print('\tConsidering adding feature {}'.format(j + 1))
                accuracy = cross_validation(data, current_features, j + 1)

                # update best accuracy if improvement exists
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    print('Updated accuracy:', best_accuracy)
                    feature_to_add = j + 1

                # keeps track of accuracies
                accuracies.add(round(best_accuracy, 3))

        # add the feature that was selected to be added
        current_features.add(feature_to_add)
        # keep track of feature sets and corresponding accuracies
        if round(best_accuracy, 3) not in feature_performances.keys():
            feature_performances[round(best_accuracy, 3)] = current_features.copy()
        print('Feature {} added on level {}\n'.format(feature_to_add, i + 1))

    print('Best features: ', feature_performances[max(accuracies)])
    print('Best accuracy: ', max(accuracies))
    print('Accuracy with all features:', best_accuracy)

    return accuracies, feature_performances

def main():
    # Prompt the user to input the dataset they want to use
    dataset = input("Please enter the dataset you want to use: ")

    # Pass the dataset to a pandas DataFrame
    data = pd.read_csv(dataset, header=None, sep=r"\s+")

    # Prompt the user to input either 1 or 2
    user_input = int(input("1. Forward Selection\n2. Backward Elimination\nEnter 1 or 2: "))

    
    
    # If the user selects 1, call the forward_selection() function
    if user_input == 1:
        
        forward_selection(data)
    # If the user selects 2, call the backward_elimination() function
    else:
        backward_elimination(data)

main()