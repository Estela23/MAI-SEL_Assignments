import math
import numpy as np
import pandas as pd
import time
from utils_random_f import CART_random_forest, bootstrap


def random_forests(train_data, test_data, min_size_node):
    # Number of features in the data
    n_feat = len(train_data[0]) - 1

    # Number of trees desired
    NT = [1, 10, 25, 50, 75, 100]
    # Number of random features used in the splitting of the nodes (RF)
    F = list(sorted({1, 3, int(math.log(n_feat, 2) + 1), math.floor(math.sqrt(n_feat))}))

    # Initialize numpy array to save the accuracies, 4 is the maximum value of different F's
    accuracies = [[[] for _ in range(len(F))] for _ in range(len(NT))]
    times = [[[] for _ in range(len(F))] for _ in range(len(NT))]

    # Initialize an empty dictionary to save the feature importances
    feat_importances = {}

    for idx_nt, n_trees in enumerate(NT):
        print("Number of trees in the Random Forest: ", n_trees)
        for idx_f, n_random_features in enumerate(F):
            print("Number of possible features considered in each split: ", n_random_features)
            # Initial time
            t0 = time.time()
            # Generate a random forest, which is a list (forest) of dictionaries (random trees)
            # feature_importances is a dictionary of the importance corresponding to each feature in the forest
            random_forest, feature_importances = generate_random_forest(train_data, n_trees, n_random_features, min_size_node)
            feat_importances["NT = {} and F = {}".format(n_trees, n_random_features)] = feature_importances

            # Prediction over the test data
            for test_instance in range(len(test_data)):
                prediction = classify_random_forest(random_forest, test_data[test_instance])
                test_data[test_instance].append(prediction)

            # Final time, after training and testing the current forest
            t1 = time.time()

            # Compute the accuracy by counting the number of instances with Class = Predicted_Class
            matches = len([True for idx in range(len(test_data)) if test_data[idx][-2] == test_data[idx][-1]])
            accuracy = matches * 100 / len(test_data)

            accuracies[idx_nt][idx_f] = accuracy
            times[idx_nt][idx_f] = t1 - t0

    # Finally, create the dataframes for the accuracies, times and feature importance of each random forest created
    names_columns = [str(F[i]) for i in range(len(F))]
    names_rows = [str(NT[i]) for i in range(len(NT))]

    df_accuracies = pd.DataFrame(accuracies, columns=names_columns)
    df_accuracies.index = names_rows

    df_times = pd.DataFrame(times, columns=names_columns)
    df_times.index = names_rows

    df_feat_importances = pd.DataFrame(feat_importances)

    return df_accuracies, df_times, df_feat_importances


def generate_random_forest(train_data, number_trees, number_random_features, min_size):
    random_forest = []
    # Initialize feature importances to 0
    feat_count = np.zeros(len(train_data[0]) - 1)

    for tree in range(number_trees):
        # For each tree we first bootstrap the data
        bootstrapped_train_data = bootstrap(train_data)

        # With this bootstrapped version of the data we generate the random tree
        # If a node has less than the 5% of the data we will make it a terminal node
        this_tree, this_feature_count = CART_random_forest(bootstrapped_train_data, number_random_features, min_size=min_size)

        # Add the new tree to the decision forest
        random_forest.append(this_tree)

        # Add the feature importances of this tree
        feat_count = [sum(x) for x in zip(feat_count, this_feature_count)]

    # Normalize the feature importances according to the whole forest
    feat_imp = feat_count / sum(feat_count)
    feat_imp_dict = {"Attribute_{}".format(i): feat_imp[i] for i in range(len(feat_imp))}

    return random_forest, feat_imp_dict


def classify_random_forest(random_forest, test_instance):
    # Initialize empty a list of predictions for each random tree
    predictions = []
    # Make a prediction with each random tree
    for tree in random_forest:
        prediction = predict_random_forest(tree, test_instance)
        predictions.append(prediction)
    return max(set(predictions), key=predictions.count)


def predict_random_forest(node, instance):
    """
    :param node: node at where we are making the prediction
    :param instance: test instance we are trying to classify
    :return: predicted class of the test instance
    """
    if instance[node['attribute_index']] < node['splitting_value']:
        if isinstance(node['X_1'], dict):
            return predict_random_forest(node['X_1'], instance)
        else:
            return node['X_1']
    else:
        if isinstance(node['X_2'], dict):
            return predict_random_forest(node['X_2'], instance)
        else:
            return node['X_2']
