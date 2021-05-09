from utils import run_if
import numpy as np
import pandas as pd
import random
from utils import CART_decision_forest


def decision_forests(train_data, test_data, min_size_node):
    # Number of features in the data
    n_feat = len(train_data[0]) - 1
    # Number of trees desired
    NT = [1, 10, 25, 50, 75, 100]

    # Initialize numpy array to save the accuracies, 4 is the maximum value of different F's
    accuracies = [[[] for _ in range(4)] for _ in range(len(NT))]
    # Initialize an empty dictionary to save the feature importances
    feat_importances = {}

    for idx_nt, n_trees in enumerate(NT):
        print("Number of trees in the Decision Forest: ", n_trees)
        # Number of random features used in the splitting of the nodes (DF)
        F = list(sorted({int(n_feat / 4), int(n_feat / 2), int(3 * n_feat / 4)})) + ["run_if"]
        for idx_f, n_random_features in enumerate(F):
            print("Number of possible features considered in each split: ", n_random_features)
            # Generate a decision forest, which is a list (forest) of dictionaries (decision trees)
            # feature_importances is a dictionary of the importance corresponding to each feature in the forest
            decision_forest, feature_importances = generate_decision_forest(train_data, n_trees, n_random_features, min_size_node)
            feat_importances["NT = {} and F = {}".format(n_trees, n_random_features)] = feature_importances

            # Prediction over the test data
            for test_instance in range(len(test_data)):
                prediction = classify_decision_forest(decision_forest, test_data[test_instance])
                test_data[test_instance].append(prediction)

            # Compute the accuracy by counting the number of instances with Class = Predicted_Class
            matches = len([True for idx in range(len(test_data)) if test_data[idx][-2] == test_data[idx][-1]])
            accuracy = matches * 100 / len(test_data)

            accuracies[idx_nt][idx_f] = accuracy

    names_columns_accuracies = [str(F[i]) for i in range(3)] + ["Run_if"]
    df_accuracies = pd.DataFrame(accuracies, columns=names_columns_accuracies)
    df_accuracies.index = [str(NT[i]) for i in range(len(NT))]

    df_feat_importances = pd.DataFrame(feat_importances)
    return df_accuracies, df_feat_importances


def generate_decision_forest(train_data, number_trees, number_random_features, min_size):
    decision_forest = []
    # Initialize feature importances to 0
    feat_count = np.zeros(len(train_data[0]) - 1)

    for tree in range(number_trees):
        # If the parameter F is run_if we generate a different number of possible features for each tree
        if number_random_features == "run_if":
            number_random_features = run_if(len(train_data[0]) - 1)

        # For each tree we pick a combination of random attributes to generate it and add the class attribute
        features_chosen = sorted(random.sample(range(len(train_data[0])-1), number_random_features))
        features_chosen.append(-1)
        filtered_train_data = [[element[i] for i in features_chosen] for element in train_data]
        # With this filtered version of the data (with only F features and the class) we generate the tree
        # If a node has less than the 5% of the data we will make it a terminal node
        this_tree, this_feature_count = CART_decision_forest(filtered_train_data, min_size=min_size)
        this_tree["features_chosen"] = features_chosen[:-1]

        # Add the new tree to the decision forest
        decision_forest.append(this_tree)

        # Add the feature importances of this tree
        for idx, feature in enumerate(features_chosen[:-1]):
            feat_count[feature] += this_feature_count[idx]

    # Normalize the feature importances according to the whole forest
    feat_imp = feat_count / sum(feat_count)
    feat_imp_dict = {"Attribute_{}".format(i): feat_imp[i] for i in range(len(feat_imp))}

    return decision_forest, feat_imp_dict


def classify_decision_forest(decision_forest, test_instance):
    # Initialize empty a list of predictions for each decision tree
    predictions = []
    # Make a prediction with each decision tree
    for tree in decision_forest:
        features_chosen = tree["features_chosen"]
        filtered_test_instance = [test_instance[i] for i in features_chosen]
        prediction = predict(tree, filtered_test_instance)
        predictions.append(prediction)
    return max(set(predictions), key=predictions.count)


def predict(node, instance):
    """
    :param node: node at where we are making the prediction
    :param instance: test instance we are trying to classify
    :return: predicted class of the test instance
    """
    if instance[node['attribute_index']] < node['splitting_value']:
        if isinstance(node['X_1'], dict):
            return predict(node['X_1'], instance)
        else:
            return node['X_1']
    else:
        if isinstance(node['X_2'], dict):
            return predict(node['X_2'], instance)
        else:
            return node['X_2']
