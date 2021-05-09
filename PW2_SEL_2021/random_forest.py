import math
import numpy as np
import pandas as pd
import random
from utils import CART_random_forest


def random_forests(train_data, test_data, min_size_node):
    # Number of features in the data
    n_feat = len(train_data[0]) - 1

    # Number of trees desired
    NT = [1, 10, 25, 50, 75, 100]
    # Number of random features used in the splitting of the nodes (RF)
    F = list(sorted({1, 3, int(math.log(n_feat, 2) + 1), math.floor(math.sqrt(n_feat))}))

    # Initialize numpy array to save the accuracies, 4 is the maximum value of different F's
    accuracies = [[[] for j in range(len(F))] for i in range(len(NT))]
    # Initialize an empty dictionary to save the feature importances
    feat_importances = {}

    for idx_nt, n_trees in enumerate(NT):
        print("Number of trees in the Random Forest: ", n_trees)
        for idx_f, n_random_features in enumerate(F):
            print("Number of possible features considered in each split: ", n_random_features)

            random_forest, feature_importances = generate_random_forest(train_data, n_trees, n_random_features, min_size_node)
            feat_importances["NT = {} and F = {}".format(n_trees, n_random_features)] = feature_importances

            # Prediction over the test data
            for test_instance in range(len(test_data)):
                prediction = classify_random_forest(random_forest, test_data[test_instance])
                test_data[test_instance].append(prediction)

            # Compute the accuracy by counting the number of instances with Class = Predicted_Class
            matches = len([True for idx in range(len(test_data)) if test_data[idx][-2] == test_data[idx][-1]])
            accuracy = matches * 100 / len(test_data)

            accuracies[idx_nt][idx_f] = accuracy

    names_columns_accuracies = [str(F[i]) for i in range(len(F))]
    df_accuracies = pd.DataFrame(accuracies, columns=names_columns_accuracies)
    df_accuracies.index = [str(NT[i]) for i in range(len(NT))]

    df_feat_importances = pd.DataFrame(feat_importances)
    return df_accuracies, df_feat_importances


def generate_random_forest(train_data, number_trees, number_random_features, min_size):
    # TODO: adapt generate decision forest to generate random forest!!!
    random_forest = []
    # Initialize feature importances to 0
    feat_count = np.zeros(len(train_data[0]) - 1)

    # TODO: For each tree the bootstraped is different! Hence the train and classifying tienen que ser seguidos digamos

    for tree in range(number_trees):
        # For each tree we pick a combination of random attributes to generate it and add the class attribute
        features_chosen = sorted(random.sample(range(len(train_data[0]) - 1), number_random_features))
        features_chosen.append(-1)
        filtered_train_data = [[element[i] for i in features_chosen] for element in train_data]
        # With this filtered version of the data (with only F features and the class) we generate the tree
        # If a node has less than the 5% of the data we will make it a terminal node
        this_tree, this_feature_count = CART_random_forest(filtered_train_data, min_size=min_size)
        this_tree["features_chosen"] = features_chosen[:-1]

        # Add the new tree to the decision forest
        random_forest.append(this_tree)

        # Add the feature importances of this tree
        for idx, feature in enumerate(features_chosen[:-1]):
            feat_count[feature] += this_feature_count[idx]

    # Normalize the feature importances according to the whole forest
    feat_imp = feat_count / sum(feat_count)
    feat_imp_dict = {"Attribute_{}".format(i): feat_imp[i] for i in range(len(feat_imp))}

    return random_forest, feat_imp_dict


def classify_random_forest(random_forest, test_instance):
    # TODO: adaptar, tengo que pasarle el test_instance entera, no solo con los atributos elegidos digamos
    # Initialize empty a list of predictions for each decision tree
    predictions = []
    # Make a prediction with each decision tree
    for tree in random_forest:
        features_chosen = tree["features_chosen"]
        filtered_test_instance = [test_instance[i] for i in features_chosen]
        prediction = predict(tree, filtered_test_instance)
        predictions.append(prediction)
    return max(set(predictions), key=predictions.count)


def predict(node, instance):
    # TODO: checkear que es igual, si no lo es cambiarla y el nombre, si lo es importar de utils (mover la otra)
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
