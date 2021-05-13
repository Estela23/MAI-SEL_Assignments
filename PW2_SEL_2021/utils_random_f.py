#######################
#### Random Forest ####
#######################
import numpy as np
import random


def bootstrap(data):
    """
    :param data: whole train dataset
    :return: a bootstrapped sample of the train dataset
    """
    indexes = np.random.choice(range(len(data)), size=len(data), replace=True)
    return [data[i] for i in indexes]


def gini_index_rf(groups, classes):
    """
    Calculate the Gini index for a split dataset
    :param groups: list of lists where, partition of the data we are splitting, X_1 and X_2
    :param classes: list with the corresponding classes of each group
    :return: Gini index for the given split
    """
    # count how many rows of data there are at the current split point
    n_rows = float(sum([len(group) for group in groups]))
    # initialize Gini index to zero
    gini_idx = 0.0
    for group in groups:
        size_group = float(len(group))

        # If a partition has only one set return the best gini index possible
        if size_group == 0:
            return 0

        score = 0.0
        # Score of each group based on the score of each class
        for this_class in classes:
            this_score = len([row[-1] for row in group if row[-1] == this_class]) / size_group
            score += this_score ** 2
        # Weight the score by the group size
        gini_idx += (1.0 - score) * (size_group / n_rows)
    return gini_idx


def test_split_continuous_rf(index_attribute, splitting_value, dataset):
    """
    Split a dataset based on a numerical attribute and a splitting value
    :param index_attribute: index of the attribute with respect to which we are splitting the data
    :param splitting_value: value of the attribute to split from it
    :param dataset: original dataset before the split
    :return: two lists of the divided data
    """
    X_1 = []
    X_2 = []
    for instance in dataset:
        if instance[index_attribute] < splitting_value:
            X_1.append(instance)
        else:
            X_2.append(instance)
    return X_1, X_2


def create_split_rf(dataset, n_features):
    """
    :param n_features: number of features used to create the split at the current node
    :param dataset: dataset over which we want to make a split
    :return: the best split possible (lowest gini index)
    """
    possible_classes = list(set(instance[-1] for instance in dataset))

    # Features chosen to split the current node, then add the class
    features_chosen = sorted(random.sample(range(len(dataset[0]) - 1), n_features))

    # Initialize gini index, in each iteration we will try to improve (decrease) it
    best_gini_index = 2
    for attribute in features_chosen:
        if isinstance(dataset[0][attribute], int) or isinstance(dataset[0][attribute], float):
            # For a numeric attribute we sort all its values as splitting values
            values = sorted(set([dataset[i][attribute] for i in range(len(dataset))]))
            splitting_values = [(values[i] + values[i+1]) / 2 for i in range(len(values)-1)]

            if not splitting_values:
                splitting_values = [dataset[0][attribute]]

            # Do the split and compute its gini index
            for splitting_value in splitting_values:
                partition = test_split_continuous_rf(index_attribute=attribute, splitting_value=splitting_value, dataset=dataset)
                gini_score = gini_index_rf(partition, possible_classes)
                if gini_score < best_gini_index:
                    best_gini_index = gini_score
                    best_splitting_value = splitting_value
                    best_attribute_index = attribute
                    best_partition = partition
        # TODO: como spliteamos para atributos categóricos!!
        # TODO: Dani dice que se hace encoding y listo, no estoy segura

    return {"attribute_index": best_attribute_index, "splitting_value": best_splitting_value, "partition": best_partition}


def to_terminal_rf(group_partition):
    """
    :param group_partition: instances of data existing in a terminal node
    :return: the corresponding class of a terminal node (usually node with less than 10% of the original data)
    """
    classes = [instance[-1] for instance in group_partition]
    return max(set(classes), key=classes.count)


def split_rf(node, min_size, feature_count, n_features):
    """
    Generates the split of a node and its child nodes
    :param node: node in which we are making the split
    :param min_size: minimum number of instances covered by a terminal node
    :param feature_count: list in which we keep track of the times each feature appears in the tree
    :param n_features: number of features to take into account when choosing the one for splitting
    """
    X_1, X_2 = node["partition"]
    del(node["partition"])
    # check if some of the sets of the partition is empty
    if not X_1 or not X_2:
        node["X_1"] = node["X_2"] = to_terminal_rf(X_1 + X_2)
        return
    # split left child with two conditions: X_1 smaller than the minimum size or all the instances are of the same class
    if len(X_1) <= min_size or all(X_1[i][-1] == X_1[0][-1] for i in range(len(X_1))):
        node["X_1"] = to_terminal_rf(X_1)
    else:
        node["X_1"] = create_split_rf(X_1, n_features)
        feature_count[node["X_1"]["attribute_index"]] += 1
        split_rf(node["X_1"], min_size, feature_count, n_features)
    # split right child
    if len(X_2) <= min_size or all(X_2[i][-1] == X_2[0][-1] for i in range(len(X_2))):
        node["X_2"] = to_terminal_rf(X_2)
    else:
        node["X_2"] = create_split_rf(X_2, n_features)
        feature_count[node["X_2"]["attribute_index"]] += 1
        split_rf(node["X_2"], min_size, feature_count, n_features)


def CART_random_forest(dataset, number_features, min_size):
    # Initialize an array of zeros to count the times each feature appears in the tree
    feature_count = np.zeros(len(dataset[0]) - 1)

    root = create_split_rf(dataset, number_features)
    feature_count[root["attribute_index"]] += 1

    split_rf(root, min_size, feature_count, number_features)
    return root, feature_count
