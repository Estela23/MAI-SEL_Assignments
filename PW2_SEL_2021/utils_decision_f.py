#########################
#### Decision Forest ####
#########################
import random
import numpy as np
import itertools


def run_if(max_n):
    n = random.randint(1, max_n)
    return n


def gini_index(groups, classes):
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


def test_split_continuous(index_attribute, splitting_value, dataset):
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


"""def power_set(iterable):
    ""
    From an iterable (list of attributes) returns the power set of the list, for categorical attributes
    :param iterable: list of attributes to take into account for splitting the data #TODO: WHAAAAAAAAAAAT
    :return: all possible combinations of splits
    ""
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(1, int(len(s)/2)+1))


def test_split_discrete(index_attribute, splitting_subsets, dataset):
    ""
    Split a dataset based on a categorical attribute and the splitting subsets
    :param index_attribute: index of the attribute with respect to which we are splitting the data
    :param splitting_subsets: possible values for the splitting attribute divided into two subsets
    :param dataset: original dataset before the split
    :return: two lists of the divided data
    ""
    X_1 = []
    X_2 = []
    for row in dataset:
        if row[index_attribute] < splitting_subsets:
            X_1.append(row)
        else:
            X_2.append(row)
    return X_1, X_2"""


def create_split(dataset):
    """
    :param dataset: dataset over which we want to make a split
    :return: the best split possible (lowest gini index)
    """
    possible_classes = list(set(instance[-1] for instance in dataset))
    # Initialize gini index, in each iteration we will try to improve (decrease) it
    best_gini_index = 2
    for attribute in range(len(dataset[0]) - 1):
        if isinstance(dataset[0][attribute], int) or isinstance(dataset[0][attribute], float):
            # For a numeric attribute we sort all its values as splitting values
            values = sorted(set([dataset[i][attribute] for i in range(len(dataset))]))
            splitting_values = [(values[i] + values[i+1]) / 2 for i in range(len(values)-1)]

            if not splitting_values:
                splitting_values = [dataset[0][attribute]]

            # Do the split and compute its gini index
            for splitting_value in splitting_values:
                partition = test_split_continuous(index_attribute=attribute, splitting_value=splitting_value, dataset=dataset)
                gini_score = gini_index(partition, possible_classes)
                if gini_score < best_gini_index:
                    best_gini_index = gini_score
                    best_splitting_value = splitting_value
                    best_attribute_index = attribute
                    best_partition = partition
        # TODO: como spliteamos para atributos categóricos!!
        # TODO: Dani dice que se hace encoding y listo, no estoy segura

    return {"attribute_index": best_attribute_index, "splitting_value": best_splitting_value, "partition": best_partition}


def to_terminal(group_partition):
    """
    :param group_partition: instances of data existing in a terminal node
    :return: the corresponding class of a terminal node (usually node with less than 10% of the original data)
    """
    classes = [instance[-1] for instance in group_partition]
    return max(set(classes), key=classes.count)


def split(node, min_size, feature_count):
    """
    Generates the split of a node and its child nodes
    :param node: node in which we are making the split
    :param min_size: minimum number of instances covered by a terminal node
    :param feature_count: list in which we keep track of the times each feature appears in the tree
    """
    X_1, X_2 = node["partition"]
    del(node["partition"])
    # check if some of the sets of the partition is empty
    if not X_1 or not X_2:
        node["X_1"] = node["X_2"] = to_terminal(X_1 + X_2)
        return
    # split left child with two conditions: X_1 smaller than the minimum size or all the instances are of the same class
    if len(X_1) <= min_size or all(X_1[i][-1] == X_1[0][-1] for i in range(len(X_1))):
        node["X_1"] = to_terminal(X_1)
    else:
        node["X_1"] = create_split(X_1)
        feature_count[node["X_1"]["attribute_index"]] += 1
        split(node["X_1"], min_size, feature_count)
    # split right child
    if len(X_2) <= min_size or all(X_2[i][-1] == X_2[0][-1] for i in range(len(X_2))):
        node["X_2"] = to_terminal(X_2)
    else:
        node["X_2"] = create_split(X_2)
        feature_count[node["X_2"]["attribute_index"]] += 1
        split(node["X_2"], min_size, feature_count)


def CART_decision_forest(dataset, min_size):
    # Initialize an array of zeros to count the times each feature (of the selected ones) appears in the tree
    feature_count = np.zeros(len(dataset[0]) - 1)

    root = create_split(dataset)
    feature_count[root["attribute_index"]] += 1

    split(root, min_size, feature_count)
    return root, feature_count





