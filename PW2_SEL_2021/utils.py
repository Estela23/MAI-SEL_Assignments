import random


def run_if(max_n):
    n = random.randint(1, max_n)
    print(n)


def gini_index(groups, classes):
    """
    Calculate the Gini index for a split dataset
    :param groups: list of lists where, each list is a group of instances
    :param classes: list with the corresponding classes of each group, note that len(groups) = len(classes)
    :return: Gini index for the given split
    """
    # count how many rows there are at the current split point
    n_rows = float(sum([len(group) for group in groups]))
    # initialize Gini index to zero
    gini_idx = 0.0
    for group in groups:
        size_group = float(len(group))
        # TODO: check how to put this without continue
        """# avoid divide by zero
        if size_group == 0:
            continue"""
        score = 0.0
        # Score of each group based on the score of each class
        for this_class in classes:
            this_score = len([row[-1] for row in group if row[-1] == this_class]) / size_group
            score += this_score ** 2
        # Weight the score by the group size
        gini_idx += (1.0 - score) * (size_group / n_rows)
    return gini_idx


def test_split(index, value, dataset):
    """
    Split a dataset based on a numerical attribute and a splitting value
    :param index: index of the attribute with respect to which we are splitting the data
    :param value: value of the attribute to split from it
    :param dataset: original dataset before the split
    :return: two lists of the divided data
    """
    left = []
    right = []
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right



