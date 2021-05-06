import math
import numpy as np


def random_forests(data_name, train_df, test_df):

    # Number of features in the data
    n_feat = train_df.shape[1] - 1

    # Number of trees desired
    NT = [1, 10, 25, 50, 75, 100]
    # Number of random features used in the splitting of the nodes (RF)
    F = {1, 3, int(math.log(n_feat + 1, 2)), math.floor(math.sqrt(n_feat))}

    # Initialize numpy array to save the accuracies
    accuracies = np.zeros((len(NT), len(F)))

    for n_trees in NT:
        for n_random_features in F:
            generate_random_forest(train_df, n_trees, n_random_features)


def generate_random_forest(train_data, number_trees, number_random_features):

    return None
