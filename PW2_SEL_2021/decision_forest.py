from utils import run_if
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sn
import matplotlib.pyplot as plt


def decision_forests(data_name, train_df, test_df):
    # Number of features in the data
    n_feat = train_df.shape[1] - 1

    # Number of trees desired
    NT = [1, 10, 25, 50, 75, 100]
    # Number of random features used in the splitting of the nodes (DF)
    F = {int(n_feat / 4), int(n_feat / 2), int(3 * n_feat / 4), run_if(n_feat)}

    # Initialize numpy array to save the accuracies
    accuracies = np.zeros((len(NT), len(F)))

    for idx_nt, n_trees in enumerate(NT):
        for idx_f, n_random_features in enumerate(F):
            decision_forest = generate_decision_forest(train_df, n_trees, n_random_features)

            # Prediction will be a df with an added column
            test_df["Predicted_Class"] = classify_decision_forest(test_df)

            # Compute the accuracy by counting the number of instances with Class = Predicted_Class
            matches = sum(np.where(test_df["Class"] == test_df["Predicted_Class"], True, False))
            accuracy = matches * 100 / test_df.shape[0]

            accuracies[idx_nt][idx_f] = accuracy

            """# Create a confusion matrix of the results of the classified instances
            cm = confusion_matrix(test_df["Class"], test_df["Predicted_Class"])
            plt.figure(figsize=(10, 7))
            sn.heatmap(cm, annot=True)
            plt.title("Confusion Matrix of {0} dataset".format(data_name))
            plt.savefig("results/cms/cm_{0}".format(data_name))"""

    return accuracies


def generate_decision_forest(train_data, number_trees, number_random_features):
    return 0


def classify_decision_forest(test_data):
    return 0
