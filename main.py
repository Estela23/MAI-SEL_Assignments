from utils.preprocess import read_file_generate_dfs
from train_RULES import train_RULES
from test_RULES import test_RULES

# We considered 3 datasets, one small, one medium and one large, with different types of attributes
# (categorical, float or integer). For each of them we properly preprocess and split the data into train and test data,
# then train a RULES classifier system to infer rules. Once we generate these rules, we classify the instances in the
# test data and save the predicted classification to an external .csv file, this was done to easily review the results.
# Finally, we evaluate the algorithm by measuring the coverage of each rule and the accuracy over the test data.

sizes = ["small", "medium", "large"]

for size in sizes:
    if size == "small":
        data_file = "data/new-thyroid.data"
        data_name = "new-thyroid"
        arff = False
        id_column = None
        class_column = 0
        discretize_integers = [1]
    elif size == "medium":
        data_file = "data/car.data"
        data_name = "car"
        arff = False
        id_column = None
        class_column = None
        discretize_integers = []
    elif size == "large":
        """data_file = "data/splice.arff"
        data_name = "splice"
        arff = True
        id_column = None
        class_column = None
        discretize_integers = None"""
        data_file = "data/kr-vs-kp.data"
        data_name = "kr-vs-kp"
        arff = False
        id_column = None
        class_column = None
        discretize_integers = []
    else:
        print("Data size must be fixed to some value, either 'small', 'medium' of 'large'.")
        break

    train_df, test_df = read_file_generate_dfs(data_file, arff=arff, id_column=id_column, class_column=class_column,
                                               discretize_integers=discretize_integers)

    rules = train_RULES(df=train_df, data_name=data_name)

    result_df, accuracy, n_unclassified = test_RULES(df=test_df, inferred_rules=rules)
    result_df.to_csv("results/predicted_{0}.csv".format(data_name), index=False)

    # Finally, we include the general results of the classifier (accuracy and number of not classified instances)
    with open("results/RULES_{0}.txt".format(data_name), "a") as output:
        output.write("\n\nClassifying {0} test data we obtained accuracy = {1:.4f}% and a total of {2} unclassified"
                     " instances.".format(data_name, accuracy, n_unclassified))
