from PW1_SEL_2021.utils.preprocess import read_file_generate_dfs
from PW1_SEL_2021.train_RULES import train_RULES
from PW1_SEL_2021.test_RULES import test_RULES

# Running this Python file we train and test a rule-based in 6 different datasets

# We considered 3 sizes for datasets: small, medium and large, and two examples of datasets for each size with different
# types of attributes (categorical, float or integer). For each of them we properly preprocess and split the data into
# train and test data, then train a RULES classifier system to infer rules. Once we generate these rules, we classify
# the instances in the test data and save the predicted classification to an external .csv file, this was done to easily
# review the results. Finally, we evaluate the algorithm by measuring the coverage of each rule, the global accuracy
# over the test data and plotting the confusion matrix to check if there is some type of imbalance in the data.

sizes = ["small", "medium", "large"]
choices = [1, 2]

for size in sizes:
    for choice in choices:
        if size == "small":
            if choice == 1:
                data_file = "data/new-thyroid.data"
                data_name = "new-thyroid"
                arff = False
                id_column = None
                class_column = 0
                discretize_integers = [1]
            elif choice == 2:
                data_file = "data/breast-cancer.data"
                data_name = "breast-cancer"
                arff = False
                id_column = None
                class_column = 0
                discretize_integers = []

        elif size == "medium":
            if choice == 1:
                data_file = "data/car.data"
                data_name = "car"
                arff = False
                id_column = None
                class_column = None
                discretize_integers = []
            elif choice == 2:
                data_file = "data/data_banknote_authentication.txt"
                data_name = "data_banknote_authentication"
                arff = False
                id_column = None
                class_column = None
                discretize_integers = []

        elif size == "large":
            if choice == 1:
                data_file = "data/phoneme.dat"
                data_name = "phoneme"
                arff = False
                id_column = None
                class_column = None
                discretize_integers = []
            elif choice == 2:
                data_file = "data/titanic.dat"
                data_name = "titanic"
                arff = False
                id_column = None
                class_column = None
                discretize_integers = []

        train_df, test_df = read_file_generate_dfs(data_file, arff=arff, id_column=id_column, class_column=class_column,
                                                   discretize_integers=discretize_integers)

        rules = train_RULES(df=train_df, data_name=data_name)

        result_df, accuracy, n_unclassified = test_RULES(df=test_df, inferred_rules=rules, data_name=data_name)
        result_df.to_csv("results/predicted_{0}.csv".format(data_name), index=False)

        # Finally, we include the general results of the classifier (accuracy and number of not classified instances)
        with open("results/RULES_{0}.txt".format(data_name), "a") as output:
            output.write("\n\nClassifying {0} test data we obtained accuracy = {1:.4f}% and a total of {2} unclassified"
                         " instances.".format(data_name, accuracy, n_unclassified))
