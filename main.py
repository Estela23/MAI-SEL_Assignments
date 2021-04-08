from utils.preprocess import read_file_generate_dfs
from train_RULES import train_RULES
from test_RULES import test_RULES

# We considered 3 datasets, one small, one medium and one large, with different types of attributes
# (categorical, numerical, ...). For each of them we properly preprocess and split the data into train and test data,
# then train a RULES classifier system to infer rules. Once we got these rules we classify the instances in the test
# data and save the classification to an external file, this was done to easily review the results.
# Finally, we evaluate the algorithm by measuring the coverage and precision of each rule.

sizes = ["small", "medium", "large"]
for size in sizes:
    if size == "small":
        data_file = "data/new-thyroid/new-thyroid.data"
        data_name = "new-thyroid"
        arff = False
        id_column = None
        class_column = 0
        discretize_integers = [1]
    elif size == "medium":
        data_file = "data/cmc.data"
        data_name = "cmc"
        arff = False
        id_column = None
        class_column = None
        discretize_integers = [0, 3]
    elif size == "large":
        data_file = "data/splice.arff"
        data_name = "splice"
        arff = True
        id_column = None
        class_column = None
        discretize_integers = None
    else:
        print("Data size must be fixed to some value, either 'small', 'medium' of 'large'.")
        break

    train_df, test_df = read_file_generate_dfs(data_file, arff=arff, id_column=id_column, class_column=class_column,
                                               discretize_integers=discretize_integers)

    rules = train_RULES(df=train_df, data_name=data_name)

    result_df = test_RULES(df=test_df, inferred_rules=rules)
    result_df.to_csv("results/predicted_{0}".format(data_name))
