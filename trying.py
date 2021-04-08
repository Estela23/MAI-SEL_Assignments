from utils.preprocess import read_file_generate_dfs
from train_RULES import train_RULES
from test_RULES import test_RULES


"""data_file = "data/magic04.data"
data_name = "magic04"
arff = False
id_column = None
class_column = None
discretize_integers = []"""

data_file = "data/new-thyroid.data"
data_name = "new-thyroid"
arff = False
id_column = None
class_column = 0
discretize_integers = [1]


train_df, test_df = read_file_generate_dfs(data_file, arff=arff, id_column=id_column, class_column=class_column,
                                           discretize_integers=discretize_integers)

rules = train_RULES(df=train_df, data_name=data_name)

result_df, accuracy, n_unclassified = test_RULES(df=test_df, inferred_rules=rules)
result_df.to_csv("results/predicted_{0}.csv".format(data_name), index=False)

# Finally, we include the general results of the classifier (accuracy and number of not classified instances)
with open("results/RULES_{0}.txt".format(data_name), "a") as output:
    output.write("\n\nClassifying {0} test data we obtained accuracy = {1:.4f}% and a total of {2} unclassified"
                 " instances.".format(data_name, accuracy, n_unclassified))
