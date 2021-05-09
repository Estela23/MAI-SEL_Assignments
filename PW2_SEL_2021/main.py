import random
from PW2_SEL_2021.preprocess import read_file_generate_lists
from PW2_SEL_2021.decision_forest import decision_forests
from PW2_SEL_2021.random_forest import random_forests

"""data_file = "data/new-thyroid.data"
data_name = "new-thyroid"
id_column = None
class_column = 0"""

"""data_file = "data/data_banknote_authentication.txt"
data_name = "data_banknote_authentication"
id_column = None
class_column = None"""

data_file = "data/segment.dat"
data_name = "segment"
id_column = None
class_column = None


train_list, test_list = read_file_generate_lists(data_file, id_column=id_column, class_column=class_column)

print("data: ", data_name)

df_accuracies_df, df_feat_importances_df = decision_forests(train_list, test_list)
df_accuracies_df.to_csv("results/{}_accuracies__decision_forests.csv".format(data_name))
df_feat_importances_df.to_csv("results/{}_feature_importances__decision_forests.csv".format(data_name))

"""
df_accuracies_rf = random_forests(train_list, test_list)
df_accuracies_rf.to_csv("results/{}_random_forests.csv".format(data_name))
"""





