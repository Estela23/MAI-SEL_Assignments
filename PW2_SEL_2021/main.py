import random
from PW1_SEL_2021.utils.preprocess import read_file_generate_dfs, preprocess_df


data_file = "data/breast-cancer.data"
data_name = "breast-cancer"
arff = False
id_column = None
class_column = 0
discretize_integers = []

train_df, test_df = read_file_generate_dfs(data_file, arff=arff, id_column=id_column, class_column=class_column,
                                           discretize_integers=discretize_integers)






