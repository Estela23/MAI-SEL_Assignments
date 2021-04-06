from utils.preprocess import read_file
from train_RULES import train_RULES
import numpy as np


# data_file = "data/ballons/yellow-small+adult-stretch.data"
# data_name = "yellow-small+adult-stretch"
"""data_file = "data/splice.arff"
data_name = "splice"""
"""data_file = "data/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
data_name = "breast-cancer-wisconsin"""
"""data_file = "data/breast-cancer-wisconsin/wpbc.data"  # arff=False, id_column=0, class_column=1
data_name = "wpbc"""

data_file = "data/new-thyroid/new-thyroid.data"
data_name = "new-thyroid"

train_df, test_df = read_file(data_file, arff=False, id_column=None, class_column=0, discretize_integers=True)

rules = train_RULES(df=train_df, data_name=data_name)

test_RULES(df=test_df, inferred_rules=rules, data_name=data_name)
