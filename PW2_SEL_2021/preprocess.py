import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


def read_file_generate_lists(data_file, id_column=None, class_column=None):
    """
    :param data_file: path of the data file we want to read and preprocess
    :param id_column: number of the column if the data contains a column of type "id_number", None in other case
    :param class_column: number of the column corresponding to the class, None if it is the last one (default)
    :return: train (70%) and test (30%) preprocessed dataframes, with no column corresponding to id
             neither missing values
    """
    # Load data from external file
    df = pd.read_csv(data_file, header=None)

    # With 'preprocess_df' we remove missing values, the column of id number and set the Class column at last position
    df = preprocess_df(df, id_column=id_column, class_column=class_column)

    # Generate the dataframe with generic names of the columns
    num_attributes = df.shape[1] - 1
    col_names = ["Attribute_{0}".format(i + 1) for i in range(num_attributes)]
    col_names.extend(["Class"])
    df = df.set_axis(col_names, axis=1)

    # Generate train and test dataframes preserving proportions of each class
    train, test = train_test_split(df, test_size=0.3, random_state=0, shuffle=True, stratify=df["Class"])

    train_list = train.values.tolist()
    test_list = test.values.tolist()

    return train_list, test_list


def preprocess_df(df, id_column, class_column):
    """
    :param df: dataframe loaded from external data file
    :param id_column: number of the column if the data contains a column of type "id_number", None in other case
    :param class_column: number of the column corresponding to the class, None if it is the last one
    :return: preprocessed version of the initial dataframe
    """
    # If class column is not the last one, we move it to the last position
    if class_column is not None:
        aux = df[class_column]
        df.drop(labels=class_column, axis=1, inplace=True)
        df.insert(len(df.columns), "Class", aux)

    # Delete the column corresponding to id number existing in some databases
    if id_column is not None:
        df = df.drop(df.columns[[id_column]], axis=1)

    # Delete instances with missing values
    df.replace({"?": np.nan}, inplace=True)
    df.dropna(inplace=True)

    return df
