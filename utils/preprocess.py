import pandas as pd
from scipy.io.arff import loadarff
from sklearn.model_selection import train_test_split
import numpy as np


def read_file(data_file, arff=None, id_column=None, class_column=None, discretize_integers=None):
    """
    :param data_file: path of the data file we want to read and preprocess
    :param arff: True if data file is an .arff file, None in other case
    :param id_column: number of the column if the data contains a column of type "id_number", None in other case
    :param class_column: number of the column corresponding to the class, None if it is the last one
    :param discretize_integers: True if we want to discretize integer values present in the data, False in other case
    :return: train, validation and test preprocessed dataframes, with no column corresponding to id
             neither missing values and all columns of type categorical # TODO: write percentages
    """

    # Load data from external file
    if arff:
        data = loadarff(data_file)
        df = pd.DataFrame(data[0])
        df = df.stack().str.decode('utf-8').unstack()
    else:
        df = pd.read_csv(data_file, header=None)    # , na_values=missing_values)

    # With 'preprocess_df' we remove missing values, the column of id number and set the Class column at last position
    df = preprocess_df(df, id_column=id_column, class_column=class_column)

    # Search for numerical columns and discretize them
    for column in df.columns[:-1]:
        column_type = df[column].dtypes
        if column_type == "float64":
            df[column] = discretize_values(df, column)
        elif column_type == "int64" and discretize_integers:
            df[column] = discretize_values(df, column)

    # Generate the dataframe with names of the columns
    num_attributes = df.shape[1] - 1
    col_names = ["Attribute_{0}".format(i + 1) for i in range(num_attributes)]
    col_names.extend(["Class"])
    df = df.set_axis(col_names, axis=1)

    # TODO: maybe we could do the train_test_split VALIDATION!
    train, test = train_test_split(df, test_size=0.15, random_state=0, shuffle=True, stratify=df["Class"])
    # train, val = train_test_split(train, test_size=0., random_state=0, shuffle=True, stratify=df["Class"])

    return train, test


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

    # Delete the column corresponding to id number in some databases
    if id_column is not None:
        df = df.drop(df.columns[[id_column]], axis=1)

    # Delete instances with missing values, if any
    df.replace({"?": np.nan}, inplace=True)
    df.dropna(inplace=True)

    return df


def discretize_values(df, column):
    """
    :param df: dataframe of which we want to discretize its columns
    :param column: column of type "float64" that will be discretized
    :return: a discretized version of that column, with 4 possible values corresponding to the four quantiles
    """
    bins = np.linspace(min(df[column]), max(df[column]), num=5)
    labels = ["Quartile_1", "Quartile_2", "Quartile_3", "Quartile_4"]
    discretized_column = pd.cut(df[column], bins=bins, labels=labels, include_lowest=True)

    return discretized_column


