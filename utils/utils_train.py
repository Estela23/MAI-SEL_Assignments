import pandas as pd


def is_unique(s):
    """
    :param s: column "Class" of a dataframe
    :return: True if all the elements of s are the same, False in other case
             first value of the column, which will be the valid_class if we obtain True
    """
    a = s.to_numpy()
    return (a[0] == a).all(), a[0]


def create_rslt_df(df, col_names, columns, combination):
    """
    :param df: original dataframe with all the instances in the database
    :param col_names: columns names of the dataframe
    :param columns: columns that this candidate rule makes reference to
    :param combination: values of the relevant attributes in the current candidate rule
    :return: resultant dataframe with only the instances that take the values of combination
    """
    # Recursively create rslt_df
    rslt_df = df.copy()
    for i in range(len(combination)):
        rslt_df = rslt_df[rslt_df[col_names[columns[i]]] == combination[i]]

    return rslt_df


def update_unclassified_df(df, rules):
    """
    :param df: dataframe of instances
    :param rules: rules with respect to which we want to update the dataframe of unclassified instances
    :return: updated dataframe with only unclassified instances wrt. rules
    """
    for rule in rules:
        aux_df = df.copy()
        for j in range(len(rule) - 1):
            aux_df = aux_df[(aux_df[rule[j][0]] == rule[j][1])]

        aux2_df = pd.merge(df, aux_df, how="outer", indicator=True)
        df = aux2_df.loc[aux2_df["_merge"] == "left_only", aux2_df.columns[:-1]]

    return df


def check_redundant(rules, aux_rule):
    """
    :param rules: complete list of the already chosen rules
    :param aux_rule: current rule, which we aim to find if it is redundant (unnecessary)
    :return: boolean variable, True if current rule is redundant
    """
    redundant = False
    for rule in rules:
        result = all(elem in aux_rule for elem in rule)
        if result:
            redundant = True
            break

    return redundant


def create_rule(col_names, columns, combination, valid_class):
    """
    :param col_names: columns names of the dataframe
    :param columns: columns that this candidate rule makes reference to
    :param combination: values of the relevant attributes in the current candidate rule
    :param valid_class: class of the instances that follow this rule
    :return: two different representations of the candidate rule, one to store and the other to print
    """
    attributes = [col_names[columns[i]] for i in range(len(columns))]
    aux_rule = list(zip(attributes, combination))
    rule = [str(aux_rule[i][0]) + " = " + str(aux_rule[i][1]) for i in range(len(aux_rule))]
    write_rule = "IF {0} THEN Class = {1} \n".format(" and ".join(rule), valid_class)
    aux_rule.append(tuple(["Class", valid_class]))

    return aux_rule, write_rule
