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
    # Recursively create rslt_df with the instances that satisfy all the conditions
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
    # For each rule, which has already been properly filtered and chosen,
    # we delete from the dataframe df all the instances that satisfy it
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
    :param aux_rule: candidate rule, which we aim to find if it is redundant (unnecessary)
    :return: boolean variable, True if candidate rule is redundant and False otherwise
    """
    # For each of the already created rules we check whether the candidate rule is subsumed by it, in that case
    # the candidate rule is redundant and hence will not be added to the list of rules (check_redundant returns True)
    for rule in rules:
        result = all(elem in aux_rule for elem in rule)
        if result:
            return True
    return False


def create_rule(col_names, columns, combination, valid_class):
    """
    :param col_names: columns names of the dataframe
    :param columns: columns that this candidate rule makes reference to
    :param combination: values of the relevant attributes in the current candidate rule
    :param valid_class: class of the instances that follow this rule
    :return: candidate rule to store as a list of tuples (attribute, value) and (Class, class value)
    """
    attributes = [col_names[columns[i]] for i in range(len(columns))]
    aux_rule = list(zip(attributes, combination))
    aux_rule.append(tuple(["Class", valid_class]))

    return aux_rule
