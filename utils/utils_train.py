import pandas as pd


def is_unique(s):
    a = s.to_numpy()
    return (a[0] == a).all(), a[0]


def create_rslt_df(df, col_names, columns, values):
    # Recursively create rslt_df
    rslt_df = df.copy()
    for i in range(len(values)):
        rslt_df = rslt_df[rslt_df[col_names[columns[i]]] == values[i]]

    return rslt_df


def update_unclassified_df(df, rules):
    unclassified = df.copy()
    for rule in rules:
        aux_df = unclassified.loc[unclassified[rule[0][0]] == rule[0][1]]
        aux2_df = pd.merge(unclassified, aux_df, how="outer", indicator=True)
        unclassified = aux2_df.loc[aux2_df["_merge"] == "left_only", aux2_df.columns[:-1]]

    return unclassified


def check_redundant(rules, aux_rule):
    redundant = False
    for rule in rules:
        result = all(elem in aux_rule for elem in rule)
        if result:
            redundant = True
            break

    return redundant


def create_rule(col_names, columns, combination, valid_class):
    attributes = [col_names[columns[i]] for i in range(len(columns))]
    aux_rule = list(zip(attributes, combination))
    rule = [str(aux_rule[i][0]) + " = " + str(aux_rule[i][1]) for i in range(len(aux_rule))]
    write_rule = "IF {0} THEN Class = {1} \n".format(" and ".join(rule), valid_class)
    aux_rule.append(tuple(["Class", valid_class]))

    return aux_rule, write_rule
