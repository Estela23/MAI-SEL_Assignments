import pandas as pd


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
