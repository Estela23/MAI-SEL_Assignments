import pandas as pd
import numpy as np
import itertools
from utils.create_rules import is_unique    # create_rule
from utils.update_dataframe import create_rslt_df, update_unclassified_df

# Create the .txt file where the rules will be
with open("rules.txt", "w") as output:
    output.write("File of rules corresponding to the ballons adult+stretch dataset \n \n")

    # Create an empty list of rules, each rule will have the form: [("Attribute_2", DIP), ("Class", T)]
    rules = []
    # Read the csv file
    df = pd.read_csv("data/ballons/adult+stretch.data", header=None)

    # Check how many attributes it has
    num_attributes = df.shape[1] - 1

    # Create the dataframe with names of the columns
    col_names = ["Attribute_{0}".format(i+1) for i in range(num_attributes)]
    col_names.extend(["Class"])
    df = df.set_axis(col_names, axis=1)
    # TODO: check whether the first column is an ID number and replace it

    # Create rules with 1 selector
    for column in df.columns[:-1]:
        for value in df[column].unique():
            rslt_df = df[df[column] == value]
            valid_rule, valid_class = is_unique(rslt_df["Class"])
            # Create the rule of length 1
            if valid_rule:
                # Adding the valid rule to the list of rules
                rules.append([tuple([column, value]), tuple(["Class", valid_class])])
                output.write("IF {0} = {1} THEN Class = {2} \n".format(column, value, valid_class))

    # Update the dataframe with only the non-classified instances
    unclassified = update_unclassified_df(df, rules)

####################
    # all_indexes = []
    all_combinations = []

    for i in range(2, num_attributes):

        values = []
        for attrib in range(num_attributes):
            this_values = np.unique(np.array(unclassified.iloc[:, attrib]))
            values.append(list(this_values))

        indexes = list(itertools.combinations(range(num_attributes), i))
        # all_indexes.extend(indexes)

        for idx in indexes:
            combination = list(itertools.product(*[values[idx[i]] for i in range(len(idx))]))
            all_combinations.append(combination)

        for idx, columns in enumerate(indexes):
            for combination in all_combinations[idx]:
                # rslt_df = df[df[columns[j]] == values[j]]   # for j in range(len(values))]
                rslt_df = create_rslt_df(df, col_names, columns, combination)
                if not rslt_df.empty:
                    valid_rule, valid_class = is_unique(rslt_df["Class"])
                    # Create the rule of length 1
                    if valid_rule:
                        attributes = [col_names[columns[i]] for i in range(len(columns))]
                        aux_rule = list(zip(attributes, combination))
                        rule = [str(aux_rule[i][0]) + " = " + str(aux_rule[i][1]) for i in range(len(aux_rule))]
                        output.write("IF {0} THEN Class = {1} \n".format(" and ".join(rule), valid_class))
                        aux_rule.append(tuple(["Class", valid_class]))
                        rules.append(aux_rule)
        unclassified = update_unclassified_df(unclassified, rules)

        # We stop if we have already classified all the instances in the dataset
        if unclassified.empty:
            print("All the instances are correctly classified, with {0} rules of 100% 'precision', STOP.".format(len(rules)))
            output.write("\nAll the instances are correctly classified, with {0} rules of 100% 'precision', STOP.".format(len(rules)))
            break

    # TODO: las instances sin classificar se convierten en reglas!
