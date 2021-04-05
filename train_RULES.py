import pandas as pd
import numpy as np
import itertools
from utils.create_rules import is_unique    # create_rule
from utils.update_dataframe import create_rslt_df, update_unclassified_df


def train_RULES(df, data_name):
    # Create the .txt file where the rules will be
    with open("obtained_rules/RULES_{0}.txt".format(data_name), "w") as output:
        output.write("File of rules corresponding to the {0} dataset \n \n".format(data_name))

        # Create an empty list of rules, each rule will have the form: [("Attribute_2", DIP), ("Class", T)]
        rules = []

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

        # Create rules with more than 1 selector and less than the number of attributes
        all_combinations = []
        for i in range(2, num_attributes):
            # We update the array of possible values of the attributes according to the remaining unclassified examples
            values = []
            for attrib in range(num_attributes):
                this_values = np.unique(np.array(unclassified.iloc[:, attrib]))
                values.append(list(this_values))

            # Generate all the possible combinations of size i (selectors in the rules)
            indexes = list(itertools.combinations(range(num_attributes), i))

            # For each combination of attributes we generate all the possible combinations of their values
            for idx in indexes:
                combination = list(itertools.product(*[values[idx[i]] for i in range(len(idx))]))
                all_combinations.append(combination)

            # For each combination of values we check if it is a candidate rule, in that case add it to the list 'rules'
            for idx, columns in enumerate(indexes):
                for combination in all_combinations[idx]:
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
                print("All instances are correctly classified with {0} rules of 100% 'precision', STOP.".format(len(rules)))
                output.write("\nAll instances are correctly classified with {0} rules of 100% 'precision', STOP."
                             .format(len(rules)))
                break

        # Create rules from the unclassified examples with the maximum selectors possible = number of attributes
        if not unclassified.empty:
            for index, instance in unclassified.iterrows():
                rule = list(zip(unclassified.columns, instance))
                rules.append(rule)

        # TODO: Look for redundant (o como se llamen) rules!

    return rules
