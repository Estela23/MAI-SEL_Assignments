import pandas as pd
import numpy as np
import itertools
from utils.utils_train import is_unique, create_rslt_df, update_unclassified_df, check_redundant, create_rule


def train_RULES(df, data_name):
    # Create the .txt file where the rules will be
    with open("obtained_rules/RULES_{0}.txt".format(data_name), "w") as output:
        output.write("File of rules corresponding to the {0} dataset \n \n".format(data_name))

        # Create an empty list of rules, each rule will have the form: [("Attribute_2", DIP), ("Class", T)]
        rules = []

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

        # Check how many attributes it has
        num_attributes = df.shape[1] - 1
        col_names = list(df.columns)

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

            # For each combination of values we check if it is a candidate rule and it is not irrelevant against all
            # previous rules, in that case add it to the list 'rules'
            for idx, columns in enumerate(indexes):
                for combination in all_combinations[idx]:
                    rslt_df = create_rslt_df(df, col_names, columns, combination)
                    if not rslt_df.empty:
                        valid_rule, valid_class = is_unique(rslt_df["Class"])
                        # Create the rule of length i
                        if valid_rule:
                            aux_rule, write_rule = create_rule(col_names, columns, combination, valid_class)
                            is_redundant = check_redundant(rules, aux_rule)
                            if not is_redundant:
                                output.write(write_rule)
                                rules.append(aux_rule)
            unclassified = update_unclassified_df(unclassified, rules)

            # We stop if we have already classified all the instances in the dataset
            if unclassified.empty:
                print("All instances in the train set are correctly classified with {0} rules of 100% 'precision', "
                      "STOP.".format(len(rules)))
                output.write("\nAll instances in the train set are correctly classified with {0} rules of 100% "
                             "'precision', STOP."
                             .format(len(rules)))
                break

        # Create rules from the unclassified examples with the maximum selectors possible = number of attributes
        if not unclassified.empty:
            for index, instance in unclassified.iterrows():
                aux_rule = list(zip(unclassified.columns, instance))
                rule = [str(aux_rule[i][0]) + " = " + str(aux_rule[i][1]) for i in range(len(aux_rule))]
                output.write("IF {0} THEN {1} \n".format(" and ".join(rule[:-1]), rule[-1]))
                rules.append(aux_rule)

            print("All instances in the train set are correctly classified with {0} rules of 100% 'precision', "
                  "STOP.".format(len(rules)))
            output.write("\nAll instances in the train set are correctly classified with {0} rules of 100% "
                         "'precision', STOP."
                         .format(len(rules)))

    return rules
