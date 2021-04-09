import pandas as pd
import numpy as np
import time
import itertools
from utils.utils_train import is_unique, create_rslt_df, update_unclassified_df, check_redundant, create_rule


def train_RULES(df, data_name):
    """
    :param df: dataframe from which we will infer the rules of the rule-based system RULES
    :param data_name: name of the data we are training the model on, to create an external files with
                      the rules displayed in an interpretable way
    :return: a list of the rules inferred from df
    """
    # Create the .txt file where the rules will be
    with open("results/RULES_{0}.txt".format(data_name), "w") as output:
        output.write("File of rules corresponding to the {0} dataset \n \n".format(data_name))

        # Initial time
        t_0 = time.time()

        # Check how many attributes the data has
        num_attributes = df.shape[1] - 1
        col_names = list(df.columns)

        # Check how many rows the train data has
        n_rows = df.shape[0]

        # Create an empty list of rules
        rules = []

        # Create rules with 1 selector exploring all the attributes and all their possible values
        for column in df.columns[:-1]:
            for value in df[column].unique():
                # Generate the rslt_df of all the instances that takes 'value' in 'column', check if Class is the same
                rslt_df = df[df[column] == value]
                valid_rule, valid_class = is_unique(rslt_df["Class"])
                # Create the rule of length 1
                if valid_rule:
                    # Add the valid rule to the list of rules, write its coverage
                    rules.append([tuple([column, value]), tuple(["Class", valid_class])])
                    output.write("IF {0} = {1} THEN Class = {2} ---> Coverage {3:.4f}% \n"
                                 .format(column, value, valid_class, rslt_df.shape[0] * 100 / n_rows))

        # Generate unclassified dataframe consisting of only the non-classified instances
        unclassified = update_unclassified_df(df, rules)

        # Create rules with more than 1 selector and less than the number of attributes
        for i in range(2, num_attributes):
            # new_rules: list of rules of length i, to later update the dataframe of unclassified instances
            new_rules = []

            # all_combinations: list of possible combinations of values with length i
            all_combinations = []

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
            # previous rules, in that case add it to the list 'rules' and compute its coverage
            for idx, columns in enumerate(indexes):
                for combination in all_combinations[idx]:
                    rslt_df = create_rslt_df(df, col_names, columns, combination)
                    if not rslt_df.empty:
                        valid_rule, valid_class = is_unique(rslt_df["Class"])
                        if valid_rule:
                            aux_rule = create_rule(col_names, columns, combination, valid_class)
                            is_redundant = check_redundant(rules, aux_rule)
                            if not is_redundant:
                                wr_rule = [str(aux_rule[i][0]) + " = " + str(aux_rule[i][1]) for i in range(len(aux_rule))]
                                output.write("IF {0} THEN Class = {1} ---> Coverage {2:.4f}% \n"
                                             .format(" and ".join(wr_rule[:-1]), valid_class, rslt_df.shape[0] * 100 / n_rows))
                                rules.append(aux_rule)
                                new_rules.append(aux_rule)

            # After creating all valid rules of length i, we update the dataframe of unclassified instances with respect
            # to the rules obtained in this iteration, since it is not necessary to compare against previous rules
            unclassified = update_unclassified_df(unclassified, new_rules)

            # We stop looking for rules if we have already classified all the instances in the dataset
            if unclassified.empty:
                # Final training time
                t_1 = time.time()
                print("All the instances in the train set are correctly classified with {0} rules of 100% precision."
                      .format(len(rules)))
                output.write("\nAll the instances in the train set are correctly classified with {0} rules of 100% "
                             "precision. \nTraining time = {1:.8f} seconds.".format(len(rules), t_1 - t_0))
                break

        # Create rules from the unclassified examples with the maximum selectors possible = number of attributes
        if not unclassified.empty:
            for index, instance in unclassified.iterrows():
                aux_rule = list(zip(unclassified.columns, instance))
                rule = [str(aux_rule[i][0]) + " = " + str(aux_rule[i][1]) for i in range(len(aux_rule))]
                output.write("IF {0} THEN {1} ---> Coverage = {2:.4f}% \n"
                             .format(" and ".join(rule[:-1]), rule[-1], 1 * 100 / n_rows))
                rules.append(aux_rule)

            # Final training time
            t_1 = time.time()
            print("All instances in the train set are correctly classified with {0} rules of 100% 'precision', "
                  "STOP.".format(len(rules)))
            output.write("\nAll instances in the train set are correctly classified with {0} rules of 100% "
                         "precision. \nTraining time = {1:.8f} seconds.".format(len(rules), t_1 - t_0))

    return rules
