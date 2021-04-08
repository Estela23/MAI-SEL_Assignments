import numpy as np


def test_RULES(df, inferred_rules):
    """
    :param df: dataframe to which apply the rule-based system RULES to classify its instances
    :param inferred_rules: list of rules used to classify new instances, inferred from the train data
    :return: df: original dataframe with an extracolumn of the 'Predicted Class'
             accuracy: accuracy of the rule-based system over the test data
             n_unclassified: number of instances in the test dataset that we could not classify with the inferred rules
    """
    # Apply the function classifier to each row and create a new column 'Predicted_Class'
    # which is None in the case that the classifier is not able to classify an instance
    df["Predicted_Class"] = [classifier(row, inferred_rules) for index, row in df.iterrows()]
    # Compute the accuracy by counting the number of instances with Class = Predicted_Class
    matches = sum(np.where(df["Class"] == df["Predicted_Class"], True, False))
    accuracy = matches * 100 / df.shape[0]
    # Compute the number of unclassified instances
    n_unclassified = df["Predicted_Class"].isna().sum()
    return df, accuracy, n_unclassified


def classifier(row, inferred_rules):
    """
    :param row: instance of a dataframe we are trying to classify
    :param inferred_rules: list of rules generated based on the training set
    :return: predicted class of the current instance if possible, if we can not classify the instance return None
    """
    predicted_class = None

    for idx, rule in enumerate(inferred_rules):
        # Check all the conditions except for the "Class" (which we are trying to predict): if the values in the current
        # row and in the rule match for all the selectors (len(check_list) == len(rule) - 1) then we classify the row
        check_list = [True for i in range(len(rule) - 1) if row[rule[i][0]] == rule[i][1]]
        if len(check_list) == len(rule) - 1:
            predicted_class = rule[-1][1]
            break   # This way if we classify an instance we keep going with the others, don't need to try other rules

    return predicted_class
