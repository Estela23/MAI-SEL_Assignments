import numpy as np


def test_RULES(df, inferred_rules):
    # Apply the function classifier to each row and create a new column 'Predicted_Class'
    # which is None in the case that the classifier is not able to classify an instance

    """    predicted = []
    for index, row in df.iterrows():
        aux_pred = classifier(row, inferred_rules)
        predicted.append(aux_pred)"""

    df["Predicted_Class"] = [classifier(row, inferred_rules) for index, row in df.iterrows()]

    return df


def classifier(row, inferred_rules):
    predicted_class = None
    # TODO: coverage, seguro que el coverage se mira en el test?
    # coverage = np.zeros(len(inferred_rules))

    for idx, rule in enumerate(inferred_rules):
        # Check all the conditions except for the "Class" (which we are trying to predict): if the values in the current
        # row and in the rule match for all the selectors (len(check_list) == len(rule) - 1) then we classify the row
        check_list = [True for i in range(len(rule) - 1) if row[rule[i][0]] == rule[i][1]]
        if len(check_list) == len(rule) - 1:
            predicted_class = rule[-1][1]
            # coverage[idx] += 1
            break   # This way if we classify an instance we keep going, we do not need to try other rules

    return predicted_class  # , coverage
