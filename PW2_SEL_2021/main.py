from PW2_SEL_2021.preprocess import read_file_generate_lists
from PW2_SEL_2021.decision_forest import decision_forests
from PW2_SEL_2021.random_forest import random_forests


for size in ["small", "medium", "large"]:
    if size == "small":
        data_file = "data/new-thyroid.data"
        data_name = "new-thyroid"
        id_column = None
        class_column = 0
    elif size == "medium":
        data_file = "data/data_banknote_authentication.txt"
        data_name = "data_banknote_authentication"
        id_column = None
        class_column = None
    else:
        data_file = "data/segment.dat"
        data_name = "segment"
        id_column = None
        class_column = None

    train_list, test_list = read_file_generate_lists(data_file, id_column=id_column, class_column=class_column)

    print("data: ", data_name)

    # Train and test decision forests
    df_accuracies_df, df_times_df, df_feat_importances_df = decision_forests(train_list, test_list, min_size_node=3)
    df_accuracies_df.to_csv("results/{}_accuracies__decision_forests.csv".format(data_name))
    df_times_df.to_csv("results/{}_times__decision_forests.csv".format(data_name))
    df_feat_importances_df.to_csv("results/{}_feature_importances__decision_forests.csv".format(data_name))

    # Train and test random forests
    df_accuracies_rf, df_times_rf, df_feat_importances_rf = random_forests(train_list, test_list, min_size_node=3)
    df_accuracies_rf.to_csv("results/{}_accuracies__random_forests.csv".format(data_name))
    df_times_df.to_csv("results/{}_times__random_forests.csv".format(data_name))
    df_feat_importances_rf.to_csv("results/{}_feature_importances__random_forests.csv".format(data_name))
