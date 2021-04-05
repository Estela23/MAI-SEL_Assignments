import pandas as pd
from scipy.io.arff import loadarff
from sklearn.model_selection import train_test_split


def read_file(data_file, arff=None):
    if arff:
        data = loadarff(data_file)
        df = pd.DataFrame(data[0])
        df = df.stack().str.decode('utf-8').unstack()
    else:
        df = pd.read_csv(data_file, header=None)

    # TODO: maybe we could do the train_test_split with stratify! Take a look
    train, test = train_test_split(df, test_size=0.2, random_state=0, shuffle=True)

    return train, test


