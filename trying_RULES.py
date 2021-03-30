import pandas as pd
import numpy as np
import itertools

"""data = open("data/ballons/adult+stretch.data", "r")
data = (data.read()).split("\n")"""

df = pd.read_csv("data/ballons/adult+stretch.data", header=None)

num_attributes = df.shape[1] - 1

col_names = ["Attribute_{0}".format(i+1) for i in range(num_attributes)]
col_names.extend(["Class"])
df = df.set_axis(col_names, axis=1)
print(df)


values = []
for attrib in range(num_attributes):
    this_values = np.unique(np.array(df.iloc[:, attrib]))
    values.append(list(this_values))
print(values)

train_instances = df.copy()

all_indexes = []
all_combinations = []

for i in range(2, num_attributes):
    print("Candidate rules of length {0}:".format(i))
    indexes = list(itertools.combinations(range(num_attributes), i))
    all_indexes.extend(indexes)
    if i == 2:
        for idx in indexes:
            combin = list(itertools.product(values[idx[0]], values[idx[1]]))  # Reglas de longitud 2
            # combin = [list(itertools.product(x)) for x in [values[i] for i in idx]]
            # combin = list(itertools.product(tuple([values[idx[i]] for i in idx])))
            # combin = list(itertools.product(values[idx[:len(idx)]]))
            all_combinations.append(combin)
    if i == 3:
        for idx in indexes:
            combin = list(itertools.product(values[idx[0]], values[idx[1]], values[idx[2]]))  # Reglas de longitud 3
            all_combinations.append(combin)
    if i == 4:
        for idx in indexes:
            combin = list(itertools.product(values[idx[0]], values[idx[1]], values[idx[2]], values[idx[3]]))  # L = 4
            all_combinations.append(combin)

print("Hello")
