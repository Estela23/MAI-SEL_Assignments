from utils.preprocess import read_file
from train_RULES import train_RULES


# data_file = "data/ballons/yellow-small+adult-stretch.data"
# data_name = "yellow-small+adult-stretch"
data_file = "data/splice.arff"
data_name = "splice"

train_df, test_df = read_file(data_file, arff=True)

# TODO: preprocess the data, missing values, discretize, etc. En vez de meterle a train_RULES el path
#       le meteríamos el dataframe ya preprocesado y habría que cambiar train_RULES

rules = train_RULES(df=df, data_name=data_name)

