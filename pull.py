from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
#load data from sklearn
cancer = load_breast_cancer()

dataframe = pd.DataFrame(data= np.c_[cancer['data'], cancer['target']],
                     columns= cancer['feature_names'].tolist()  + ['target'])

#save data
dataframe.to_csv('data/breast_cancer.csv', index = False)