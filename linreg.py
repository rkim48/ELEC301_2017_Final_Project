import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('TrainingMetadata.csv')
unlabeled_dataset = pd.read_csv('UnlabeledTestMetadata.csv')
X = dataset.drop(['Type','Number'],axis = 1)
y = dataset.iloc[:,1].values
X_final = unlabeled_dataset.drop(['Number'], axis = 1)

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, \
                                                    random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

from sklearn.metrics import r2_score
score = r2_score(y_test, y_pred)
print(score)

# Score was 0.167 which is pretty bad
