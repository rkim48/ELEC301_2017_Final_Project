import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv

dataset = pd.read_csv(r'C:\Users\Robin\Desktop\ELEC301\TrainingMetadata.csv')
unlabeled_dataset = pd.read_csv(r'C:\Users\Robin\Desktop\ELEC301\UnlabeledTestMetadata.csv')

X = dataset.drop(['Type','Number'],axis = 1)
y = dataset.iloc[:,1].values

X_unlabeled = unlabeled_dataset.drop(["Number"],axis = 1)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_scaled = sc.fit_transform(X)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(max_depth = 3, random_state = 0)
classifier.fit(X_scaled, y)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_scaled, \
                                 y = y, cv = 10)

print('Mean accuracy for random forest is', accuracies.mean())
print('Standard deviation for random forest is', accuracies.std())

y_submission = classifier.predict(X_unlabeled)
np.savetxt("submission_2.csv", y_submission,delimiter=",")