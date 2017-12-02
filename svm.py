import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r'C:\Users\Robin\Desktop\ELEC301\TrainingMetadata.csv')
unlabeled_dataset = pd.read_csv(r'C:\Users\Robin\Desktop\ELEC301\UnlabeledTestMetadata.csv')
X = dataset.drop(['Type','Number'],axis = 1)
y = dataset.iloc[:,1].values

from sklearn.svm import SVC
classifier_lin = SVC(kernel = 'linear', random_state = 0)
classifier_lin.fit(X, y)
classifier_rbf = SVC(kernel = 'rbf', random_state = 0 )
classifier_rbf.fit(X, y)
classifier_poly = SVC(kernel = 'poly', random_state = 0 )
classifier_poly.fit(X, y)

from sklearn.model_selection import cross_val_score
accuracies_lin = cross_val_score(estimator = classifier_lin, X = X, \
                                 y = y, cv = 10)
accuracies_rbf = cross_val_score(estimator = classifier_rbf, X = X, \
                                 y = y, cv = 10)
accuracies_poly = cross_val_score(estimator = classifier_poly, X = X, \
                                 y = y, cv = 10)

print('Mean accuracy for linear SVM is ', accuracies_lin.mean())
print('Standard deviation for linear SVM is', accuracies_lin.std())
print('Mean accuracy for Gaussian SVM is ', accuracies_rbf.mean())
print('Standard deviation for Gaussian SVM is ', accuracies_rbf.std())
print('Mean accuracy for polynomial SVM is ', accuracies_poly.mean())
print('Standard deviation for polynomial SVM is ', accuracies_poly.std())


