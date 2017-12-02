import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r'C:\Users\Robin\Desktop\ELEC301\TrainingMetadata.csv')
unlabeled_dataset = pd.read_csv(r'C:\Users\Robin\Desktop\ELEC301\UnlabeledTestMetadata.csv')

X = dataset.drop(['Type','Number'],axis = 1)
y = dataset.iloc[:,1].values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_scaled = sc.fit_transform(X)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
k_range = range(1,30)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    scores = cross_val_score(knn, X_scaled, y, cv = 10, \
                             scoring = 'accuracy')
    k_scores.append(scores.mean())
    
print(k_scores)
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Accuracy')

print('Best value of K is', np.argmax(k_scores), 'with an accuracy of ', max(k_scores))






