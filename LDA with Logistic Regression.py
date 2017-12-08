import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import glob
import matplotlib.image as mpimg
import matplotlib.colors as mpc
import colorgram
import PIL.Image 

#Colorgram Test
color_list = [colorgram.extract(file,6) for file in sorted(\
              glob.glob(r'C:\Users\Robin\Desktop\ELEC301\TrainingImages\*.png'), key = lambda name: int(name[46:-4]))]      
color_list_test = [colorgram.extract(file,6) for file in sorted(glob.glob(r'C:\Users\Robin\Desktop\ELEC301\TestImages\*.png'),
                                  key=lambda name: int(name[42:-4]))]    
#two_colors = []
#for i in range(0,601):
#    for j in range(1,4):
#        two_colors.append([color_list[i][j].rgb.r, color_list[i][j].rgb.g,\
#                      color_list[i][j].rgb.b])
#two_colors = np.ravel(two_colors)/256
#two_colors = np.reshape(two_colors, [601,9])
#
two_colors = []
for i in range(0,601):
    if len(color_list[i]) < 5:
        for j in range(1,4):
            two_colors.append([color_list[i][j].rgb.r, color_list[i][j].rgb.g,\
                      color_list[i][j].rgb.b])
        two_colors.append([np.mean([color_list[i][1].rgb.r, color_list[i][2].rgb.r, color_list[i][3].rgb.r]), \
                               np.mean([color_list[i][1].rgb.g, color_list[i][2].rgb.g, color_list[i][3].rgb.g]),\
                               np.mean([color_list[i][1].rgb.b,color_list[i][2].rgb.b, color_list[i][3].rgb.b])])
    else:
        for j in range(1,5):
            two_colors.append([color_list[i][j].rgb.r, color_list[i][j].rgb.g,\
                      color_list[i][j].rgb.b])
    
two_colors = np.ravel(two_colors)/256
two_colors = np.reshape(two_colors, [601,12])

two_colors_test = []
for i in range(0,201):
    if len(color_list_test[i]) < 5:
        for j in range(1,4):
            two_colors_test.append([color_list_test[i][j].rgb.r, color_list_test[i][j].rgb.g,\
                      color_list[i][j].rgb.b])
        two_colors_test.append([np.mean([color_list_test[i][1].rgb.r, color_list_test[i][2].rgb.r, color_list_test[i][3].rgb.r]), \
                               np.mean([color_list_test[i][1].rgb.g, color_list_test[i][2].rgb.g, color_list_test[i][3].rgb.g]),\
                               np.mean([color_list_test[i][1].rgb.b,color_list_test[i][2].rgb.b, color_list_test[i][3].rgb.b])])
    else:
        for j in range(1,5):
            two_colors_test.append([color_list_test[i][j].rgb.r, color_list_test[i][j].rgb.g,\
                      color_list_test[i][j].rgb.b])
    
two_colors_test = np.ravel(two_colors_test)/256
two_colors_test = np.reshape(two_colors_test, [201,12])

size = []
for i in range(0,601):
    prop = 1 - color_list[i][0].proportion
    size.append(prop)
size = np.reshape(size, [601,1])

size_test = []
for i in range(0,201):
    prop = 1 - color_list_test[i][0].proportion
    size_test.append(prop)
size_test = np.reshape(size_test, [201,1])

dataset = pd.read_csv(r'C:\Users\Robin\Desktop\ELEC301\TrainingMetadata.csv')
unlabeled_dataset = pd.read_csv(r'C:\Users\Robin\Desktop\ELEC301\UnlabeledTestMetadata.csv')
Daniel_features = pd.read_csv(r'C:\Users\Robin\Desktop\ELEC301\fast4features.csv')
Daniel_features_test = pd.read_csv(r'C:\Users\Robin\Desktop\ELEC301\fastfortest.csv')
edge = Daniel_features.iloc[:,:].values
edge_test = Daniel_features_test.iloc[:,:].values
X = np.hstack((dataset.drop(['Type','Number'],axis = 1), two_colors, size, edge))
y = dataset.iloc[:,1].values
X_unlabeled = np.hstack((unlabeled_dataset.drop(["Number"],axis = 1), two_colors_test, size_test, edge_test))

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_sc = sc.fit_transform(X)
X_unlabeled = sc.transform(X_unlabeled)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

#n_range = range(1,25)
#n_scores = []
#for n in n_range:
#    lda = LDA(n_components = n)
#    X = lda.fit_transform(X_sc, y)
#    classifier = LogisticRegression(penalty = 'l2', C = 0.7 , random_state = 5)
#    scores = cross_val_score(classifier, X, y, cv = 10, \
#                             scoring = 'accuracy')
#    n_scores.append(scores.mean())
    
#print('Best value of n is', np.argmax(n_scores), 'with an accuracy of ', max(n_scores))

# Apply LDA on X_sc. Apply LDA on X_unlabeled. Learn parameters from the recently
# dimensionality reduced X and untouched y using logistic regression
lda = LDA(n_components = 16)
X = lda.fit_transform(X_sc, y)
X_test = lda.transform(X_unlabeled)
classifier = LogisticRegression(C = 0.7, random_state = 5)
classifier.fit(X, y) 

# Best is 36.61% with n = 12, penalty = 'l1'
# Best is 47.32% with n = 15, penalty = 'L1'
# Best is 47.88% with n = 15, penalty = 'L2'
# Best is 48.2% with n = 15, penalty = 'L2' and C = 0.8
# Best is 48.32% with n = 15, C = 0.7
# Best is 64.22% with n = 14, C = 0.7, fast2features
# Best is 69.36% with n = 16, C = 0.7, fast3features
# Best is 72.25% with n = 16, C = 0.7, fast4features

# Use parameters learned from above to predict types for X_test, unseen data
y_submission = classifier.predict(X_test)
np.savetxt("dethroned.csv", y_submission, delimiter=",")