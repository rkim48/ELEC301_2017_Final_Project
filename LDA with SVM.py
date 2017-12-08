import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import glob
import matplotlib.image as mpimg
import matplotlib.colors as mpc
import colorgram
import PIL.Image 

# Create color_list and color_list_test which contain top 6 colors for images
color_list = [colorgram.extract(file,6) for file in sorted(\
              glob.glob(r'C:\Users\Robin\Desktop\ELEC301\TrainingImages\*.png'), key = lambda name: int(name[46:-4]))]    
color_list_test = [colorgram.extract(file,6) for file in sorted(glob.glob(r'C:\Users\Robin\Desktop\ELEC301\TestImages\*.png'),
                                  key=lambda name: int(name[42:-4]))]    

# 4 sets of RGB values (12 columns) appended to two_colors and two_colors_test
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

# Size feature vectors for training and test
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

# X is training data with stats, colors, size and X_unlabeled is test data
dataset = pd.read_csv(r'C:\Users\Robin\Desktop\ELEC301\TrainingMetadata.csv')
unlabeled_dataset = pd.read_csv(r'C:\Users\Robin\Desktop\ELEC301\UnlabeledTestMetadata.csv')
Daniel_features = pd.read_csv(r'C:\Users\Robin\Desktop\ELEC301\fast4features.csv')
Daniel_features_test = pd.read_csv(r'C:\Users\Robin\Desktop\ELEC301\fastfortest.csv')
X = np.hstack((dataset.drop(['Type','Number'],axis = 1), two_colors, size, Daniel_features))
y = dataset.iloc[:,1].values
X_unlabeled = np.hstack((unlabeled_dataset.drop(["Number"],axis = 1), two_colors_test, size_test, Daniel_features_test))

# Scale X and X_unlabeled. Notice X_sc is sc.fit_transform(X) and X_unlabeled
# is just sc.transform(X_unlabeled)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_sc = sc.fit_transform(X)
X_unlabeled = sc.transform(X_unlabeled)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

#n_range = range(1,25)
#n_scores = []
#for n in n_range:
#    lda = LDA(n_components = n)
#    X = lda.fit_transform(X_sc, y)
#    classifier = SVC(kernel = 'linear', C =1, random_state = 1)
#    classifier.fit(X, y)
#    scores = cross_val_score(classifier, X, y, cv = 10,\
#                             scoring = 'accuracy')
#    n_scores.append(scores.mean())

# Apply LDA on X_sc. Apply LDA on X_unlabeled. Learn parameters from the recently
# dimensionality reduced X and untouched y using linear kernel SVM
lda = LDA(n_components = 16)
X = lda.fit_transform(X_sc, y)
X_test = lda.transform(X_unlabeled)
classifier = SVC(kernel = 'linear', random_state = 1)
classifier.fit(X,y)    

#print(n_scores)
#plt.plot(n_range, n_scores)
#plt.xlabel('Value of n for LDA')
#plt.ylabel('Accuracy')
#
#print('Best value of n is', np.argmax(n_scores), 'with an accuracy of ', max(n_scores))

# Use parameters learned from above to predict types for X_test, unseen data
y_submission = classifier.predict(X_test)
np.savetxt("dethroned_svm.csv", y_submission,delimiter=",")

# Best is 41.57% with n = 14