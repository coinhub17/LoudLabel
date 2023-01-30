
from sklearn.externals import joblib
from sklearn import datasets
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn import preprocessing
import numpy as np
from collections import Counter

#fetch data
dataset = datasets.fetch_mldata("MNIST Original")

features = np.array(dataset.data, 'int16') 
labels = np.array(dataset.target, 'int')

list_hog_fd = []
for feature in features:
    #reshape hog features from skimage.feature 
    fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
    list_hog_fd.append(fd)
hog_features = np.array(list_hog_fd, 'float64')
#fitting features
pp = preprocessing.StandardScaler().fit(hog_features)
hog_features = pp.transform(hog_features)

print ("digit count:", Counter(labels))

#looking for best fit
clf = LinearSVC()

clf.fit(hog_features, labels)

#store the model to 
joblib.dump((clf, pp), "digit.csv", compress=3)