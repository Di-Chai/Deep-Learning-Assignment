from sklearn.ensemble import AdaBoostClassifier
from sklearn.cross_validation import KFold
import numpy as np
from load_data import loadCSVfile
import csv

features = loadCSVfile('traindata.csv', 'float')
target = np.ravel(loadCSVfile('trainlabel.csv', 'float'))
test = loadCSVfile('testdata.csv', 'float')

# Normalization
# Subtract the mean for each feature
features -= np.mean(features, axis=0)
# Divide each feature by its standard deviation
features /= np.std(features, axis=0)

# Subtract the mean for each feature
test -= np.mean(test, axis=0)
# Divide each feature by its standard deviation
test /= np.std(test, axis=0)

# Cross Validation
folds = 10
kf = KFold(n=len(target), n_folds=folds, shuffle=False)

# Training AdaBoost Model
model = AdaBoostClassifier(n_estimators=100)
model.fit(features, target)
# Measuring training and test accuracy

test_label = list(model.predict(test))

csvFile = open('testlabel.csv', "w")
writer = csv.writer(csvFile)
for line in test_label:
    writer.writerow([int(line)])
csvFile.close()

