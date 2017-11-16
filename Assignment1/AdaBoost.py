from sklearn.ensemble import AdaBoostClassifier
from sklearn.cross_validation import KFold
import numpy as np
from load_data import loadCSVfile

features = loadCSVfile('traindata.csv', 'float')
target = np.ravel(loadCSVfile('trainlabel.csv', 'float'))

# Normalization
# Subtract the mean for each feature
features -= np.mean(features, axis=0)
# Divide each feature by its standard deviation
features /= np.std(features, axis=0)

# Cross Validation
folds = 10
kf = KFold(n=len(target), n_folds=folds, shuffle=False)

cv = 0
mean_error = 0
estimatorNumber = 100
for tr, tst in kf:

    # Train Test Split
    tr_features = features[tr, :]
    tr_target = target[tr]

    tst_features = features[tst, :]
    tst_target = target[tst]

    # Training AdaBoost Model
    model = AdaBoostClassifier(n_estimators=estimatorNumber)
    model.fit(tr_features, tr_target)
    # Measuring training and test accuracy
    tr_accuracy = np.mean(model.predict(tr_features) == tr_target)
    tst_accuracy = np.mean(model.predict(tst_features) == tst_target)
    mean_error = mean_error + tst_accuracy
    print("%d Fold Train Accuracy:%f, Test Accuracy:%f" % (cv, tr_accuracy, tst_accuracy))
    cv += 1
    # print(tr_accuracy, tst_accuracy)
mean_error = mean_error / folds
print("AdaBoostClassifier with %s estimators %s folds CV with accuracy: %s" % (estimatorNumber, folds, mean_error))


