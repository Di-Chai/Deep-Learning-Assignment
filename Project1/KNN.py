from load_data import loadCSVfile
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold

features = loadCSVfile('traindata.csv', 'float')
target = np.ravel(loadCSVfile('trainlabel.csv', 'float'))

# Normalization
# Subtract the mean for each feature
features -= np.mean(features, axis=0)
# Divide each feature by its standard deviation
features /= np.std(features, axis=0)


# k = int(input('k = '))
k = 5
for k in range(1, 21):
    # 5 Fold Cross Validation
    folds = 10
    kf = KFold(n=len(target), n_folds=folds, shuffle=False)

    cv = 0
    mean_error = 0
    for tr, tst in kf:

        # Train Test Split
        tr_features = features[tr, :]
        tr_target = target[tr]

        tst_features = features[tst, :]
        tst_target = target[tst]

        # Training SVM Model
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(tr_features, tr_target)
        # Measuring training and test accuracy
        tr_result = model.predict(tr_features)
        tst_result = model.predict(tst_features)
        tr_accuracy = np.mean(model.predict(tr_features) == tr_target)
        tst_accuracy = np.mean(model.predict(tst_features) == tst_target)
        mean_error = mean_error + tst_accuracy
        # print("%d Fold Train Accuracy:%f, Test Accuracy:%f" % (cv, tr_accuracy, tst_accuracy))
        cv += 1
    mean_error = mean_error / folds
    print("%s NeighborsClassifier, %s folds CV with accuracy: %s" % (k, folds, mean_error))
