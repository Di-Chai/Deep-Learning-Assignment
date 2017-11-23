from localPath import dataPath
import numpy as np
import matplotlib.pylab as plt
from skimage import transform
from scipy import misc
with open(dataPath + 'train.txt', 'r') as f:
    trainFile = f.readlines()
with open(dataPath + 'val.txt', 'r') as f:
    valFile = f.readlines()
with open(dataPath + 'test.txt', 'r') as f:
    testFile = f.readlines()

picSize = (100, 100)

trainingLabel = np.array([int(e.strip().split()[1]) for e in trainFile], dtype=np.int32)
trainFile = [e.strip().split()[0] for e in trainFile]
valLabel = np.array([int(e.strip().split()[1]) for e in valFile], dtype=np.int32)
valFile = [e.strip().split()[0] for e in valFile]

testFile = [e.strip() for e in testFile]

trainingData = []
trainDataShape = []
for file in trainFile:
    imagePath = dataPath + file[2:]
    im = plt.imread(dataPath + file[2:])
    dst = transform.resize(im, picSize)
    trainingData.append(dst)
trainingData = np.array(trainingData, dtype=np.float32)

valData = []
for file in valFile:
    imagePath = dataPath + file[2:]
    im = plt.imread(dataPath + file[2:])
    dst = transform.resize(im, picSize)
    valData.append(dst)
valData = np.array(valData, dtype=np.float32)

testData = []
for file in testFile:
    imagePath = dataPath + file[2:]
    im = plt.imread(dataPath + file[2:])
    dst = transform.resize(im, picSize)
    testData.append(dst)
testData = np.array(testData, dtype=np.float32)