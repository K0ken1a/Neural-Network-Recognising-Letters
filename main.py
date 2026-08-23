import numpy as np
import pandas as pd
from neuralNetwork import NeuralNetwork


df = pd.read_csv('A_Z Handwritten Data.csv')

data = np.array(df)
rows, cols = data.shape
np.random.shuffle(data)

testSize = rows // 5

labels = data[:, 0].astype(int)

images = data[:, 1:785].astype(np.float32) / 255.0

images = images.T

testLabels = labels[0:testSize]
trainLabels = labels[testSize:]

testImgs = images[:, 0:testSize]
trainImgs = images[:, testSize:]
trainImgs.shape

myNN = NeuralNetwork()
myNN.gradientDescent(trainImgs, trainLabels)
myNN.predict(9, testImgs, testLabels)

