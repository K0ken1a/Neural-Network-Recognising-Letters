import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

def initParams():
    w1 = np.random.randn(128, 784) * np.sqrt(2.0 / 784)
    b1 = np.zeros((128,1))
    
    w2 = np.random.randn(64, 128) * np.sqrt(2.0 / 128)
    b2 = np.zeros((64,1))
    
    w3 = np.random.randn(26, 64) * np.sqrt(2.0 / 64)
    b3 = np.zeros((26,1))

    return w1, b1, w2, b2, w3, b3

def reLU(x):
    return np.maximum(0, x)

def softmax(x):
    shift_x = x - np.max(x, axis=0, keepdims=True)
    exps = np.exp(shift_x)
    return exps / np.sum(exps, axis=0, keepdims=True)

def convertLabels(y):
    y = np.array(y, dtype=int)
    
    m = np.zeros((y.size, 26))
    m[np.arange(y.size), y] = 1
    return m.T

def oneHotEnc(x):
    predictedClasses = np.argmax(x, axis=0)

    oneHot = np.zeros_like(x)
    oneHot[predictedClasses, np.arange(x.shape[1])] = 1
    return oneHot
    
def forwardProp(w1, b1, w2, b2, w3, b3, x):
    z1 = np.matmul(w1,x) + b1
    a1 = reLU(z1)

    z2 = np.matmul(w2,a1) + b2
    a2 = reLU(z2)

    z3 = np.matmul(w3,a2) + b3
    a3 = softmax(z3)

    return z1, a1, z2, a2, z3, a3

def backProp(z1, a1, z2, a2, w2, a3, w3, x, y):
    oneHot = convertLabels(y)
    m = oneHot.shape[1]
    dZ3 = a3 - oneHot
    dW3 = (1/m)*np.matmul(dZ3, a2.T)
    dB3 = np.mean(dZ3, axis=1, keepdims=True)

    da2 = np.matmul(w3.T, dZ3)
    dZ2 = da2 * (z2 > 0)
    dW2 = (1/m)*np.matmul(dZ2, a1.T)
    dB2 = np.mean(dZ2, axis=1, keepdims=True)

    da1 = np.matmul(w2.T, dZ2)
    dZ1 = da1 * (z1 > 0)
    dW1 = (1/m)*np.matmul(dZ1, x.T)
    dB1 = np.mean(dZ1, axis=1, keepdims=True)
    
    return dW1, dB1, dW2, dB2, dW3, dB3

def adjustNetwork(w1, b1, w2, b2, w3, b3, dW1, dB1, dW2, dB2, dW3, dB3, learningRate):
    w1 = w1 - learningRate * dW1
    b1 = b1 - learningRate * dB1

    w2 = w2 - learningRate * dW2
    b2 = b2 - learningRate * dB2

    w3 = w3 - learningRate * dW3
    b3 = b3 - learningRate * dB3

    return w1, b1, w2, b2, w3, b3

def getPredictions(a3):
    return np.argmax(a3, 0) if a3.ndim > 1 else np.argmax(a3)

def getAccuracy(predictions, y):
    return np.sum(predictions == y)/ y.size

def gradientDescent(x, y, epochs=10, batchSize=128, learningRate=0.1):
    w1, b1, w2, b2, w3, b3 = initParams()
    m = x.shape[1]

    for epoch in range(epochs):
        permutation = np.random.permutation(m)
        xShuffled = x[:, permutation]
        yShuffled = y[permutation]
        
        for start in range(0, m, batchSize):
            end = min(start + batchSize, m)
            
            xMini = xShuffled[:, start:end]
            yMini = yShuffled[start:end]
    
            z1, a1, z2, a2, z3, a3 = forwardProp(w1, b1, w2, b2, w3, b3, xMini)
            dW1, dB1, dW2, dB2, dW3, dB3 = backProp(z1, a1, z2, a2, w2, a3, w3, xMini, yMini)
            
            w1, b1, w2, b2, w3, b3 = adjustNetwork(w1, b1, w2, b2, w3, b3, dW1, dB1, dW2, dB2, dW3, dB3, learningRate)
            
        _, _, _, _, _, a3Full = forwardProp(w1, b1, w2, b2, w3, b3, x)
        acc = getAccuracy(getPredictions(a3Full), y)
        
        print(f"Epoch {epoch + 1}/{epochs} | Accuracy: {acc:.4f}")
        
    return w1, b1, w2, b2, w3, b3

w1, b1, w2, b2, w3, b3 = gradientDescent(trainImgs, trainLabels)

def testPrediction(index, w1, b1, w2, b2, w3, b3, yData, yLabels):
    sampleImg = yData[:, index:index+1]
    sampleLabel = yLabels[index]

    _, _, _, _, _, prediction = forwardProp(w1, b1, w2, b2, w3, b3, sampleImg)

    print("\nPrediction:", chr(getPredictions(prediction)[0] + 65))
    print("Actual Character:", chr(sampleLabel + 65))

    sampleImg = sampleImg.reshape(28, 28)

    plt.imshow(sampleImg, cmap='gray')
    plt.title(f"Label: {sampleLabel}")
    plt.axis('off')
    plt.show()

testPrediction(12, w1, b1, w2, b2, w3, b3, testImgs, testLabels)