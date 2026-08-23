import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork():
    def __init__(self, features = 784, layers = 3, layerNeurons = [128, 64, 26]):
        self.features = features
        self.layers = layers
        self.layerNeurons = layerNeurons

        self.network = [np.random.randn(self.layerNeurons[0], self.features) * np.sqrt(2.0 / self.features), np.zeros((self.layerNeurons[0] ,1))]

        for i in range(1, self.layers):
            self.network.extend([np.random.randn(self.layerNeurons[i], self.layerNeurons[i-1]) * np.sqrt(2.0 / self.layerNeurons[i-1]), np.zeros((self.layerNeurons[i] ,1))])

    def reLU(self, x):
        return np.maximum(0, x)

    def reLU_deriv(self, x):
        return (x > 0).astype(float)

    def softmax(self, x):
        shift_x = x - np.max(x, axis=0, keepdims=True)
        exps = np.exp(shift_x)
        return exps / np.sum(exps, axis=0, keepdims=True)

    def convertLabels(self, y):
        y = np.array(y, dtype=int)
        
        m = np.zeros((y.size, self.layerNeurons[-1]))
        m[np.arange(y.size), y] = 1
        return m.T

    def forwardProp(self, x):
        self.finalActivations = []
        prevA = x
        for i in range(self.layers):
            W, b = self.network[2*i], self.network[2*i + 1]
            z = np.matmul(W, prevA) + b
            a = self.softmax(z) if i == self.layers - 1 else self.reLU(z)
            self.finalActivations.extend([z, a])
            prevA = a

    def backProp(self, x, y):
        oneHot = self.convertLabels(y)
        m = oneHot.shape[1]
        L = self.layers

        grads = [None] * (2 * L) 

        aL = self.finalActivations[2*L - 1]
        dZ = aL - oneHot

        for i in reversed(range(L)):
            prevA = x if i == 0 else self.finalActivations[2*(i-1) + 1]

            dW = (1/m) * np.matmul(dZ, prevA.T)
            db = np.mean(dZ, axis=1, keepdims=True)
            grads[2*i] = dW
            grads[2*i + 1] = db

            if i > 0:
                W = self.network[2*i]
                z_prev = self.finalActivations[2*(i-1)]
                dA_prev = np.matmul(W.T, dZ)
                dZ = dA_prev * self.reLU_deriv(z_prev)

        return grads

    def updateParams(self, grads, lr):
        for j in range(len(self.network)):
            self.network[j] -= lr * grads[j]

    def getPredictions(self, a3):
        return np.argmax(a3, 0) if a3.ndim > 1 else np.argmax(a3)

    def getAccuracy(self, predictions, y):
        return np.sum(predictions == y)/ y.size

    def gradientDescent(self, x, y, epochs=10, batchSize=128, learningRate=0.1):
        m = x.shape[1]

        for epoch in range(epochs):
            permutation = np.random.permutation(m)
            xShuffled = x[:, permutation]
            yShuffled = y[permutation]
            
            for start in range(0, m, batchSize):
                end = min(start + batchSize, m)
                
                xMini = xShuffled[:, start:end]
                yMini = yShuffled[start:end]
        
                self.forwardProp(xMini)
                grads = self.backProp(xMini, yMini)   
                self.updateParams(grads, learningRate)
                
            self.forwardProp(x)
            acc = self.getAccuracy(self.getPredictions(self.finalActivations[-1]), y)
            
            print(f"Epoch {epoch + 1}/{epochs} | Accuracy: {acc:.4f}")

    def predict(self, index, yData, yLabels):
        sampleImg = yData[:, index:index+1]
        sampleLabel = yLabels[index]

        self.forwardProp(sampleImg)

        print("Prediction:", chr(self.getPredictions(self.finalActivations[-1])[0] + 65))
        print("Actual Character:", chr(sampleLabel + 65))

        sampleImg = sampleImg.reshape(28, 28)

        plt.imshow(sampleImg, cmap='gray')
        plt.title(f"Label: {sampleLabel}")
        plt.axis('off')
        plt.show()