'''
    tool box function for deep leaning.
'''
import numpy as np
from os import listdir

def softmax1(x, k, K):
    '''
        return the value of softmax function. 
        softmax(x, k, K) = exp(xi) / (sum(exp(xi), i from 0 ~K-1)) 
        x    : (K)*1, one sample.
        k    : the k value, between 0 ~ K-1
    '''
    expV = x 
    #####exp overflow procedure 
    if x.dtype == np.float32: tinf = 80
    else: tinf = 700
    maxI = expV.argmax(0)[0,0]
    if expV[maxI, 0] > tinf:
        ###overfolw
        if maxI == k: return float(1.0)
        return float(0.0)

    expV = np.exp(expV)
    #print(expV)
    normalize = np.sum(expV)
    #print(normalize) 
    return expV[k,0]/normalize

def softmax(x, k, K):
    '''
        return the value of softmax function. 
        softmax(x, k, K) = exp(xi) / (1 + sum(exp(xi), i from 0 ~K-2)) 
        x    : (K-1)*1, one sample.
        k    : the k value, when k == K-1, it return 1/(1 + sum(exp(xi), i from 0 ~K-2))
    '''
    expV = x 
    #####exp overflow procedure 
    if x.dtype == np.float32: tinf = 80
    else: tinf = 700
    maxI = expV.argmax(0)[0,0]
    if expV[maxI, 0] > tinf:
        ###overfolw
        if maxI == k: return float(1.0)
        return float(0.0)

    expV = np.exp(expV)
    #print(expV)
    normalize = 1 + np.sum(expV)
    #print(normalize) 
    if k == (K-1): return 1/normalize
    return expV[k,0]/normalize

def sigmoid(z):
    '''
        sigmoid function: f(z) = 1 / (1 + exp(-z)).
        z should be np.mat[n,1].
    '''
    cz = z.copy()
    ##overflow detection
    if z.dtype == np.float32: tinf = 80
    else: tinf = 700
    mz = cz < -tinf
    cz[mz] = float(0)
    result = 1 + np.exp(-cz)
    result = 1 / result
    result[mz] = float(0)
    return result

def tanh(z):
    '''
        tanh function: f(z) = (exp(z) - exp(-z)) / (exp(z) + exp(-z)).
        z should be np.mat[n,1].
    '''
    cz = z.copy()
    ##overflow detection
    if z.dtype == np.float32: tinf = 80
    else: tinf = 700
    mzNeg = cz < -tinf
    mzPos = cz > tinf 
    cz[mzNeg] = float(0)
    cz[mzPos] = float(0)
    expzP = np.exp(cz)
    expzN = np.exp(-cz)
    result = (expzP - expzN) / (expzP + expzN)
    result[mzNeg] = float(-1)
    result[mzPos] = float(1)
    return result 

def rectifiedLinear(z):
    '''
        rectified linear function: f(z) = max(0, z).
        z should be np.mat[n,1].
    '''
    cz = z.copy()
    cz[cz<0] = float(0)
    return cz


def img2vector(filename):
    returnVect = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def getDataSet(dirName):
    hwLabels = []
    fileList = listdir(dirName)
    m = len(fileList)
    dataMat = np.zeros((m,1024))
    for i in range(m):
        fileNameStr = fileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        dataMat[i,:] = img2vector('%s/%s' % (dirName, fileNameStr))

    return dataMat, hwLabels 

if __name__ == "__main__":
    trainX, trainY = getDataSet('trainingDigits')
    print(trainX.shape)
    print(len(trainY))
    testX, testY = getDataSet('testDigits')
    print(testX.shape)
    print(len(testY))

    #tz = np.mat([1, -200, 800], dtype=np.float32)
    #tz = tz.T
    #print(tz.shape)
    #print(tz)
    #print(sigmoid(tz))

