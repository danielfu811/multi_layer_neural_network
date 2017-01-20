'''
    implementation of artificial neural network.(deep learning)
    output layer is softmax.
    reference: http://ufldl.stanford.edu/tutorial/supervised/MultiLayerNeuralNetworks/
'''

import numpy as np
import util 
import random 
from sklearn.metrics import classification_report 

init_dev = 0.01
grad_dev = 0.0001

class Layer:
    def __init__(self, layNum, nodeC, nextL, wd=0.1, activateF='rectified', mode='common'):
        '''
            layNum: current layer number 
            nodeC : the node count of the current layer
            NextL : the node count of the next layer
            activateF: activate function, it should be 'rectified', 'sigmoid' or 'tanh'
            wd: weight decay parameter, 0<= wd <= 1, for regulation
            mode: the hidden layer operate mode, it should be 'common' or 'bypass'.
                  'bypass' mode is used next to the output layer.
        '''
        self.layNum = layNum
        self.nodeC = nodeC
        self.nextL = nextL
        self.activateF = activateF 
        self.wd = wd
        self.mode = mode

        self.w = init_dev * np.mat(np.random.randn(nextL, nodeC))  #weight matrix
        self.b = init_dev * np.mat(np.random.randn(nextL, 1))      #interception 
        self.a = None                                              #z(w*x + b) after activate: size: nodeC * 1
        self.dw = None                                             #the partial derivative of self.w 
        self.db = None                                             #the partial derivative of self.b
        self.accDw = np.mat(np.zeros((nextL, nodeC)))              #accumulated dw
        self.accDb = np.mat(np.zeros((nextL, 1)))                  #accumulated db
        self.accC = 0                                              #accumulated times
        self.delta = None                                          #the error term of current layer 

    def printLayer(self):
        print("###   layer number: ", self.layNum)
        print("###     node count: ", self.nodeC)
        print("###next node count: ", self.nextL)
        print("###   weight decay: ", self.wd)
        print("###       activate: ", self.activateF)
        print("###           mode: ", self.mode)
        print("##################")

    
    def printInternal(self):
        print("###lay number: ", self.layNum)
        print("a : ", self.a)
        print("w : ", self.w)
        print("b : ", self.b)
        print("dw: ", self.dw)
        print("db: ", self.db)
        print("delta: ", self.delta)

    def activateImp(self, zz):
        if self.activateF == 'rectified':
            return util.rectifiedLinear(zz)
        if self.activateF == 'sigmoid':
            return util.sigmoid(zz)
        if self.activateF == 'tanh':
            return util.tanh(zz)
        return None 

    def actDerivativeImp(self):
        '''
            use self.a to calculate derivative for backpropagation. 
            self.a = activateF(z)
        '''
        ad = None
        if self.activateF == 'rectified':
            ad = self.a.copy()
            ad[ad > 0] = float(1)
            ad[ad <= 0] = float(0)

        if self.activateF == 'sigmoid':
            ad = np.multiply(self.a, (1 - self.a))

        if self.activateF == 'tanh':
            ad = 1 - np.power(self.a, 2)

        return ad 

    def getL2(self): return np.sum(np.multiply(self.w, self.w))

    def gradCheckInit(self):
        '''
            init for gradient check.
            random check one gradient in a hidden unit.
            return a list contain candidated gradient checking, its length should be self.nextL.
            the element of the list: (coor-r, coor-c, a-pos, square-wpos, a-neg, square-wneg)
        '''
        gradChList = []
        for i in range(self.nextL):
            ##random select column index
            coorC = random.randint(0, self.nodeC-1)
            cw0 = np.copy(self.w)
            cw1 = np.copy(self.w)
            cw0[i,coorC] += grad_dev 
            cw1[i,coorC] -= grad_dev
            sPos = np.sum(np.multiply(cw0, cw0))
            sNeg = np.sum(np.multiply(cw1, cw1))
            nz = cw0 * self.a + self.b
            if self.mode == 'bypass': aPos = nz 
            else: aPos = self.activateImp(nz)
            nz = cw1 * self.a + self.b
            if self.mode == 'bypass': aNeg = nz 
            else: aNeg = self.activateImp(nz)
            gradChList.append((i, coorC, aPos, sPos, aNeg, sNeg))
        return gradChList

    def gradCheck(self, checkList):
        '''
            check one gradient in a hidden unit.
            the check list length should be self.nextL.
            the element of the list: (coor-r, coor-c, costJ-pos, costJ-neg)
        '''
        status = True 
        for (i, j, costJPos, costJNeg) in checkList:
            estiG = (costJPos - costJNeg) / (2 * grad_dev)
            ddw = self.dw[i,j] + self.wd*self.w[i,j]
            if np.abs(ddw - estiG) > 1e-6:
                status = False
                print("[ERROR]layer %d, gradient check[%d,%d], expected %f, but %f" % \
                      (self.layNum, i, j, estiG, ddw))
            else:
                print("[PASS]layer %d, gradient check[%d,%d], numberic %f, bp %f" % \
                      (self.layNum, i, j, estiG, ddw))

        return status

    def forwardProgation(self, lastA):
        '''
            lastA: the a value from last layer(self.layNum-1), size: self.nodeC*1.
            return new z and a to next layer(self.layNum+1), size: self.nextL*1.
        '''
        self.a = lastA
        nz = self.w * self.a + self.b
        if self.mode == 'bypass': return nz
        na = self.activateImp(nz)
        return na

    def backProgation(self, deltaN):
        '''
            deltaN: the error term from next layer(self.layNum+1), size: self.nextL*1.
            return new delta to last layer(self.layNum-1), size; self.nodeC*1.
        '''
        self.dw = deltaN * self.a.T
        self.db = deltaN

        ##accumulate dw and db
        self.accDw += self.dw
        self.accDb += self.db
        self.accC += 1
        
        ##the first layer(input layer) doesn't neet to calculate error term. 
        if self.layNum == 1: self.delta = None
        else: self.delta = np.multiply(self.w.T * deltaN, self.actDerivativeImp())

        return self.delta  

    def updateParameters(self, alpha=0.1):
        '''
            alpha: learning rate.
        '''
        ##update w and b
        self.w = self.w - alpha * (self.accDw/self.accC + self.wd * self.w)
        self.b = self.b - alpha * (self.accDb/self.accC)
        ##clear to zero
        self.accDw[:,:] = 0  
        self.accDb[:,:] = 0 
        self.accC = 0                                              

class OutputLayer:
    '''
        softmax layer.
    '''
    def __init__(self, layNum, K):
        '''
            layNum: current layer number 
            K     : the class count.
        '''
        self.layNum = layNum
        self.K = K
        self.z = None          #theta*x + b, size: (K)*1
        self.delta = None      #current error term, size: (K)*1

    def printLayer(self):
        print("###layer number: %d(output, softmax)" % (self.layNum))
        print("###  node count: ", self.K)
        print("### label count: ", self.K)
        print("##################")

    def printInternal(self):
        print("###output layer delta: ", self.delta)

    def costJ(self, z, y):
        '''
            z: output from last hidden layer, size: (K)*1.
            y: current sample label.
        '''
        return -np.log(util.softmax1(z, y, self.K))

    def predict(self, z):
        '''
            z: output from last hidden layer, size: (K)*1.
        '''
        judge = z 
        maxI = judge.argmax(0)
        curC = maxI[0,0]
        return curC

    def forwardProgation(self, z, y):
        '''
            z: output from last hidden layer, size: (K)*1.
            y: current sample label.
        '''
        self.z = z
        self.delta = np.mat(np.zeros((self.K, 1))) 
        for i in range(self.K):
            self.delta[i][0] = util.softmax1(z, i, self.K)
        self.delta[y][0] -= 1

    def backProgation(self):
        return self.delta


class AnnDL:
    '''
        the top level of artificial neural network.
    '''
    def __init__(self, layerStack, K, wd= 0.1, activateF='rectified'):
        '''
        initial the whole neural network.
            layerStack: a list contains the count of neural unit for every layer. 
                        the total layers should be the length of layerStack.
                        the first element is the input layer, while the last one is output layer.
                        the unit count of the output unit should be (K)
            K: the total class for the output layer.
            wd: weight decay.
            activateF: the activate function.
        ''' 
        self.layerStack = [] 
        self.totalLayer = len(layerStack)
        self.wd = wd
        for i in range(self.totalLayer):
            nodeC = layerStack[i]
            if i == self.totalLayer-2: mode = 'bypass'
            else: mode = 'common'
            if i == self.totalLayer-1:
                ##output layer
                layer = OutputLayer(i+1, K)
            else:
                layer = Layer(i+1, nodeC, layerStack[i+1], wd, activateF, mode) 
            self.layerStack.append(layer)
        for layer in self.layerStack:
            layer.printLayer()

    def getCostJ(self, a, k, y):
        while k < self.totalLayer-1:
            a = self.layerStack[k].forwardProgation(a)
            k += 1
        return self.layerStack[-1].costJ(a, y)

    def getL2(self, k, sk):
        l2 = sk 
        i = 0
        while i < self.totalLayer-1:
            if i != k: l2 += self.layerStack[i].getL2()
            i += 1
        return l2*self.wd*0.5

    def gradCheck(self, y):
        tmpList = []
        status = True
        for k in range(self.totalLayer-2, -1, -1):
            gradChList = self.layerStack[k].gradCheckInit()
            for (i, j, aPos, sPos, aNeg, sNeg) in gradChList:
                costPos = self.getCostJ(aPos, k+1, y) + self.getL2(k, sPos)
                costNeg = self.getCostJ(aNeg, k+1, y) + self.getL2(k, sNeg)
                tmpList.append((i,j, costPos, costNeg))
            status = self.layerStack[k].gradCheck(tmpList)
            if status == False: break
            del tmpList[:]
        return status 

    def train(self, xTrain, yTrain, gcEnable=False, numIter=30, stopT=1e-3, debug=False):
        m,n = xTrain.shape
        print("training begin...")
        for j in range(numIter):
            print("batchTrain: %d round..." % (j))
            for i in range(m):
                ##forward Progation
                a = np.mat(xTrain[i])
                a = np.reshape(a, (n,1))
                for k in range(self.totalLayer-1):
                    a = self.layerStack[k].forwardProgation(a)
                self.layerStack[-1].forwardProgation(a, yTrain[i])
                ##back Progation
                delta = self.layerStack[-1].backProgation()
                if debug: self.layerStack[-1].printInternal()
                for k in range(self.totalLayer-2, -1, -1):
                    delta = self.layerStack[k].backProgation(delta)
                    if debug: self.layerStack[k].printInternal()
                ##gradient check
                if gcEnable and self.gradCheck(yTrain[i]) == False: 
                    print("[ERROR] gradient check fail!!")
                    return 

            #parameters update 
            alpha = 1.0/(1+j) + 0.0001
            for k in range(self.totalLayer-1):
                self.layerStack[k].updateParameters(alpha)
        ###
        print("calculate training error...")
        estY = self.predictX(xTrain)
        error = 0
        for i in range(m):
            if estY[i] != yTrain[i]: error += 1
        print("train error: %f(%d/%d)" % (error/(m-0.0), error, m))


    def predictOne(self, x):
        a = x
        for k in range(self.totalLayer-1):
            a = self.layerStack[k].forwardProgation(a)
        return self.layerStack[-1].predict(a)

    def predictX(self, X):
        estY = []
        m,n = X.shape
        for k in range(m):
            a = np.mat(X[k])
            a = np.reshape(a, (n,1))
            estY.append(self.predictOne(a))
        return estY

#############################################################################

def outputLayerTest():
    print("loading train data...")
    trainX, trainY = util.getDataSet('trainingDigits')
    m, n = trainX.shape
    np.random.seed(0)
    random.seed(0)
    print("%d samples, with %d dimensions" % (m, n))

    print("training ...")
    ann = AnnDL([1024, 10], 10)
    ann.train(trainX, trainY, numIter=30)
    pY = ann.predictX(trainX)
    target_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    print(classification_report(trainY, pY, target_names=target_names))

    print("loading test data ...")
    testX, testY = util.getDataSet('testDigits')
    m, n = testX.shape
    print("%d samples, with %d dimensions" % (m, n))
    pY = ann.predictX(testX)
    error = 0 
    for i in range(m):
        if pY[i] != testY[i]: error += 1
    print("test error: %f(%d/%d)" % (error/(m-0.0), error, m))
    print(classification_report(testY, pY, target_names=target_names))

def gradientTest():
    trainX = np.mat([[3, 1, 2], [1, 1, 1], [4, 3, 2]], dtype=np.float64)
    trainY = [0, 1, 2] 
    np.random.seed(0)
    random.seed(0)

    print("training ...")
    ann = AnnDL([3, 6, 3, 3], K=3, activateF='tanh')
    ann.train(trainX, trainY, numIter=1, gcEnable=True, debug=True)
    print("finish!!")

def normalTest():
    print("loading train data...")
    trainX, trainY = util.getDataSet('trainingDigits')
    m, n = trainX.shape
    np.random.seed(0)
    random.seed(0)
    print("%d samples, with %d dimensions" % (m, n))

    print("training ...")
    ann = AnnDL([1024, 512, 10], K=10, wd=0.001, activateF='rectified')
    ann.train(trainX, trainY, numIter=30, gcEnable=False)
    pY = ann.predictX(trainX)
    target_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    print(classification_report(trainY, pY, target_names=target_names))

    print("loading test data ...")
    testX, testY = util.getDataSet('testDigits')
    m, n = testX.shape
    print("%d samples, with %d dimensions" % (m, n))
    pY = ann.predictX(testX)
    error = 0 
    for i in range(m):
        if pY[i] != testY[i]: error += 1
    print("test error: %f(%d/%d)" % (error/(m-0.0), error, m))
    print(classification_report(testY, pY, target_names=target_names))

##unit test
if  __name__ == '__main__':
    #outputLayerTest()
    #gradientTest()
    normalTest()

   

