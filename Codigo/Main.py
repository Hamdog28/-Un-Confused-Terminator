from __future__ import division
import cPickle as pickle
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

from PIL import Image
import PIL.ImageOps 
import scipy.signal

import random
from sklearn.metrics import mean_squared_error
from scipy.ndimage.filters import gaussian_filter

#Librerias de la escritura en excel
from openpyxl import Workbook
wb = Workbook()



#librerias del set de datos
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split

# XOR
X = np.array(([0, 0], [0, 1], [1, 0], [1,1]), dtype=float)
y = np.array(([0], [1], [1],[0]), dtype=float)



def save_img(img,name):
    img = np.ndarray.transpose(img)
    for i in xrange(len(img)):
        rescaled = (255.0 / img.max() * (img - img.min())).astype(np.uint8)
        im = Image.fromarray(np.reshape(rescaled[i],(28,28)))

        #im = Image.fromarray(rescaled)
        im.save("imagenes/" + name + str(i) + ".jpeg")
    return 0

class Neural_Network(object):
  def __init__(self):
    #parameters
    self.inputSize = 784
    self.outputSize = 10
    self.hiddenSize = 128

    #weights
    self.W1 = np.random.randn(self.inputSize, self.hiddenSize)/np.sqrt(self.inputSize) # (128x784) weight matrix from input to hidden layer
    self.W2 = np.random.randn(self.hiddenSize, self.hiddenSize )/np.sqrt(self.hiddenSize) # (128x128) weight matrix from hidden to second hidden layer
    self.W3 = np.random.randn(self.hiddenSize, self.outputSize)/np.sqrt(self.hiddenSize) # (128x10) weight matrix from second hidden to output layer
        
  def forward(self, X,drop):
    
    #forward propagation through our network
    self.z = np.dot(X, self.W1) # dot product of X (input) and first set of 3x2 weights
    if (drop):
        self.indx = random.sample(range(self.z.shape[0]), self.z.shape[0]//2)
        self.z[self.indx]=0
    self.z2 = self.ReLU(self.z) # activation 
    
    self.z3 = np.dot(self.z2, self.W2) # dot product of hidden layer (z2) and second set of 3x1 weights
    self.z4 = self.ReLU(self.z3) # activation function

    self.z5 = np.dot(self.z4, self.W3) # dot product of hidden layer (z2) and second set of 3x1 weights
    #o = self.sigmoid(self.z5) # final activation function\
    o = self.softmax(self.z5)
    return o 

  def forward1(self, X,drop):
        
    #forward propagation through our network
    self.z = np.dot(X, self.W1) # dot product of X (input) and first set of 3x2 weights
    if (drop):
        self.indx = random.sample(range(self.z.shape[0]), self.z.shape[0]//2)
        self.z[self.indx]=0
    self.z2 = self.ReLU(self.z) # activation 
    
    self.z3 = np.dot(self.z2, self.W3) 
    o = self.softmax(self.z3)
    return o 

  def ReLU(self,x):
        #return x * (x > 0)
        x[x < 0] = 0
        return x

  def dReLU(self,x):
        return 1. * (x > 0)

  def sigmoid(self, s):
    return 1/(1+np.exp(-s))

  def sigmoidPrime(self, s):
    return s * (1 - s)

  def delta_cross_entropy(self,y,grad):
        """
        grad -> resultado de softmax
        y one hot vector
        """
        return y-grad
    
  def cross_entropy(self,y,soft):
        """
        soft -> resultado de softmax
        y -> one hot vector
        """
        
        soft[soft == 0] = np.finfo(float).eps
        soft[soft == 1] = 0.9
        loss = np.mean(np.sum(np.nan_to_num(-y*np.log(soft)-(1-y)*np.log(1-soft)),axis = 1))
        return loss
    
  def softmax(self,X):
        e = np.exp(X-np.max(X,axis=-1, keepdims=True))
        return e/np.sum(e, axis=-1, keepdims=True) 
            

  def backward(self, X, y, o,lr=0.0085):
    #One hot vector
    b = np.zeros((y.shape[0], 10))
    b[np.arange(y.shape[0]), y] = 1

    # backward propgate through the network
    self.o_error = self.cross_entropy(b,o)#b - o # error in output
    self.o_delta = self.o_error*self.delta_cross_entropy(b,o)#self.o_error*self.sigmoidPrime(o) # applying derivative of sigmoid to error
   
    self.z4_error = self.o_delta.dot(self.W3.T)
    self.z4_delta = self.z4_error * self.dReLU(self.z4)

    self.z2_error = self.z4_delta.dot(self.W2.T) # z2 error: how much our hidden layer weights contributed to output error
    self.z2[self.indx]=1
    self.z2_delta = self.z2_error * self.dReLU(self.z2) # applying derivative of sigmoid to z2 error
    self.z2_delta[self.indx]=0
    self.W1 += X.T.dot(lr * self.z2_delta) # adjusting first set (input --> hidden) weights
    self.W2 += self.z2.T.dot(lr * self.z4_delta) 
    self.W3 += self.z4.T.dot(lr * self.o_delta) # adjusting second set (hidden --> output) weights

  def backward1(self, X, y, o,lr=0.0085):
    #One hot vector
    b = np.zeros((y.shape[0], 10))
    b[np.arange(y.shape[0]), y] = 1

    # backward propgate through the network
    self.o_error = self.cross_entropy(b,o)# error in output
    self.o_delta = self.o_error*self.delta_cross_entropy(b,o)# applying derivative of cross entropy to error


    self.z2_error = self.o_delta.dot(self.W3.T) # z2 error: how much our hidden layer weights contributed to output error
    self.z2[self.indx]=1
    self.z2_delta = self.z2_error * self.dReLU(self.z2) # applying derivative of sigmoid to z2 error
    self.z2_delta[self.indx]=0
    self.W1 += X.T.dot(lr * self.z2_delta) # adjusting first set (input --> hidden) weights
    self.W3 += self.z2.T.dot(lr * self.o_delta) # adjusting second set (hidden --> output) weights

    
    #print dic_pesos

  def XOR(self, X, y): 
        x_test = X
        y_test = y
        for j in range(100):
            for i in xrange(2):
            
                indx = random.sample(range(X.shape[0]), 2)
                x_aux = X[indx, :]
                y_aux = y[indx]
                X = np.delete(X,indx,0)
                y = np.delete(y,indx,0)
                o = self.forward(x_aux)
                print "Loss: \n" + str(np.mean(np.square(y - o))) # mean sum squared loss
                self.backward(x_aux, y_aux, o)
            X = x_test
            y = y_test
        #TEST
        o = self.forward(x_test)
        print y_test.shape[0]
        print o
        print  ((np.count_nonzero(np.subtract(o.argmax(axis=1),y_test) == 0))*100)/y_test.shape[0] 

  def train (self, X, y,verify_x,verify_y):
    ws = wb.active
    ws1 = wb.create_sheet("128")
    loss_ = 20
    for j in xrange (4):
            print j
            for i in xrange(500):
                indx = random.sample(range(X.shape[0]), 25)
                x_aux = X[indx, :]
                y_aux = y[indx]
                    
                X = np.delete(X,indx,0)
                y = np.delete(y,indx,0)
                o = self.forward(x_aux,True)

                b = np.zeros((y_aux.shape[0], 10))
                b[np.arange(y_aux.shape[0]), y_aux] = 1
                #print "Loss: \n" + str(np.mean(np.square(b - o)))
                self.backward(x_aux, y_aux, o)
                if((i+1) % 50 == 0):
                    #one hot vector
                    b = np.zeros((verify_y.shape[0], 10))
                    b[np.arange(verify_y.shape[0]), verify_y] = 1
                    o = self.forward(verify_x,False)

                    loss=self.cross_entropy(b,o)
                    print "Loss: \n" + str(loss)
                    ws1.append([((i//50)+1)+(j*10),loss,(((np.count_nonzero(np.subtract(o.argmax(axis=1),verify_y) == 0))*100)/verify_y.shape[0])])
                    
                    #se actualizan los pesos en el archivo
                    dic_pesos = { "W1": self.W1, "W2": self.W2, "W3": self.W3 }
                    pickle.dump( dic_pesos, open( "pesos.p", "wb" ) )

                
            save_img(self.W1,str(j+1)+"/")

    wb.save("resultado.xlsx")
    
  
  def test(self, X,y,archivo,Invert=False):
        if archivo is None:
          archivo="pesos_128d.p"
        else:
          X = Image.open(X)   
          X = X.resize((28,28),Image.ANTIALIAS) 
          X = X.convert('L')
          if Invert:
            X = PIL.ImageOps.invert(X)
            X.save("invertido.jpeg")
          X = np.array(X)
          X = X.ravel()
        
        
        dic_pesos = pickle.load( open( archivo, "rb" ) )
        o = self.forward(X,False)
        self.W1 = dic_pesos.values()[2] 
        self.W2 = dic_pesos.values()[1]
        self.W3 = dic_pesos.values()[0]
        o = self.forward(X,False)
        if y is not None:
          print  ((np.count_nonzero(np.subtract(o.argmax(axis=1),y) == 0))*100)/y.shape[0]
          return o.argmax(axis = 1)
        else:
          return o.argmax(axis = 0)
def main():
    print "Obteniendo datos"
    mnist = fetch_mldata('MNIST original')

    train_img, test_img, train_lbl, test_lbl = train_test_split(
    mnist.data, mnist.target, test_size=1/7.0, random_state=0)
    NN = Neural_Network()
    NN.train(train_img[:50000]/255,train_lbl.astype(int),train_img[-10000:]/255,train_lbl[-10000:].astype(int))
    NN.test(test_img,test_lbl,None)
    return 0

#para correr la interfaz se debe comentar la siguiente linea
main()

#para aplicarle filtro a las imagenes
#gaussian_filter(train_img[:50000], sigma = 2.5)/255






