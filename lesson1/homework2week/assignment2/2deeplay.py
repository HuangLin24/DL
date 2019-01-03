import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

train_set_x_orig,train_set_y,test_set_x_orig,test_set_y,classes=load_dataset()
m_train=train_set_x_orig.shape[0]
m_test=test_set_x_orig.shape[0]
num_px=train_set_x_orig.shape[1]
train_set_x=(train_set_x_orig.reshape(m_train,num_px*num_px*3).T)/255
test_set_x=(test_set_x_orig.reshape(m_test,num_px*num_px*3).T)/255

print(test_set_x.shape[0])
def sigmoid(z):
    return 1/(1+np.exp(-z))

def layer_sizes(X,numhidden,Y):
    return X.shape[0],numhidden,Y.shape[0]
def initialize_parameters(Xsize,numhidden,Ysize):
    W1=np.random.randn(numhidden,Xsize)
    b1=np.random.randn(numhidden,1)
    W2=np.random.randn(Ysize,numhidden)
    b2=np.random.randn(Ysize,1)
    return {"W1":W1,"W2":W2,"b1":b1,"b2":b2}
def forward_propagation(X,parameters):
    W1=parameters["W1"]
    W2=parameters["W2"]
    b1=parameters["b1"]
    b2=parameters["b2"]
    Z1=np.dot(W1,X)+b1
    A1=np.tanh(Z1)
    Z2=np.dot(W2,A1)+b2
    A2=sigmoid(Z2)
    return {"Z1":Z1,"A1":A1,"Z2":Z2,"A2":A2}
def backward_propagation(parameters,cache,X,Y):
    m=X.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]
    dZ2 = A2-Y
    dW2 = 1/m*np.dot(dZ2,A1.T)
    db2 = 1/m * np.sum(dZ2,axis=1,keepdims=True)
    dZ1 = np.dot(W2.T,dZ2)*(1-np.power(A1,2))
    dW1 = 1/m * np.dot(dZ1,X.T)
    db1 = 1/m * np.sum(dZ1,axis=1,keepdims=True)
    return {"dW1":dW1,"dW2":dW2,"db1":db1,"db2":db2}
def update_parameters(parameters,grads,learning_rate=0.9):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    return parameters
def compute_cost(A2, Y, parameters):
    m = Y.shape[1] # number of example
    logprobs =  np.multiply(np.log(A2),Y) + np.multiply(np.log(1 - A2),1 - Y) 
    cost = -np.sum(logprobs)/m
    cost = np.squeeze(cost)
    assert(isinstance(cost, float))
    return cost
def nn_model(X, Y, numhidden, num_iterations = 10000, learning_rate=1.2,print_cost=False):
    np.random.seed(3)
    n_x,n_h,n_y = layer_sizes(X,numhidden,Y)
    parameters=initialize_parameters(n_x,n_h,n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    for i in range(num_iterations):
        cache = forward_propagation(X, parameters)
        cost = compute_cost(cache["A2"], Y, parameters)
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads, learning_rate)
        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f learning_rate:%f" %(i, cost,learning_rate))
            #learning_rate=learning_rate-0.2
    return parameters



def predict(parameters, X):
    cache = forward_propagation(X, parameters)
    predictions = np.around(cache["A2"])
    return predictions


parameters=nn_model(train_set_x,train_set_y,4,5000,print_cost=True)
predictions=predict(parameters,train_set_x)
print ('Accuracy: %d' % float((np.dot(train_set_y,predictions.T) + np.dot(1-train_set_y,1-predictions.T))/float(train_set_y.size)*100) + '%')


predictions=predict(parameters,test_set_x)
print ('Accuracy: %d' % float((np.dot(test_set_y,predictions.T) + np.dot(1-test_set_y,1-predictions.T))/float(test_set_y.size)*100) + '%')


