import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

def sigmoid(z):
    return 1/(1+np.exp(-z))
def initialize_with_zeros(dim):
    #return np.zeros(dim).reshape(dim,1),b
    return np.random.randn(dim,1),b
def propagate(w,b,X,Y):
    #include forward propagate and back propagate
    #get dw and db
    #it can be used to grandient descent
    m=X.shape[1]
    A=sigmoid(np.dot(w.T,X)+b)
    cost = (-1/m) * (np.dot(Y,np.log(A).T)+np.dot((1-Y),np.log(1-A).T))
    dw=1/m*(np.dot(X,(A-Y).T))
    db=1/m*np.sum((A-Y))
    cost=np.squeeze(cost)
    assert(cost.shape==())
    grads={"dw":dw,"db":db}
    return grads,cost
def optimize(w,b,X,Y,num_iterations,learning_rate,print_cost=False):
    costs=[]
    for i in range(num_iterations):
        grads,cost=propagate(w,b,X,Y)
        dw=grads["dw"]
        db=grads["db"]
        w=w-learning_rate*dw
        b=b-learning_rate*db
        if i%100==0:
            costs.append(cost)
        if print_cost and i%100==0:
            print("cost after iteration: %i:%f" %(i,cost))
    params={"w":w,"b":b}
    grads={"dw":dw,"db":db}
    return params,grads,costs
def predict(w,b,X):
    m=X.shape[1]
    Y_prediction=np.zeros((1,m))
    w=w.reshape(X.shape[0],1)
    A=sigmoid(np.dot(w.T,X)+b)
    for i in range(A.shape[1]):
        if A[0,i]>0.5:
            Y_prediction[0,i]=1
        else:
            Y_prediction[0,i]=0
    return Y_prediction
def model(X_train,Y_train,X_test,Y_test,num_iterations=2000,learning_rate=0.5,print_cost=False):
    w,b=initialize_with_zeros(X_train.shape[0])
    parameters,grads,costs=optimize(w,b,X_train,Y_train,num_iterations,learning_rate,print_cost=True)
    w=parameters["w"]
    b=parameters["b"]
    Y_prediction_test=predict(w,b,X_test)
    Y_prediction_train=(predict(w,b,X_train))
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d


    

train_set_x_orig,train_set_y,test_set_x_orig,test_set_y,classes=load_dataset()
#train_set_x_orig.shape=(209,64,64,3)
#train_set_y.shape=(1,209)
#test_set_x_orig.shape=(50,64,64,3)
#test_set_y.shape=(1,50)
#classes.shape=(2,)

index=24
#plt.imshow(train_set_x_orig[index])
#plt.show()
print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")
m_train=train_set_x_orig.shape[0]
m_test=test_set_x_orig.shape[0]
num_px=train_set_x_orig.shape[1]

train_set_x_flatten=train_set_x_orig.reshape(m_train,num_px*num_px*3).T
#train_set_x_flatten=train_set_x_orig.reshape((num_px**2)*3,m_train)
test_set_x_flatten=test_set_x_orig.reshape(m_test,num_px*num_px*3).T
print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))#(12288,209)
print ("train_set_y shape: " + str(train_set_y.shape))#(1,209)
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print ("test_set_y shape: " + str(test_set_y.shape))
print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0]))
train_set_x=train_set_x_flatten/255#normalizetion
test_set_x=test_set_x_flatten/255

#testing function propagate() and optimize()
w, b, X, Y = np.array([[1],[2]]), 2, np.array([[1,2],[3,4]]), np.array([[1,0]])
grads, cost = propagate(w, b, X, Y)
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))
print ("cost = " + str(cost))
params, grads, costs = optimize(w, b, X, Y, num_iterations= 100, learning_rate = 0.009, print_cost = False)
print ("w = " + str(params["w"]))
print ("b = " + str(params["b"]))
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))


print("prdictions = "+str(predict(w,b,X)))

d=model(train_set_x,train_set_y,test_set_x,test_set_y,num_iterations=2000,learning_rate=0.5)
#different learning_rate makes different result
