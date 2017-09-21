import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
import pickle

def preprocess():
    """ 
     Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set
    """

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    n_feature = mat.get("train1").shape[1]
    n_sample = 0
    for i in range(10):
        n_sample = n_sample + mat.get("train" + str(i)).shape[0]
    n_validation = 1000
    n_train = n_sample - 10 * n_validation

    # Construct validation data
    validation_data = np.zeros((10 * n_validation, n_feature))
    for i in range(10):
        validation_data[i * n_validation:(i + 1) * n_validation, :] = mat.get("train" + str(i))[0:n_validation, :]

    # Construct validation label
    validation_label = np.ones((10 * n_validation, 1))
    for i in range(10):
        validation_label[i * n_validation:(i + 1) * n_validation, :] = i * np.ones((n_validation, 1))

    # Construct training data and label
    train_data = np.zeros((n_train, n_feature))
    train_label = np.zeros((n_train, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("train" + str(i)).shape[0]
        train_data[temp:temp + size_i - n_validation, :] = mat.get("train" + str(i))[n_validation:size_i, :]
        train_label[temp:temp + size_i - n_validation, :] = i * np.ones((size_i - n_validation, 1))
        temp = temp + size_i - n_validation

    # Construct test data and label
    n_test = 0
    for i in range(10):
        n_test = n_test + mat.get("test" + str(i)).shape[0]
    test_data = np.zeros((n_test, n_feature))
    test_label = np.zeros((n_test, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("test" + str(i)).shape[0]
        test_data[temp:temp + size_i, :] = mat.get("test" + str(i))
        test_label[temp:temp + size_i, :] = i * np.ones((size_i, 1))
        temp = temp + size_i

    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis=0)
    index = np.array([])
    for i in range(n_feature):
        if (sigma[i] > 0.001):
            index = np.append(index, [i])
    train_data = train_data[:, index.astype(int)]
    validation_data = validation_data[:, index.astype(int)]
    test_data = test_data[:, index.astype(int)]

    # Scale data to 0 and 1
    train_data /= 255.0
    validation_data /= 255.0
    test_data /= 255.0
    """
    train_data = train_data[::50]
    test_data = test_data[::50]
    validation_data = validation_data[::50]
    train_label = train_label[::50]
    test_label = test_label[::50]
    validation_label = validation_label[::50]
    """

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def blrObjFunction(initialWeights, *args):
    """
    blrObjFunction computes 2-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector (w_k) of size (D + 1) x 1 
        train_data: the data matrix of size N x D
        labeli: the label vector (y_k) of size N x 1 where each entry can be either 0 or 1 representing the label of corresponding feature vector

    Output: 
        error: the scalar value of error function of 2-class logistic regression
        error_grad: the vector of size (D+1) x 1 representing the gradient of
                    error function
    """
    train_data, labeli = args
    n_data = train_data.shape[0]
    
    #n_features = train_data.shape[1]
    #error = 0
    #error_grad = np.zeros((n_features + 1, 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    # Adding bias to train_data
    train_data_bias = np.concatenate((np.ones([int(train_data.shape[0]),1]),train_data), axis = 1)

    # theta_n = sigmoid(w.T * X)
    theta_n = sigmoid((np.dot(initialWeights, train_data_bias.T)).T)
    error1 = ((np.dot(labeli.T, (np.log(theta_n).reshape([int(theta_n.shape[0]), 1])))) + (np.dot((1-labeli).T, (np.log(1-theta_n).reshape([int(theta_n.shape[0]), 1])))))/float(n_data)

    error = (-1) * (error1[0])
    error_grad = (np.dot(((theta_n.reshape(labeli.shape) - labeli).T),train_data_bias)/float(n_data)).flatten()

    return error, error_grad


def blrPredict(W, data):
    """
     blrObjFunction predicts the label of data given the data and parameter W 
     of Logistic Regression
     
     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight 
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D
         
     Output: 
         label: vector of size N x 1 representing the predicted label of 
         corresponding feature vector given in data matrix

    """

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    label = np.zeros((data.shape[0], 1))
    r = sigmoid(np.dot((np.concatenate((np.ones([int(data.shape[0]),1]),data), axis = 1)), W))

    for i in range(label.shape[0]):
        label[i] = np.argmax(r[i])

    return label


def mlrObjFunction(params, *args):
    """
    mlrObjFunction computes multi-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector of size (D + 1) x 1
        train_data: the data matrix of size N x D
        labeli: the label vector of size N x 1 where each entry can be either 0 or 1
                representing the label of corresponding feature vector

    Output:
        error: the scalar value of error function of multi-class logistic regression
        error_grad: the vector of size (D+1) x 10 representing the gradient of
                    error function
    """
    #n_feature = train_data.shape[1]
    #error_grad = np.zeros((n_feature + 1, n_class))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    train_data1, labeli = args
    train_data = np.concatenate((np.ones([int(train_data1.shape[0]), 1]),train_data1), axis=1)
    n_data = train_data.shape[0]
    weights = params.reshape([716, 10])
    error = 0

    # Calculating the exp values respective to each class
    exp_list = []
    # Calculating individual W.T * X for each class
    for i in range(0, 10):
        exp_list.append(np.exp(np.dot(train_data, weights.T[i])))

    exp_sum = np.zeros(exp_list[0].shape)
    exp_prob = np.zeros([n_data, 10])

    # Calculating sum of 10 arrays W.T * X  for each class
    for i in range(0, 10):
        exp_sum = np.add(exp_sum, exp_list[i])

    # Calculating probabilities for each class
    for i in range(0, 10):
        exp_prob[::, i] = np.divide(exp_list[i], exp_sum)

    log_theta = -1 * np.log(exp_prob)

    for n in range(0, n_data):
        for k in range(labeli.shape[1]):
            error = error + (labeli[n][k] * log_theta[n][k])

    error = (error / float(n_data))
    error_grad = ((np.dot((exp_prob - labeli).T, train_data)) / float(n_data)).T.flatten()

    return error, error_grad


def mlrPredict(W, data):
    """
     mlrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix

    """

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    label = np.zeros((data.shape[0], 1))
    r = sigmoid(np.dot((np.concatenate(((np.ones([int(data.shape[0]), 1])), data), axis=1)), W))
    for i in range(label.shape[0]):
        label[i] = np.argmax(r[i])

    return label

"""
Script for Logistic Regression
"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

# number of classes
n_class = 10

# number of training samples
n_train = train_data.shape[0]

# number of features
n_feature = train_data.shape[1]

Y = np.zeros((n_train, n_class))
for i in range(n_class):
    Y[:, i] = (train_label == i).astype(int).ravel()

# Logistic Regression with Gradient Descent
W = np.zeros((n_feature + 1, n_class))
initialWeights = np.zeros((n_feature + 1, 1))
opts = {'maxiter': 100}
for i in range(n_class):
    labeli = Y[:, i].reshape(n_train, 1)
    args = (train_data, labeli)
    nn_params = minimize(blrObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
    W[:, i] = nn_params.x.reshape((n_feature + 1,))

# Find the accuracy on Training Dataset
predicted_label = blrPredict(W, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label = blrPredict(W, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label = blrPredict(W, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')

"""
Script for Support Vector Machine
"""

print('\n\n--------------SVM-------------------\n\n')
from sklearn import svm
from sklearn.metrics import accuracy_score

print('\n Linear Kernel')
clf= svm.SVC(kernel='linear')
clf.fit(train_data,train_label.ravel())
predicted_label=clf.predict(train_data)
print('\n Training set Accuracy:')
print(accuracy_score(predicted_label, train_label)*100)
predicted_label=clf.predict(validation_data)
print('\n Validation set Accuracy:')
print(accuracy_score(predicted_label, validation_label)*100)
predicted_label=clf.predict(test_data)
print('\n Test set Accuracy:')
print(accuracy_score(predicted_label, test_label)*100)


# rbf and gamma = 1
print('\n RBF and GAMMA =1')
clf= svm.SVC(kernel='rbf', gamma = 1)
clf.fit(train_data,train_label.ravel())
predicted_label=clf.predict(train_data)
print('\n Training set Accuracy:')
print(accuracy_score(predicted_label, train_label)*100)
predicted_label=clf.predict(validation_data)
print('\n Validation set Accuracy:')
print(accuracy_score(predicted_label, validation_label)*100)
predicted_label=clf.predict(test_data)
print('\n Test set Accuracy:')
print(accuracy_score(predicted_label, test_label)*100)


# rbf and gamma = default
print('\n RBF and GAMMA =DEFAULT')
clf= svm.SVC(kernel='rbf')
clf.fit(train_data,train_label.ravel())
predicted_label=clf.predict(train_data)
print('\n Training set Accuracy:')
print(accuracy_score(predicted_label, train_label)*100)
predicted_label=clf.predict(validation_data)
print('\n Validation set Accuracy:')
print(accuracy_score(predicted_label, validation_label)*100)
predicted_label=clf.predict(test_data)
print('\n Test set Accuracy:')
print(accuracy_score(predicted_label, test_label)*100)


# rbf and gamma = default, C=1

print('\n C=1')
clf= svm.SVC(C=1,kernel='rbf')
clf.fit(train_data,train_label.ravel())
predicted_label=clf.predict(train_data)
print('\n Training set Accuracy:')
print(accuracy_score(predicted_label, train_label)*100)
predicted_label=clf.predict(validation_data)
print('\n Validation set Accuracy:')
print(accuracy_score(predicted_label, validation_label)*100)
predicted_label=clf.predict(test_data)
print('\n Test set Accuracy:')
print(accuracy_score(predicted_label, test_label)*100)


# rbf and gamma = default, C=10
print('\n C=10')
clf= svm.SVC(C=10,kernel='rbf')
clf.fit(train_data,train_label.ravel())
predicted_label=clf.predict(train_data)
print('\n Training set Accuracy:')
print(accuracy_score(predicted_label, train_label)*100)
predicted_label=clf.predict(validation_data)
print('\n Validation set Accuracy:')
print(accuracy_score(predicted_label, validation_label)*100)
predicted_label=clf.predict(test_data)
print('\n Test set Accuracy:')
print(accuracy_score(predicted_label, test_label)*100)



# rbf and gamma = default, C=20
print('\n C=20')
clf= svm.SVC(C=20,kernel='rbf')
clf.fit(train_data,train_label.ravel())
predicted_label=clf.predict(train_data)
print('\n Training set Accuracy:')
print(accuracy_score(predicted_label, train_label)*100)
predicted_label=clf.predict(validation_data)
print('\n Validation set Accuracy:')
print(accuracy_score(predicted_label, validation_label)*100)
predicted_label=clf.predict(test_data)
print('\n Test set Accuracy:')
print(accuracy_score(predicted_label, test_label)*100)

# rbf and gamma = default, C=30
print('\n C=30')
clf= svm.SVC(C=30,kernel='rbf')
clf.fit(train_data,train_label.ravel())
predicted_label=clf.predict(train_data)
print('\n Training set Accuracy:')
print(accuracy_score(predicted_label, train_label)*100)
predicted_label=clf.predict(validation_data)
print('\n Validation set Accuracy:')
print(accuracy_score(predicted_label, validation_label)*100)
predicted_label=clf.predict(test_data)
print('\n Test set Accuracy:')
print(accuracy_score(predicted_label, test_label)*100)

# rbf and gamma = default, C=40
print('\n C=40')
clf= svm.SVC(C=40,kernel='rbf')
clf.fit(train_data,train_label.ravel())
predicted_label=clf.predict(train_data)
print('\n Training set Accuracy:')
print(accuracy_score(predicted_label, train_label)*100)
predicted_label=clf.predict(validation_data)
print('\n Validation set Accuracy:')
print(accuracy_score(predicted_label, validation_label)*100)
predicted_label=clf.predict(test_data)
print('\n Test set Accuracy:')
print(accuracy_score(predicted_label, test_label)*100)

# rbf and gamma = default, C=50
print('\n C=50')
clf= svm.SVC(C=50,kernel='rbf')
clf.fit(train_data,train_label.ravel())
predicted_label=clf.predict(train_data)
print('\n Training set Accuracy:')
print(accuracy_score(predicted_label, train_label)*100)
predicted_label=clf.predict(validation_data)
print('\n Validation set Accuracy:')
print(accuracy_score(predicted_label, validation_label)*100)
predicted_label=clf.predict(test_data)
print('\n Test set Accuracy:')
print(accuracy_score(predicted_label, test_label)*100)

# rbf and gamma = default, C=60
print('\n C=60')
fclf= svm.SVC(C=60,kernel='rbf')
clf.fit(train_data,train_label.ravel())
predicted_label=clf.predict(train_data)
print('\n Training set Accuracy:')
print(accuracy_score(predicted_label, train_label)*100)
predicted_label=clf.predict(validation_data)
print('\n Validation set Accuracy:')
print(accuracy_score(predicted_label, validation_label)*100)
predicted_label=clf.predict(test_data)
print('\n Test set Accuracy:')
print(accuracy_score(predicted_label, test_label)*100)

# rbf and gamma = default, C=70
print('\n C=70')
clf= svm.SVC(C=70,kernel='rbf')
clf.fit(train_data,train_label.ravel())
predicted_label=clf.predict(train_data)
print('\n Training set Accuracy:')
print(accuracy_score(predicted_label, train_label)*100)
predicted_label=clf.predict(validation_data)
print('\n Validation set Accuracy:')
print(accuracy_score(predicted_label, validation_label)*100)
predicted_label=clf.predict(test_data)
print('\n Test set Accuracy:')
print(accuracy_score(predicted_label, test_label)*100)

# rbf and gamma = default, C=80
print('\n C=80')
fclf= svm.SVC(C=80,kernel='rbf')
clf.fit(train_data,train_label.ravel())
predicted_label=clf.predict(train_data)
print('\n Training set Accuracy:')
print(accuracy_score(predicted_label, train_label)*100)
predicted_label=clf.predict(validation_data)
print('\n Validation set Accuracy:')
print(accuracy_score(predicted_label, validation_label)*100)
predicted_label=clf.predict(test_data)
print('\n Test set Accuracy:')
print(accuracy_score(predicted_label, test_label)*100)

# rbf and gamma = default, C=90
print('\n C=90')
clf= svm.SVC(C=90,kernel='rbf')
clf.fit(train_data,train_label.ravel())
predicted_label=clf.predict(train_data)
print('\n Training set Accuracy:')
print(accuracy_score(predicted_label, train_label)*100)
predicted_label=clf.predict(validation_data)
print('\n Validation set Accuracy:')
print(accuracy_score(predicted_label, validation_label)*100)
predicted_label=clf.predict(test_data)
print('\n Test set Accuracy:')
print(accuracy_score(predicted_label, test_label)*100)

# rbf and gamma = default, C=100
print('\n C=100')
clf= svm.SVC(C=100,kernel='rbf')
clf.fit(train_data,train_label.ravel())
predicted_label=clf.predict(train_data)
print('\n Training set Accuracy:')
print(accuracy_score(predicted_label, train_label)*100)
predicted_label=clf.predict(validation_data)
print('\n Validation set Accuracy:')
print(accuracy_score(predicted_label, validation_label)*100)
predicted_label=clf.predict(test_data)
print('\n Test set Accuracy:')
print(accuracy_score(predicted_label, test_label)*100)

# -*- coding: utf-8 -*-
"""
import pylab as pl
import numpy as np

x1 = [1,10,20,30,40,50,60,70,80,90,100]
y1 = [94.29,97.13, 97.95, 98.37, 98.70, 99.00, 99.19,99.34,99.43,99.54,99.61]
x2 = [1,10,20,30,40,50,60,70,80,90,100]
y2 = [94.02,96.18,96.9,97.1,97.23,97.31,97.38,97.36,97.39,97.36,97.41]
x3 = [1,10,20,30,40,50,60,70,80,90,100]
y3 = [94.42,96.1,96.67,97.04,97.19,97.19,97.16,97.26,97.33,97.34,97.4]

my_xticks = ['C-1','C-10','C-20','C-30','C-40','C-50','C-60','C-70','C-80','C-90','C-100']
pl.xticks(x1, my_xticks)
plot1 = pl.plot(x1, y1, 'r',label="train")
plot2 = pl.plot(x2, y2, 'g',label="Validation")
plot3 = pl.plot(x3, y3, 'b',label="test")

pl.title('Accuracies VS C')
# make axis labels
pl.xlabel('Values of C')
pl.ylabel('Accuracies')
#pl.xticks(np.arange(min(x1)-1, max(x1)+1, 10))
pl.legend(loc='upper right')
# set axis limits
pl.xlim(1, 100)
pl.ylim(92, 102)
#pl.legend([plot1,plot2,plot3], ["red line", "green circles","abc"])
# show the plot on the screen
pl.show()
"""

#Script for Extra Credit Part

# FOR EXTRA CREDIT ONLY
W_b = np.zeros((n_feature + 1, n_class))
initialWeights_b = np.zeros((n_feature + 1, n_class))
opts_b = {'maxiter': 100}

args_b = (train_data, Y)
nn_params = minimize(mlrObjFunction, initialWeights_b, jac=True, args=args_b, method='CG', options=opts_b)
W_b = nn_params.x.reshape((n_feature + 1, n_class))

# Find the accuracy on Training Dataset
predicted_label_b = mlrPredict(W_b, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label_b == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label_b = mlrPredict(W_b, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_b == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label_b = mlrPredict(W_b, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_b == test_label).astype(float))) + '%')

f1 = open('params.pickle', 'wb') 
pickle.dump(W, f1) 
f1.close()

f2 = open('params_bonus.pickle', 'wb')
pickle.dump(W_b, f2)
f2.close()
