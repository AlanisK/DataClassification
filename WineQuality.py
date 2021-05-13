import numpy as np
import pandas as pd
import sklearn.svm as svm
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
import sklearn.svm as svm 
import matplotlib.pyplot as plt


    #reading dataset
def get_dataset(all_data):
    
    # features = ["volatile acidity","residual sugar","chlorides","total sulfur dioxide","density","pH","sulphates","alcohol"]
    # picking 5 random features to test different accuracies.
    features = ["residual sugar","density","pH","alcohol"]
    features2 = ["volatile acidity","chlorides","total sulfur dioxide","sulphates"]
    features3 = ["volatile acidity","residual sugar","chlorides","total sulfur dioxide"]
    features4 = ["residual sugar", "volatile acidity", "alcohol"]
    features5 = ["alcohol", "volatile acidity", "density", "chlorides"]

    data = pd.read_csv(all_data, header=0)
    labels = np.array(data["quality"])
    dataset = np.array(data[features5])

    count = np.zeros(9)

    for i in range(0, len(labels)):
        index = labels[i]
        count[index] += 1
    


    return dataset, labels, count




x_data, y_data, count = get_dataset("winequality-red.csv")
#split data set


# plt.scatter(x_data, y_data, alpha=.5)
# plt.show()

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size= .75, shuffle=True, random_state=200)
 


#Logistic Regression
logReg_model = LogisticRegression(max_iter=20000)
logReg_model.fit(x_train, y_train)
logReg_predict = logReg_model.predict(x_test)

# SVM
gamma = 1
c = 1
svm_model = svm.SVC(kernel='linear', C=c,gamma=gamma, max_iter=20000)
svm_model.fit(x_train, y_train)
svm_predict = svm_model.predict(x_test)

#Nueral Network
layer_number = 75
NN = MLPClassifier(hidden_layer_sizes=layer_number)
NN.fit(x_train, y_train)
NN_predict = NN.predict(x_test)

results = 'Logistic Regression Results:\n\nConfusion Matrix:\n' + (metrics.confusion_matrix(y_test, logReg_predict)).__str__() + '\n'+ (metrics.classification_report(y_test, logReg_predict)).__str__() +'\n\n'

results += 'SVM Results with gamma = ' + gamma.__str__()+ ' and C = '+ c.__str__() + '\n\nConfusion Matrix:\n' + (metrics.confusion_matrix(y_test, svm_predict)).__str__() + '\n' + (metrics.classification_report(y_test, svm_predict)).__str__() +'\n\n'

results += 'NN Results with layer number of = ' + layer_number.__str__() + '\n\nConfusion Matrix:\n' + (metrics.confusion_matrix(y_test, NN_predict)).__str__() + '\n' + (metrics.classification_report(y_test, NN_predict)).__str__() +'\n\n'

print (results)
plt.bar([i for i in range(0,9)], count, alpha=.5)
plt.show()



