import numpy as np
import pandas as pd
import sklearn.svm as svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import sklearn.svm as svm 
import matplotlib.pyplot as plt


    #reading dataset
def get_dataset(all_data):
    
    features = ["volatile acidity","residual sugar","chlorides","total sulfur dioxide","density","pH","sulphates","alcohol"]
   

    data = pd.read_csv(all_data, header=0)
    labels = data["quality"]
    dataset = data[features]
    
    return dataset, labels


x_data, y_data = get_dataset("winequality-red.csv")
#split data set

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size= .75, shuffle=True)
 


#Logistic Regression
logReg_model = LogisticRegression()
logReg_model.fit(x_train, y_train)
logReg_predict = logReg_model.predict(x_test)

# SVM
gamma = .005
c = 1
svm_model = svm.SVC(kernel='poly', C=c,gamma=gamma)
svm_model.fit(x_train, y_train)
svm_predict = svm_model.predict(x_test)

results = 'Logistic Regression Results:\n\nConfusion Matrix:\n' + (metrics.confusion_matrix(y_test, logReg_predict)).__str__() + '\n'+ (metrics.classification_report(y_test, logReg_predict)).__str__() +'\n\n'

results += 'SVM Results with gamma = ' + gamma.__str__()+ ' and C = '+ c.__str__() + '\n\nConfusion Matrix:\n' + (metrics.confusion_matrix(y_test, svm_predict)).__str__() + '\n' + (metrics.classification_report(y_test, svm_predict)).__str__() +'\n\n'

print (results)



