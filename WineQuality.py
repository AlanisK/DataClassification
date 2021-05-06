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
    
    features = ["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]
   

    data = pd.read_csv(all_data, header=0)
    labels = data["quality"]
    dataset = data[features]
    
    return dataset, labels


x_data, y_data = get_dataset("winequality-red.csv")
#split data set

x_train, y_train, x_test, y_test = train_test_split(x_data, y_data, train_size= .75, shuffle=True)
 
print(y_data.shape)


#Logistic Regression
logReg_model = LogisticRegression()
logReg_model.fit(x_train, y_train)
logReg_predict = logReg_model.predict(x_test)

# SVM
gamma = 0.001
c = 2
svm_model = svm.SVC(kernel='poly', C=c,gamma=gamma)
svm_model.fit(x_train, y_train)
svm_predict = svm_model.predict(x_test)





