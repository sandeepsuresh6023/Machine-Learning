import numpy as np
import pandas as pd
import random
from sklearn.neural_network import MLPClassifier

training_set_filename=input('Enter the training set filename\n')
f=open(training_set_filename,'r+')
training_set_file=f.read()
f.close()
training_set_file=training_set_file.replace("  ",",")
f=open(training_set_filename,'w')
f.write(training_set_file)
f.close()
training_set_df=pd.read_csv(training_set_filename,names=["SL","SW","PL","PW","Class"])
training_set_df.replace('Iris-setosa',2,inplace=True)
training_set_df.replace('Iris-versicolor',4,inplace=True)
training_set_df.replace('Iris-virginica',6,inplace=True)
train_data=training_set_df.astype(float).values.tolist()
X_train=[]
Y_train=[]
for i in train_data:
    X_train.append(i[:-1])
    Y_train.append(i.pop())

test_set_filename=input('Enter the test set filename\n')
f=open(test_set_filename,'r+')
test_set_file=f.read()
f.close()
test_set_file=test_set_file.replace("  ",",")
f=open(test_set_filename,'w')
f.write(test_set_file)
f.close()
test_set_df=pd.read_csv(test_set_filename,names=["SL","SW","PL","PW","Class"])
test_set_df.replace('Iris-setosa',2,inplace=True)
test_set_df.replace('Iris-versicolor',4,inplace=True)
test_set_df.replace('Iris-virginica',6,inplace=True)
test_data=test_set_df.astype(float).values.tolist()
X_test=[]
Y_test=[]
for i in test_data:
    X_test.append(i[:-1])
    Y_test.append(i.pop())

neural_network=MLPClassifier(max_iter=10000,hidden_layer_sizes=(3,),activation='logistic',random_state=42,alpha=1)
neural_network.fit(X_train,Y_train)
print("2 stands for Iris-setosa     4 stands for Iris-versicolor    6 stands for Iris-virginica")
print(neural_network.predict(X_test))
print("Test set accuracy ",neural_network.score(X_test,Y_test))
