import numpy as np
from math import sqrt
from collections import Counter
import pandas as pd
import random

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

train_set={2:[],4:[],6:[]}
test_set={2:[],4:[],6:[]}

for i in train_data:
    train_set[i[-1]].append(i[:-1])
    
for i in test_data:
    test_set[i[-1]].append(i[:-1])

def k_nearest_neighbours(data,predict,k=5):
    distances=[]
    for group in data:
        for features in data[group]:
            euclidean_distance=np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_distance,group])
    votes=[i[-1] for i in sorted(distances)[:k]]
    #print(votes)
    vote_result=Counter(votes).most_common(1)[0][0]
    return vote_result
    
correct=0
total=0

for group in test_set:
    for data in test_set[group]:
        vote=k_nearest_neighbours(train_set,data,k=1)
        if group==vote:
            correct+=1
        total+=1
print('Accuracy:',correct/total)

