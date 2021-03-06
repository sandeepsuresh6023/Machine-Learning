import re
import numpy as np
import random
import copy

f=open('image.data','r+')
s=f.read()
pixel_set=[]
image_set=re.findall(r'(P1)\n(#X|#O)\n(10 10)\n(\d+)\n(\d+)\n',s)
real_class=[]

for i in image_set:
    x=i[3]+i[4]
    new_set=[]
    for j in x:
        new_set.append(int(j))
    pixel_set.append(new_set)
    if(i[1]=='#X'):
        real_class.append(1)
    elif(i[1]=='#O'):
        real_class.append(0)
    
connections=[]
for f in range(100):
    connections.append(random.choice([True,False]))

feature_set=[]
for j in pixel_set:
    feature_inner_set=[1]
    for k in range(50):
        c1=random.randint(0, 99)
        c2=random.randint(0, 99)
        c3=random.randint(0, 99)
        c4=random.randint(0, 99)
        c=[c1,c2,c3,c4]
        total=0
        for r in c:
            if(j[r]==connections[r]):
                total=total+1
        if(total>=3):
            feature_inner_set.append(1)
        else:
            feature_inner_set.append(0)
    feature_set.append(feature_inner_set)

def predict(inputs,weights):
    activation=0.0
    for i,w in zip(inputs,weights):
        activation += i*w
    return 1.0 if activation>=0.0 else 0.0
    
def accuracy(matrix,weights):
    num_correct = 0.0
    preds=[]
    for i in range(len(matrix)):
        pred = predict(matrix[i][:-1],weights)
        preds.append(pred)
        if pred==matrix[i][-1]:
            num_correct+=1.0
    print("Predictions:",preds)
    return num_correct/float(len(matrix))

def train_weights(matrix,weights,nb_epoch=10,l_rate=1.00):
    for epoch in range(nb_epoch):
        cur_acc = accuracy(matrix,weights)
        print("\nEpoch %d "%epoch)
        print("Accuracy: ",cur_acc)
        for i in range(len(matrix)):
            prediction = predict(matrix[i][:-1],weights)
            error = matrix[i][-1]-prediction
            for j in range(len(weights)):
                weights[j] = weights[j]+(l_rate*error*matrix[i][j])
    return weights                

data_matrix=copy.deepcopy(feature_set)
for q in range(len(data_matrix)):
    data_matrix[q].append(real_class[q])

epoch=1000
learning_rate=1.00
weights_set=[]
for i in range(len(data_matrix[0])):
    weights_set.append(random.uniform(0.0,1.0))

print(train_weights(matrix=data_matrix,weights=weights_set,nb_epoch=epoch,l_rate=learning_rate))
#print(feature_set)

 
 
 
 
 
 
 
 
        
