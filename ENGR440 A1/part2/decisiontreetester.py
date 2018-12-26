import re
import numpy as np
import pickle

testing_set_filename=input('Enter the testing set filename\n')
f=open(testing_set_filename,'r+')
testing_set_file=f.read()
f.close()
testing_set_input=re.findall(r'(.+)',testing_set_file)

testing_set=[]
for i in testing_set_input:
    a=re.findall(r'(\S+)',i)
    testing_set.append(a)
testing_set[0]=testing_set[0][0]+'/'+testing_set[0][1]

b=testing_set.pop(0)
testing_set[0].append(b)

for j in range(len(testing_set)):
    if(j>0):
        c=testing_set[j].pop(0)
        testing_set[j].append(c)

for m in range(len(testing_set)):
    for n in range(len(testing_set[m])):
        if(testing_set[m][n]=='true'):
            testing_set[m][n]=1
        elif(testing_set[m][n]=='false'):
            testing_set[m][n]=0
        elif(testing_set[m][n]=='live'):
            testing_set[m][n]=1
        elif(testing_set[m][n]=='die'):
            testing_set[m][n]=0

x=testing_set.pop(0)

pickle_in=open('dtreeclassifier.pickle','rb')
tree=pickle.load(pickle_in)

def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0
 

def evaluate_algorithm(testset, algorithm, *args):
    test_set=[]
    for row in testset:
        row_copy = list(row)
        test_set.append(row_copy)
        row_copy[-1] = None
        predicted = algorithm(test_set, *args)
    actual = testset[-1]
    accuracy = accuracy_metric(actual, predicted)
    return accuracy
    
def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']
 

def decision_tree(test, max_depth, min_size):
    predictions = list()
    for row in test:
        prediction = predict(tree, row)
        predictions.append(prediction)
    return(predictions)


max_depth = 100
min_size = 1
accuracy_final = evaluate_algorithm(testing_set, decision_tree, max_depth, min_size)
print('Accuracy: %s' % accuracy_final)





 
