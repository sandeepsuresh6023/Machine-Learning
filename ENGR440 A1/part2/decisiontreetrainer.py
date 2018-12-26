import re
import numpy as np
import pickle

training_set_filename=input('Enter the training set filename\n')
f=open(training_set_filename,'r+')
training_set_file=f.read()
f.close()
training_set_input=re.findall(r'(.+)',training_set_file)

training_set=[]
for i in training_set_input:
    a=re.findall(r'(\S+)',i)
    training_set.append(a)
training_set[0]=training_set[0][0]+'/'+training_set[0][1]

b=training_set.pop(0)
training_set[0].append(b)

for j in range(len(training_set)):
    if(j>0):
        c=training_set[j].pop(0)
        training_set[j].append(c)

for m in range(len(training_set)):
    for n in range(len(training_set[m])):
        if(training_set[m][n]=='true'):
            training_set[m][n]=1
        elif(training_set[m][n]=='false'):
            training_set[m][n]=0
        elif(training_set[m][n]=='live'):
            training_set[m][n]=1
        elif(training_set[m][n]=='die'):
            training_set[m][n]=0

x=training_set.pop(0)            

def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right
 

def gini_index(groups, classes):
    n_instances = float(sum([len(group) for group in groups]))
    gini = 0.0
    for group in groups:
        size = float(len(group))
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        gini += (1.0 - score) * (size / n_instances)
    return gini
 
def get_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    for index in range(len(dataset[0])-1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index':b_index, 'value':b_value, 'groups':b_groups}
 

def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)
 

def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del(node['groups'])
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth+1)
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth+1)
 

def build_tree(train, max_depth, min_size):
    root = get_split(train)
    split(root, max_depth, min_size, 1)
    return root
 

def print_tree(node,depth=0):
    if isinstance(node, dict):
        print('%s[%s < %.3f]' % ((depth*' ', x[node['index']], node['value'])))
        print_tree(node['left'], depth+1)
        print_tree(node['right'], depth+1)
    else:
        print('%s[%s]' % ((depth*' ', node)))

tree=build_tree(training_set,100,1)
with open('dtreeclassifier.pickle','wb') as f:
    pickle.dump(tree,f)
print_tree(tree)
