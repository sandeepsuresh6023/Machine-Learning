import random
import csv
import itertools
import pandas as pd
import copy

import numpy

from operator import and_
from operator import or_
from operator import not_
from operator import add
from operator import sub
from operator import mul
from operator import lt
from operator import eq

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

#The data is extracted from the 'breast-cancer-wisconsin.data' file by using the pandas library
df=pd.read_csv('breast-cancer-wisconsin.data',names=["id_no","clump_thickness","uni_cell_size","uni_cell_shape","marg_adhesion","single_epi_cell_size","bare_nuclei","bland_chromatin","norm_nucleoli","mitoses","class"])
#All the '?' in the file is replaced by -1 as recommended in the assignment question sheet
df.replace('?',-1,inplace=True)
#The id_no column is removed because it obviously has no bearing on whether a person has cancer or not
df.drop(['id_no'],1,inplace=True)
#All the values are converted to float to ensure that there are no unspecified formats the deap algorithm cannot handle. The data is then shuffled.
full_data=df.astype(float).values.tolist()
random.shuffle(full_data)

#The data is seperated into test and training sets
test_size=0.2
train_data=full_data[:-int(test_size*len(full_data))]
test_data=full_data[-int(test_size*len(full_data)):]

#The content of the training set is stored in training.txt as mentioned in the question
f=open('training.txt','w')
f.write(str(train_data))
f.close()

#The content of the test set is stored in test.txt as mentioned in the question
f=open('test.txt','w')
f.write(str(test_data))
f.close()

#The training set is seperated into input values and labels
Y_train=[]
X_train=copy.deepcopy(train_data)
for i in range(len(X_train)):
    Y_train.append(X_train[i].pop())

Y_train_bool=[]
for i in Y_train:
    if(i==2.0):
        Y_train_bool.append(bool(1))
    elif(i==4.0):
        Y_train_bool.append(bool(0))

#The testing set is seperated into input values and labels        
Y_test=[]
X_test=copy.deepcopy(test_data)
for i in range(len(X_test)):
    Y_test.append(X_test[i].pop())

Y_test_bool=[]
for i in Y_test:
    if(i==2.0):
        Y_test_bool.append(bool(1))
    elif(i==4.0):
        Y_test_bool.append(bool(0))

# Defining a new primitive set for strongly typed GP
pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, 9), bool, "IN")

# The boolean operators are specified
pset.addPrimitive(and_, [bool, bool], bool)
pset.addPrimitive(or_, [bool, bool], bool)
pset.addPrimitive(not_, [bool], bool)

# The floating point operators are specified
# Define a protected division function
def protectedDiv(left, right):
    try: return left / right
    except ZeroDivisionError: return 1

pset.addPrimitive(add, [float,float], float)
pset.addPrimitive(sub, [float,float], float)
pset.addPrimitive(mul, [float,float], float)
pset.addPrimitive(protectedDiv, [float,float], float)

# The logic operators are specified
# Define a new if-then-else function
def if_then_else(input, output1, output2):
    if input: return output1
    else: return output2

pset.addPrimitive(lt, [float, float], bool)
pset.addPrimitive(eq, [float, float], bool)
pset.addPrimitive(if_then_else, [bool, float, float], float)

# The terminals are given
pset.addEphemeralConstant("random_values", lambda: random.random() * 100, float)
pset.addTerminal(False, bool)
pset.addTerminal(True, bool)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

#The toolbox is filled with all the necessary evolution tools
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

#The evalCancer function is fitness function which tests the fitness of an individual
def evalCancer(X_train, Y_train_bool, individual):
    func = toolbox.compile(expr=individual) 
    result = sum(bool(func(*X_train[k])) is bool(Y_train_bool[k]) for k in range(len(Y_train_bool)))
    return result,
    
toolbox.register("evaluate", evalCancer, X_train, Y_train_bool)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)


random.seed(10)
pop = toolbox.population(n=100)
hof = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", numpy.mean)
stats.register("std", numpy.std)
stats.register("min", numpy.min)
stats.register("max", numpy.max)
    
algorithms.eaSimple(pop, toolbox, 0.5, 0.2, 40, stats, halloffame=hof)

print('\n')
print('Obtained Function i.e individual in hall of fame')
print('################################################')
print(hof[0].__str__())

correct=0
total=0
final_function=copy.deepcopy(hof[0].__str__())
for i in range(len(X_train)):
    IN0,IN1,IN2,IN3,IN4,IN5,IN6,IN7,IN8=X_train[i]
    if(eval(final_function)==Y_train_bool[i]):
        correct=correct+1
    total=total+1
print('\n')
print('Accuracy in training set:',correct/total)


correct=0
total=0
for i in range(len(X_test)):
    IN0,IN1,IN2,IN3,IN4,IN5,IN6,IN7,IN8=X_test[i]
    if(eval(final_function)==Y_test_bool[i]):
        correct=correct+1
    total=total+1
print('\n')
print('Accuracy in test set:',correct/total)






