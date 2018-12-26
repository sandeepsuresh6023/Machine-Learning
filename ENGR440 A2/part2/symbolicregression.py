import re
import operator
import math
import random

import numpy

#individual is not an input value. It is a candidate solution
#population is the set of all individuals
#primitives include both operators and terminals

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

#Extracting the data from 'regression.txt' and seperating it into input values and labels
f=open('regression.txt','r+')
data=f.read()
data_set=re.findall(r'([-]?\d+[.]?\d+)(\s+)(\d+[.]?\d+)',data)
x_values=[]
y_values=[]
for i in data_set:
    x_values.append(float(i[0]))
    y_values.append(float(i[2]))

# Defining a new division function so as not to get an error when dividing by zero
def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

#The primitive set is created and all operators are added
pset = gp.PrimitiveSet("MAIN", 1)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.sin, 1)
pset.addEphemeralConstant("rand101", lambda: random.randint(-1,1))
pset.renameArguments(ARG0='x')

#creator function in deap is used to create classes for fitness and individuals

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

#The evalSymbReg function evaluates the fitness of an individual
def evalSymbReg(x_values, y_values, individual):
    func = toolbox.compile(expr=individual)
    sqerrors=[]
    for i in range(len(x_values)):
        sqerrors.append((func(x_values[i]) - y_values[i])**2)
    return math.fsum(sqerrors) / len(x_values),
    
#The toolbox is created with all the tools necessary for evolution
toolbox.register("evaluate", evalSymbReg, x_values, y_values)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

random.seed(318)

pop = toolbox.population(n=300)
hof = tools.HallOfFame(1)
    
stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
stats_size = tools.Statistics(len)
mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
mstats.register("avg", numpy.mean)
mstats.register("std", numpy.std)
mstats.register("min", numpy.min)
mstats.register("max", numpy.max)

pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 100, stats=mstats, halloffame=hof, verbose=True)
print(log)
print('\n')
print('Obtained function i.e individual in hall of fame')
print('################################################')
print(hof[0].__str__())

