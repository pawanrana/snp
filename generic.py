from libs.utils import load_pandas
import StringIO
import pandas as pd
from pandas.tools.plotting import autocorrelation_plot
from pandas.tools.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from snp import tf_confusion_metrics,training_predictors_tf,training_classes_tf,predictors_tf,test_predictors_tf,test_classes_tf,classes_tf,training_set_size,test_set_size
from fnn import train_NN

count = 0
length = 0
target = 0.0
iterations = 100

y = training_predictors_tf
c = training_classes_tf
y_ = test_predictors_tf
c_ = test_classes_tf
size = len(y.columns)

training_classes_tf = classes_tf[:training_set_size]
test_predictors_tf = predictors_tf[training_set_size:]

def individual(length):
	np.random.choice(size,length, replace=False)
	pass

def population(count, length):
	individual(length)
	pass

def fitness(individual, target):
	pass

def grade_of_generation(pop, target):
	pass

def evolve(pop, target, retain=0.2, random_select=0.05, mutate=0.01):
	pass

p = population(count,length)
for i in xrange(iterations):
	p = evolve(p, target)
	#print grade_of_generation(p, target)

print train_NN(y,c,y_,c_)