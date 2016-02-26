# NNet training with linear outlayer
import csv
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
import pybrain as pb
from pybrain.datasets import SupervisedDataSet,ClassificationDataSet
from pybrain.structure import FeedForwardNetwork,LinearLayer, SigmoidLayer,FullConnection
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer

############ ds - dataset ##############

# Load the dataset
data=pd.read_csv('train_preprocessed.csv', sep=',',header=None)
data = data.as_matrix()

total_data_train = data[:,2:][:40000]
hazard_train = data[:,1][:40000]

total_data_test = data[:,2:][40000:]
hazard_test = data[:,1][40000:]

def numpy_to_nnet_data(Input,Output):
 	neuraltrain = SupervisedDataSet(111,1)
	X=[]
	Y=[]
	for x in Input:
		X.append(x)
	for y in Output:
		Y.append(y)
	Z=zip(X,Y)
	for d in Z:
		neuraltrain.addSample(d[0].tolist(),d[1])
	return neuraltrain

############ building nnet ###############

input_to_nnet = numpy_to_nnet_data(total_data_train,hazard_train)
output_to_nnet = numpy_to_nnet_data(total_data_test,hazard_test)

ffnnet = FeedForwardNetwork()

inLayer = LinearLayer(input_to_nnet.indim)
hiddenLayer = SigmoidLayer(56)	# ~111/2 number of hidden units
outLayer = LinearLayer(input_to_nnet.outdim)

ffnnet.addInputModule(inLayer)
ffnnet.addModule(hiddenLayer)
ffnnet.addOutputModule(outLayer)

in_to_hidden = FullConnection(inLayer, hiddenLayer)
hidden_to_out = FullConnection(hiddenLayer, outLayer)

ffnnet.addConnection(in_to_hidden)
ffnnet.addConnection(hidden_to_out)

ffnnet.sortModules()

# weight decay ~ ln(x) = -18


### training ###
wd = 1.523E-08
trainer = BackpropTrainer(ffnnet, dataset=input_to_nnet, learningrate = 0.01, momentum=0.1, verbose=True, weightdecay=wd)

for i in range(2):
	trainer.trainEpochs(10)
	res = trainer.testOnClassData(dataset=output_to_nnet)

print "Training Done"

coeff_list = ffnnet.params.tolist()

resultFile = open("new1.csv",'wb')
wr = csv.writer(resultFile)
wr.writerow(coeff_list)

with open('new1.csv') as f:
    reader = csv.reader(f)
    cols = []
    for row in reader:
        cols.append(row)

############ write in csv file #######
with open('coeff_nnet.csv', 'wb') as f:
    writer = csv.writer(f)
    for i in range(len(max(cols, key=len))):
        writer.writerow([(c[i] if i<len(c) else '') for c in cols])

