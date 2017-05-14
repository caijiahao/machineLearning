#!/usr/bin/env python
# encoding: utf-8
import sys

from pybrain.datasets import ClassificationDataSet
from pybrain.tools.validation import ModuleValidator, CrossValidator
from pybrain.utilities import percentError
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer

alldata = ClassificationDataSet(2, 1, nb_classes=2)
alldata.addSample([-1, -1], [0])
alldata.addSample([-1, -1], [0])
alldata.addSample([-1, -1], [0])
alldata.addSample([-1, -1], [0])
alldata.addSample([-1, -1], [0])

alldata.addSample([1, 1], [1])
alldata.addSample([1, 1], [1])
alldata.addSample([1, 1], [1])
alldata.addSample([1, 1], [1])
alldata.addSample([1, 1], [1])

tstdata, trndata = alldata.splitWithProportion(0.25)
trndata._convertToOneOfMany()
tstdata._convertToOneOfMany()

# We can also examine the dataset
print "Number of training patterns: ", len(trndata)
print "Input and output dimensions: ", trndata.indim, trndata.outdim
print "First sample (input, target, class):"
print trndata['input'][0], trndata['target'][0], trndata['class'][0]

fnn     = buildNetwork( trndata.indim, 5, trndata.outdim, recurrent=False )
trainer = BackpropTrainer( fnn, dataset=trndata, momentum=0.1, verbose=True, weightdecay=0.01 )

# I am not sure about this, I don't think my production code is implemented like this
modval = ModuleValidator()
trainer.trainEpochs(20)
trainer.trainOnDataset(dataset=trndata)
cv = CrossValidator( trainer, trndata, n_folds=5, valfunc=modval.MSE )
print "MSE %f" %( cv.validate() )
