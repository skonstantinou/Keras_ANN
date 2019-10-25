#!/usr/bin/env python
#uproot: https://www.indico.shef.ac.uk/event/11/contributions/338/attachments/281/319/rootTutorialWeek5_markhod_2018.pdf
import keras
import uproot
import numpy
import pandas
import ROOT
import array
import os
import math
import json
import random as rn
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

import plot
import func

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #Disable AVX/FMA Warning

#Main
# Definitions
filename = "histograms-TT_19var.root"
tfile    = ROOT.TFile.Open(filename)
debug    = 0
seed     = 1234
nepochs  = 5000
nbatch   = 500

if 0:
    nepochs = 10
    nbatch = 5000

# Setting the seed for numpy-generated random numbers
numpy.random.seed(seed)
# Setting the seed for python random numbers
rn.seed(seed)
# Setting the graph-level random seed.
tf.set_random_seed(seed)

#Signal and background branches
signal     = uproot.open(filename)["treeS"]
background = uproot.open(filename)["treeB"]

# Input list
inputList = []
inputList.append("TrijetPtDR")
inputList.append("TrijetDijetPtDR")
inputList.append("TrijetBjetMass")
inputList.append("TrijetLdgJetBDisc")
inputList.append("TrijetSubldgJetBDisc")
inputList.append("TrijetBJetLdgJetMass")
inputList.append("TrijetBJetSubldgJetMass")
inputList.append("TrijetMass")
inputList.append("TrijetDijetMass")
inputList.append("TrijetBJetBDisc")
inputList.append("TrijetSoftDrop_n2")
inputList.append("TrijetLdgJetCvsL")
inputList.append("TrijetSubldgJetCvsL")
inputList.append("TrijetLdgJetPtD")
inputList.append("TrijetSubldgJetPtD")
inputList.append("TrijetLdgJetAxis2")
inputList.append("TrijetSubldgJetAxis2")
inputList.append("TrijetLdgJetMult")
inputList.append("TrijetSubldgJetMult")

nInputs = len(inputList)

#Signal and background dataframes
df_signal     = signal.pandas.df(inputList)
df_background = background.pandas.df(inputList)

nsignal = len(df_signal.index)
print "=== Number of signal events: ", nsignal

#Signal and background datasets
dset_signal     = df_signal.values
dset_background = df_background.values

ds_signal     = pandas.DataFrame(data=dset_signal,columns=inputList)
ds_background = pandas.DataFrame(data=dset_background,columns=inputList)

# Concat signal, background datasets
df_signal = df_signal.assign(signal=1)
df_background = df_background.assign(signal=0)

df_list = [df_signal, df_background]
df_all = pandas.concat(df_list)

dset_signal = df_signal.values
dset_background = df_background.values
dataset = df_all.values

numpy.random.seed(seed)

#https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
# Define keras model
model = Sequential()
model.add(Dense(36, input_dim = nInputs))
model.add(Activation('relu'))
model.add(Dense(nInputs))
model.add(Activation('relu'))
model.add(Dense(1))
#model.add(Activation('softmax'))
model.add(Activation('sigmoid'))

model.summary()

#Split data into input (X) and output (Y)
X = dataset[:2*nsignal,0:nInputs]
Y = dataset[:2*nsignal,nInputs:]
#
X_signal     = dset_signal[:nsignal, 0:nInputs]
X_background = dset_background[:nsignal, 0:nInputs]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.5, random_state=seed, shuffle=True)

earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=50)
callbacks_list = [earlystop]

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc']) #acc: accuracy #soti: Optimize loss function, optimizer

modelName = "Model_%s.h5" % (filename.replace(".root",""))
model.save(modelName)

hist = model.fit(
    X_train,
    Y_train,
    validation_data=(X_test, Y_test),
    epochs=nepochs,    #soti: Optimize epochs!
    batch_size=nbatch, #soti: Optimize batch size!
    shuffle=False,
    verbose=1,
    callbacks=callbacks_list
)

if not debug:
    modelName = "Model_%s_trained.h5" % (filename.replace(".root",""))
    model.save(modelName)

    # serialize model to JSON
    model_json = model.to_json()
    with open("model_architecture.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights('model_weights.h5', overwrite=True)
    model.save(modelName)
    
pred_train = model.predict(X_train)
pred_test = model.predict(X_test)
pred_signal = model.predict(X_signal)
pred_background = model.predict(X_background)

XY_train     = numpy.concatenate((X_train, Y_train), axis=1)
XY_test      = numpy.concatenate((X_test, Y_test), axis=1)

x_train_S = XY_train[XY_train[:,nInputs] == 1]; x_train_S = x_train_S[:,0:nInputs]
x_train_B = XY_train[XY_train[:,nInputs] == 0]; x_train_B = x_train_B[:,0:nInputs]
x_test_S  = XY_test[XY_test[:,nInputs] == 1];   x_test_S  = x_test_S[:,0:nInputs]
x_test_B  = XY_test[XY_test[:,nInputs] == 0];   x_test_B  = x_test_B[:,0:nInputs]
pred_train_S =  model.predict(x_train_S)
pred_train_B =  model.predict(x_train_B)
pred_test_S  =  model.predict(x_test_S)
pred_test_B  =  model.predict(x_test_B)

dirName = plot.getDirName("TopTag")

X_signal     = dset_signal[:nsignal, 0:nInputs]
X_background = dset_background[:nsignal, 0:nInputs]

func.PlotOutput(pred_signal, pred_background, dirName, "Output_SB.pdf", 1)
func.PlotOutput(pred_train, pred_test, dirName, "Output_pred.pdf", 0)
func.PlotOutput(pred_train_S, pred_train_B, dirName, "Output_SB_train.pdf", 1)
func.PlotOutput(pred_test_S, pred_test_B, dirName, "Output_SB_test.pdf", 1)

# Calculate efficiency
htrain_s, htest_s, htrain_b, htest_b = func.PlotOvertrainingTest(pred_train_S, pred_test_S, pred_train_B, pred_test_B, dirName, "OvertrainingTest.pdf")
func.PlotEfficiency(htest_s, htest_b, dirName, "Efficiency.pdf")



