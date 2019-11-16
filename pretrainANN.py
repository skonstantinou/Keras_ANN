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
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import keras.backend as K

from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #Disable AVX/FMA Warning

#Run in single CPU: this ensures reproducible results!
config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1,
                        allow_soft_placement=True, device_count = {'CPU': 1})
session = tf.Session(config=config)
K.set_session(session)

ACT1 = 'relu'
ACT2 = 'relu'
ACT3 = 'sigmoid'
ACT4 = 'sigmoid'
ACT5 = 'sigmoid'
OPT = 'adam'
from optparse import OptionParser
parser = OptionParser(usage="Usage: %prog [options]")
parser.add_option("--act1",dest="act1",  default=ACT1,   help="Activation function 1 (Default: %s)" % ACT1)
parser.add_option("--act2",dest="act2",  default=ACT2,   help="Activation function 2 (Default: %s)" % ACT2)
parser.add_option("--act3",dest="act3",  default=ACT3,   help="Activation function 3 (Default: %s)" % ACT3)
parser.add_option("--act4",dest="act4",  default=ACT4,   help="Activation function 2 (Default: %s)" % ACT4)
parser.add_option("--act5",dest="act5",  default=ACT5,   help="Activation function 3 (Default: %s)" % ACT5)
parser.add_option("--opt",dest="opt",  default=OPT,   help="Regressor: Optimizer (Default: %s)" % OPT)
opt, args = parser.parse_args()

def getInputs():
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
    return inputList

def make_trainable(model, isTrainable):
    model.trainable = isTrainable
    for l in model.layers:
        l.trainable = isTrainable

def make_loss_D(c):
    def loss_D(y_true, y_pred):
        return c * K.binary_crossentropy(y_true, y_pred)
    return loss_D

def make_loss_R(c):
    def loss_R(z_true, z_pred):
        return c * keras.losses.mean_squared_logarithmic_error(z_true, z_pred)
    return loss_R

#Main
# Definitions
filename = "histograms-TT_19var.root"
tfile    = ROOT.TFile.Open(filename)
debug    = 0
seed     = 1234
nepochs  = 5000
nbatch   = 100
lam = 10

print "Classifier: Activations: %s_%s_%s_sigmoid" % (opt.act1, opt.act2, opt.act3)
print "Regressor:  Activations: %s_%s" % (opt.act5, opt.act4)
print "Regressor:  OptimizerL %s" % (opt.opt)
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
inputList = getInputs()
nInputs = len(inputList)

#Signal and background dataframes
df_signal     = signal.pandas.df(inputList)
df_background = background.pandas.df(inputList)

nsignal = len(df_signal.index)
print "=== Number of signal events: ", nsignal

# Concat signal, background datasets
df_signal = df_signal.assign(signal=1)
df_background = df_background.assign(signal=0)

#Signal and background datasets
dset_signal     = df_signal.values
dset_background = df_background.values
#
X_signal     = dset_signal[:nsignal, 0:nInputs]
X_background = dset_background[:nsignal, 0:nInputs]

dataset = pandas.concat([df_signal, df_background]).values
dataset_target_all = pandas.concat([signal.pandas.df(["TrijetMass"]), background.pandas.df(["TrijetMass"])]).values
#Split data into input (X) and output (Y)
X = dataset[:2*nsignal,0:nInputs]
Y = dataset[:2*nsignal,nInputs:]
target = dataset_target_all[:2*nsignal, :]

numpy.random.seed(seed)

# Define keras models

# Get training and test samples
X_train, X_test, Y_train, Y_test, target_train, target_test = train_test_split(
    X, Y, target, test_size=0.5, random_state=seed, shuffle=True)


inputs = Input(shape=(X_train.shape[1],))
Dx = Dense(32, activation=opt.act1)(inputs)
Dx = Dense(32, activation=opt.act2)(Dx)
Dx = Dense(32, activation=opt.act3)(Dx)
Dx = Dense(1, activation='sigmoid')(Dx)
D = Model(inputs=[inputs], outputs=[Dx])
D.compile(loss="binary_crossentropy", optimizer="adam")

print "=== D model: Fit"
D.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, verbose=1)

y_predD = D.predict(X_test)
for i in range(20):
    print("Predicted=%s (%s)"% (target_test[i], y_predD[i]))

###################################################
#Pretraining
###################################################

inputs = Input(shape=(X_train.shape[1],))

Dx = Dense(32, activation=opt.act1)(inputs)
Dx = Dense(32, activation=opt.act2)(Dx)
Dx = Dense(32, activation=opt.act3)(Dx)
Dx = Dense(1, activation='sigmoid')(Dx)
D = Model(inputs=[inputs], outputs=[Dx])

inp_target = Input(shape=(y_predD.shape[1],))
Rx = Dense(16, activation=opt.act4)(inp_target) #epochs = 10, dense = 16
Rx = Dense(1, activation = opt.act5)(Rx)
R = Model(inputs=[inp_target], outputs=[Rx])

print "===D model"
D.summary()
print "===R model"
R.summary()
R.compile(loss="msle", optimizer=opt.opt)
R.fit(Y_train, target_train, validation_data=(Y_test, target_test), epochs=5, verbose=1)

#y_pred = R.predict(y_predD)
#for i in range(20):
#    print("Predicted=%.3f (%.3f) (%.3f) (%.3f)"% (target_test[i], y_pred[i], Y_test[i], y_predD[i]))


