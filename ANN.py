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
from array import array
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import keras.backend as K
from keras.optimizers import SGD, Adam
from sklearn.metrics import roc_auc_score

import plot
import func
import tdrstyle

# Do not display canvases
ROOT.gROOT.SetBatch(ROOT.kTRUE)
# Disable screen output info
ROOT.gROOT.ProcessLine( "gErrorIgnoreLevel = 1001;")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #Disable AVX/FMA Warning

#Run in single CPU: this ensures reproducible results!
config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1,
                        allow_soft_placement=True, device_count = {'CPU': 1})
session = tf.Session(config=config)
K.set_session(session)

#Default values:
LAMBDA       = 1
NEPOCHS      = 200
OPTIMIZER    = "SGD"
GRIDX        = False
GRIDY        = False

LR = 0.01 # Default value for SGD
helpText = "* lambda: Penalty factor for loss function. Describes the mass dependence."
from optparse import OptionParser
parser = OptionParser(helpText)

parser.add_option("--lr",dest="lr",  default=LR,  type = float,   help="Learning Rate (Default: %s)" % LR)
parser.add_option("--opt",dest="opt",  default=OPTIMIZER,   help="Optimizer (Default: %s)" % OPTIMIZER)
parser.add_option("--lam",dest="lam",  default=LAMBDA,  type = int,   help="Lambda (Default: %s)" % LAMBDA)
parser.add_option("--nepochs", dest="nepochs", default = NEPOCHS, type = int, help ="Number of epochs (Default: %s)" % NEPOCHS)
parser.add_option("--gridX", dest="gridX", action="store_true", default=GRIDX, help="Enable the x-axis grid lines [default: %s]" % GRIDX)
parser.add_option("--gridY", dest="gridY", action="store_true", default=GRIDY, help="Enable the y-axis grid lines [default: %s]" % GRIDY)

opt, args = parser.parse_args()

style = tdrstyle.TDRStyle() 
style.setOptStat(False) 
style.setGridX(opt.gridX)
style.setGridY(opt.gridY)

###########################################################
# Plot Loss Function
###########################################################

def PlotLossFunction(losses, nepochs, saveDir):
    
    def ApplyStyle(h, ymin, ymax, color):
        h.SetLineColor(color)
        h.SetMarkerColor(color)
        h.SetMarkerStyle(8)
        h.SetMarkerSize(0.8)
        h.SetMaximum(ymax)
        h.SetMinimum(ymin)
        h.GetXaxis().SetTitle("# epoch")
        h.GetYaxis().SetTitle("Loss")

    ROOT.gStyle.SetOptStat(0)
    canvas = ROOT.TCanvas("canvas", "canvas",0,0,800,800)
    canvas.cd()
    xarr = []
    for i in range(1, nepochs+1):
        xarr.append(i)

    hLf  = ROOT.TGraph(len(xarr), array('d', xarr), array('d', losses["L_f"]))
    hLr  = ROOT.TGraph(len(xarr), array('d', xarr), array('d', losses["L_r"]))
    hLfr = ROOT.TGraph(len(xarr), array('d', xarr), array('d', losses["L_f - L_r"]))

    ymax = max(hLf.GetHistogram().GetMaximum(), hLr.GetHistogram().GetMaximum(), hLfr.GetHistogram().GetMaximum())
    ymin = min(hLf.GetHistogram().GetMinimum(), hLr.GetHistogram().GetMinimum(), hLfr.GetHistogram().GetMinimum())
    ymax = ymax + 1
    ymin = ymin - 1

    
    ApplyStyle(hLf, ymin, ymax, ROOT.kRed)
    ApplyStyle(hLr, ymin, ymax, ROOT.kBlue)
    ApplyStyle(hLfr, ymin, ymax, ROOT.kGreen+1)
    
    hLf.Draw("pa")
    hLr.Draw("p same")
    hLfr.Draw("p same")

    leg=plot.CreateLegend(0.6, 0.75, 0.9, 0.85)
    leg.AddEntry(hLf, "L_{f}","ple")
    leg.AddEntry(hLr,"L_{r}","ple")
    leg.AddEntry(hLfr,"L_{f} - #lambda L_{r}","ple")
    leg.Draw()

    #canvas.SetLogy()

    saveName = "Loss_lam%s_opt%s.pdf" %(opt.lam, opt.opt)#str(opt.lr).replace(".","p"))
    plot.SavePlot(canvas, saveDir, saveName)
    canvas.Close()
    return

###########################################################
# Get List of inputs
###########################################################

def getInputs():
    inputList = []
    inputList.append("TrijetPtDR")
    inputList.append("TrijetDijetPtDR")
    inputList.append("TrijetBjetMass")
    inputList.append("TrijetLdgJetBDisc")
    inputList.append("TrijetSubldgJetBDisc")
    inputList.append("TrijetBJetLdgJetMass")
    inputList.append("TrijetBJetSubldgJetMass")
    #inputList.append("TrijetMass")
    #inputList.append("TrijetDijetMass")
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


###########################################################
# Get Classifier
###########################################################

def getClassifier(model_clf, nInputs):
    model_clf.add(Dense(32, input_dim =  nInputs)) 
    model_clf.add(Activation('relu'))
    model_clf.add(Dense(32))
    model_clf.add(Activation('relu'))
    model_clf.add(Dense(32))
    model_clf.add(Activation('relu'))
    model_clf.add(Dense(1))
    model_clf.add(Activation('sigmoid'))
    return model_clf

###########################################################
# Get Regressor
###########################################################

#def getRegressor(model_reg,nInputs):    
def getRegressor(model_clf,model_reg):    
    print nInputs
    #model_reg = getClassifier(model_reg, nInputs)
    model_reg.add(model_clf)
    model_reg.add(Dense(16))
    model_reg.add(Activation("relu"))
    model_reg.add(Dense(1))
    model_reg.add(Activation("relu"))
    return model_reg

###########################################################
# Get Classifier Layers
###########################################################

def getClassifierLayers(inputs):
    Dx = Dense(32)(inputs)
    Dx = Activation("relu")(Dx)
    Dx = Dense(32)(Dx)
    Dx = Activation("relu")(Dx)
    Dx = Dense(32)(Dx)
    Dx = Activation("relu")(Dx)
    Dx = Dense(1)(Dx)
    Dx = Activation("sigmoid")(Dx)
    return Dx

###########################################################
# Get Regressor Layers
###########################################################

def getRegressorLayers(Dx, inputs):
    Rx = Dx
    Rx = Dense(16)(Rx)
    Rx = Activation("relu")(Rx)
    Rx = Dense(1)(Rx)
    Rx = Activation("relu")(Rx)    
    return Rx

###########################################################
# Make selected layers trainiable / not trainable
# For not trainable layers, the weights are not updated
###########################################################

def make_trainable(model, isTrainable):
    model.trainable = isTrainable
    for l in model.layers:
        l.trainable = isTrainable

###########################################################
# Get classifier's loss function
###########################################################

def getLoss_clf(c):
    def loss_C(y_true, y_pred):
        return c * K.binary_crossentropy(y_true, y_pred)
    return loss_C

###########################################################
# Get regressor's loss function
# c factor: controls the mass dependence
###########################################################

def getLoss_reg(c):
    def loss_R(z_true, z_pred):
        return c * keras.losses.mean_squared_logarithmic_error(z_true, z_pred)
    return loss_R

#Main
# Definitions
filename = "histograms-TT_19var.root"
tfile    = ROOT.TFile.Open(filename)
debug    = 0
seed     = 1234
nepochs  = opt.nepochs
nbatch   = 100
lam      = opt.lam

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

#Get signal and background dataframes and assign signal/background flag
df_signal     = signal.pandas.df(inputList)
df_background = background.pandas.df(inputList)
#
df_signal = df_signal.assign(signal=1)
df_background = df_background.assign(signal=0)

nsignal = len(df_signal.index)
print "=== Number of signal events: ", nsignal

#Get signal and background datasests
dset_signal     = df_signal.values
dset_background = df_background.values

# Get inputs for signal and background
X_signal     = dset_signal[:nsignal, 0:nInputs]
X_background = dset_background[:nsignal, 0:nInputs]

# Concat signal, background datasets
dataset = pandas.concat([df_signal, df_background]).values
dataset_target_all = pandas.concat([signal.pandas.df(["TrijetMass"]), background.pandas.df(["TrijetMass"])]).values

#Split data into input (X) and output (Y)
X = dataset[:2*nsignal,0:nInputs]
Y = dataset[:2*nsignal,nInputs:]

# Get target (top mass)
target = dataset_target_all[:2*nsignal, :]

numpy.random.seed(seed)

# Get training and test samples
X_train, X_test, Y_train, Y_test, target_train, target_test = train_test_split(
    X, Y, target, test_size=0.5, random_state=seed, shuffle=True)


###################################################
# Define classifier
###################################################
inputs = Input(shape=(X_train.shape[1],))
model_clf = Sequential()
model_clf = getClassifier(model_clf, nInputs)

model_clf.compile(loss="binary_crossentropy", optimizer="adam")

print "=== Classifier: Fit"
model_clf.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, verbose=1)


###################################################
# Define Models
###################################################
'''
# Classifier (characterizes events as signal or background)
# Inputs: List of input variables
# Output: classifier
# Regressor: Predicts the top-quark mass from the classifiers output
'''

inputs = Input(shape=(X_train.shape[1],))

model_clf = Sequential()
model_clf = getClassifier(model_clf, nInputs)
model_reg = Sequential()
model_reg = getRegressor(model_clf, model_reg)

model_clf.compile(loss="binary_crossentropy", optimizer="adam")

###################################################
# Combined models
###################################################

opt_comb = opt.opt
opt_adv = opt.opt

if opt.opt == "SGD":    
    opt_comb = SGD(momentum=0.0)
    opt_adv = SGD(momentum=0.0)

# Combined network: updates the classifier's weights
model_comb = Model(inputs=[inputs], outputs=[model_clf(inputs), model_reg(inputs)])
make_trainable(model_reg, False)
make_trainable(model_clf, True)
# Compile combined model
model_comb.compile(loss=[getLoss_clf(c=1.0), getLoss_reg(c=-lam)], optimizer=opt_comb)

# Adversary network: predicts top-quark mass (classifier's layers are not trainable!)
model_adv = Model(inputs=[inputs], outputs=[model_reg(inputs)])
make_trainable(model_reg, True)
make_trainable(model_clf, False)
model_adv.compile(loss=[getLoss_reg(c=1.0)], optimizer=opt_adv)

###################################################
# Pretrain Classifier
###################################################
make_trainable(model_reg, False)
make_trainable(model_clf, True)

print "=== Classifier: Fit"
model_clf.fit(X_train, Y_train, epochs=10)

###################################################
# Pretrain  Regressor
# Here the output is the top-quark mass (target)
###################################################
make_trainable(model_reg, True)
make_trainable(model_clf, False)

print "=== Adversarial Network: Fit"
model_adv.fit(X_train, target_train, epochs=10)

###################################################
# Train the model
###################################################

losses = {"L_f": [], "L_r": [], "L_f - L_r": []}

batch_size = 128

for i in range(nepochs): #201
    l = model_comb.evaluate(X_test, [Y_test, target_test], verbose=1)
    lf = l[1][None][0]
    lr = -l[2][None][0]
    lfr = l[0][None][0]
    losses["L_f - L_r"].append(lfr)
    losses["L_f"].append(lf)
    losses["L_r"].append(lr)
    print "=== Epoch: %s / %s. Losses: Lf = %.3f, Lr = %.3f, (L_f - L_r) = %.3f" % (i, nepochs, lf, lr, lfr)
    
    # Fit Classifier (with updated weights) to minimize joint loss function
    make_trainable(model_reg, False)
    make_trainable(model_clf, True)
    indices = numpy.random.permutation(len(X_train))[:batch_size]
    model_comb.train_on_batch(X_train[indices], [Y_train[indices], target_train[indices]])
        
    # Fit Regressor (with updated weights) to minimize Lr
    make_trainable(model_reg, True)
    make_trainable(model_clf, False)
    model_adv.fit(X_train, target_train, batch_size=batch_size, epochs=1, verbose=0)

###################################################
# Test classifier's performance
###################################################

y_pred = model_clf.predict(X_test)
roc_auc_score(Y_test, y_pred)

for i in range(50):
    print("Predicted=%s (%s) (%s)"% (target_test[i], y_pred[i], Y_test[i]))

###################################################
# Plot Loss function vs # epoch
###################################################

saveDir = plot.getDirName("TopTag")
PlotLossFunction(losses, nepochs, saveDir)

###################################################
# Classifier: save model
###################################################

# serialize weights to HDF5
model_clf.save_weights('modelANN_weights_lam%s.h5' % (opt.lam), overwrite=True)
model_clf.save("modelANN_lam%s.h5" % (opt.lam))

# serialize architecture to JSON
model_json = model_clf.to_json()
with open("modelANN_architecture_lam%s.json" % (opt.lam), "w") as json_file:
    json_file.write(model_json)

# write weights and architecture in txt file
func.WriteModel(model_clf, model_json, "modelANN_lam%s.txt" % (opt.lam))

