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

config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1,
                        allow_soft_placement=True, device_count = {'CPU': 1})
session = tf.Session(config=config)
K.set_session(session)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #Disable AVX/FMA Warning
#Default values:
LAMBDA = 1
NEPOCHS = 200
OPTIMIZER = "Adam"
LR = 0.01 # Default value for SGD
helpText = "* lambda: Penalty factor for loss function. Describes the mass dependence."
from optparse import OptionParser
parser = OptionParser(helpText)

parser.add_option("--lr",dest="lr",  default=LR,  type = float,   help="Learning Rate (Default: %s)" % LR)
parser.add_option("--opt",dest="opt",  default=OPTIMIZER,   help="Optimizer (Default: %s)" % OPTIMIZER)
parser.add_option("--lam",dest="lam",  default=LAMBDA,  type = int,   help="Lambda (Default: %s)" % LAMBDA)
parser.add_option("--nepochs", dest="nepochs", default = NEPOCHS, type = int, help ="Number of epochs (Default: %s)" % NEPOCHS)
opt, args = parser.parse_args()


###########################################################
# Plot Loss Function
###########################################################

def PlotLossFunction(losses, nepochs, saveDir):
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

    hLf.SetLineColor(ROOT.kRed)
    hLf.SetMarkerColor(ROOT.kRed)
    hLf.SetMarkerStyle(8)
    hLf.SetMarkerSize(0.8)
    hLf.SetMaximum(ymax)
    hLf.SetMinimum(ymin)
    hLf.Draw("pa")
    hLf.GetXaxis().SetTitle("# epoch")
    hLf.GetYaxis().SetTitle("Loss")
    
    hLr.SetLineColor(ROOT.kBlue)
    hLr.SetMarkerColor(ROOT.kBlue)
    hLr.SetMarkerStyle(8)
    hLr.SetMarkerSize(0.8)
    hLr.SetMaximum(ymax)
    hLr.SetMinimum(ymin)
    hLr.Draw("p same")
    hLr.GetXaxis().SetTitle("# epoch")
    hLr.GetYaxis().SetTitle("Loss")

    hLfr.SetLineColor(ROOT.kGreen)
    hLfr.SetMarkerColor(ROOT.kGreen)
    hLfr.SetMarkerStyle(8)
    hLfr.SetMarkerSize(0.8)
    hLfr.SetMaximum(ymax)
    hLfr.SetMinimum(ymin)
    hLfr.Draw("p same")
    hLfr.GetXaxis().SetTitle("# epoch")
    hLfr.GetYaxis().SetTitle("Loss")

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
nepochs  = opt.nepochs
nbatch   = 100
lam = opt.lam

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
Dx = Dense(32, activation="relu")(inputs)
Dx = Dense(32, activation="relu")(Dx)
Dx = Dense(32, activation="relu")(Dx)
Dx = Dense(1, activation="sigmoid")(Dx)
D = Model(inputs=[inputs], outputs=[Dx])
D.compile(loss="binary_crossentropy", optimizer="adam")
#print "=== D model: summary"
#D.summary()

print "=== D model: Fit"
D.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, verbose=1)

y_predD = D.predict(X_test)
for i in range(20):
    print("Predicted=%s (%s) (%s)"% (target_test[i], y_predD[i], Y_test[i]))

###################################################
#Pretraining
###################################################

inputs = Input(shape=(X_train.shape[1],))

Dx = Dense(32, activation="relu")(inputs)
Dx = Dense(32, activation="relu")(Dx)
Dx = Dense(32, activation="relu")(Dx)
Dx = Dense(1, activation="sigmoid")(Dx)
D = Model(inputs=[inputs], outputs=[Dx])

Rx = Dx
Rx = Dense(16, activation="relu")(Rx)
Rx = Dense(1, activation = "relu")(Rx)
R = Model(inputs=[inputs], outputs=[Rx])

#D.compile(loss=[make_loss_D(c=1.0)], optimizer="sgd")
D.compile(loss="binary_crossentropy", optimizer="adam")

###################################################
# Combined models
###################################################

opt_DRf = opt.opt
opt_DfR = opt.opt

if opt.opt == "SGD":    
    opt_DRf = SGD(momentum=0.0)
    opt_DfR = SGD(momentum=0.0)

DRf = Model(inputs=[inputs], outputs=[D(inputs), R(inputs)])
make_trainable(R, False)
make_trainable(D, True)
DRf.compile(loss=[make_loss_D(c=1.0), make_loss_R(c=-lam)], optimizer=opt_DRf)


DfR = Model(inputs=[inputs], outputs=[R(inputs)])
make_trainable(R, True)
make_trainable(D, False)
DfR.compile(loss=[make_loss_R(c=1.0)], optimizer=opt_DfR)

# Pretraining of D
make_trainable(R, False)
make_trainable(D, True)

print "=== D model: Fit"
D.fit(X_train, Y_train, epochs=10)

# Pretraining of R
make_trainable(R, True)
make_trainable(D, False)

print "=== DfR model: Fit"
DfR.fit(X_train, target_train, epochs=10)

###################################################
# Adversarial Network
###################################################

losses = {"L_f": [], "L_r": [], "L_f - L_r": []}

batch_size = 128

for i in range(nepochs): #201
    l = DRf.evaluate(X_test, [Y_test, target_test], verbose=1)
    lf = l[1][None][0]
    lr = -l[2][None][0]
    delta_lfr = l[0][None][0]
    losses["L_f - L_r"].append(delta_lfr)
    losses["L_f"].append(lf)
    losses["L_r"].append(lr)
    #print(losses["L_r"][-1] / lam)
    #print lf, lr, delta_lfr
    print "=== Epoch: %s / %s. Losses: Lf = %.3f, Lr = %.3f, (L_f - L_r) = %.3f" % (i, nepochs, lf, lr, delta_lfr)
    
#    if i % 5 == 0:
#        plot_losses(i, losses)

    # Fit D
    make_trainable(R, False)
    make_trainable(D, True)
    indices = numpy.random.permutation(len(X_train))[:batch_size]
    DRf.train_on_batch(X_train[indices], [Y_train[indices], target_train[indices]])
        
    # Fit R
    make_trainable(R, True)
    make_trainable(D, False)
    DfR.fit(X_train, target_train, batch_size=batch_size, epochs=1, verbose=0)

y_pred = D.predict(X_test)
roc_auc_score(Y_test, y_pred)

for i in range(50):
    print("Predicted=%s (%s) (%s)"% (target_test[i], y_pred[i], Y_test[i]))

saveDir = plot.getDirName("TopTag")
PlotLossFunction(losses, nepochs, saveDir)


# serialize weights to HDF5
D.save_weights('modelANN_weights_lam%s.h5' % (opt.lam), overwrite=True)
D.save("modelANN_lam%s.h5" % (opt.lam))
# serialize model to JSON
model_json = D.to_json()
with open("modelANN_architecture_lam%s.json" % (opt.lam), "w") as json_file:
    json_file.write(model_json)

