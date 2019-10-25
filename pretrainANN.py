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

def getDirName():
    dirName = "TopTag"
    dirName = dirName.replace(".", "p")
    dirName = "/afs/cern.ch/user/s/skonstan/public/html/"+dirName
    return dirName

def SavePlot(canvas, saveDir, saveName):
    savePath = "%s/%s" % (saveDir, saveName)
    saveURL  = savePath.replace("/afs/cern.ch/user/s/","https://cmsdoc.cern.ch/~")
    saveURL  = saveURL.replace("/public/html/","/")
    canvas.SaveAs(savePath)
    savePath = savePath.replace("pdf","root")
    canvas.SaveAs(savePath)
    print "=== ", saveURL
    return

def PlotOvertrainingTest(Y_train_S, Y_test_S, Y_train_B, Y_test_B, saveDir, saveName):
    ROOT.gStyle.SetOptStat(0)
    canvas = ROOT.TCanvas("canvas", "canvas",0,0,800,800)
    canvas.cd()
    canvas.SetLogy()

    hList = []
    DataList = [Y_train_S, Y_test_S, Y_train_B, Y_test_B]
    ymax = 0
    nbins = 500
    htrain_s=ROOT.TH1F('train_s', '', nbins, 0.0, 1.)
    hList.append(htrain_s)
    htest_s=ROOT.TH1F('test_s', '', nbins, 0.0, 1.)
    hList.append(htest_s)
    htrain_b=ROOT.TH1F('train_b', '', nbins, 0.0, 1.)
    hList.append(htrain_b)
    htest_b=ROOT.TH1F('test_b', '', nbins, 0.0, 1.)
    hList.append(htest_b)
            
    for i in range(len(DataList)):
        for r in DataList[i]:
            hList[i].Fill(r)        
        ymax = max(ymax, hList[i].GetMaximum())
    
    htrain_s1 = htrain_s.Clone("train_s")
    htrain_b1 = htrain_b.Clone("train_b")
    htest_s1  = htest_s.Clone("test_s")
    htest_b1  = htest_b.Clone("test_b")

    drawStyle = "HIST SAME"
    leg=plot.CreateLegend(0.50, 0.65, 0.85, 0.85)    
    for h in hList:
        h.SetMaximum(ymax*2)        
        h.GetXaxis().SetTitle("Output")
        h.GetYaxis().SetTitle("Entries")
        h.Rebin(10)
        # Legend
        legText, legStyle = plot.GetLegendStyle(h.GetName())
        leg.AddEntry(h, legText, legStyle)
        
        plot.ApplyStyle(h)
        h.Draw(plot.DrawStyle(h.GetName())+" SAME")

    graph = plot.CreateGraph([0, 0], [0, ymax*1.1])
    #graph.Draw("same")
    leg.Draw()

    SavePlot(canvas, saveDir, saveName)
    canvas.Close()
    return htrain_s1, htest_s1, htrain_b1, htest_b1

def PlotEfficiency(htest_s, htest_b, saveDir, saveName):
    ROOT.gStyle.SetOptStat(0)
    canvas = ROOT.TCanvas("canvas", "canvas",0,0,800,800)
    canvas.cd()
    nbins = htest_s.GetNbinsX()
    # Initialize sigma variables
    sigmaAll = ROOT.Double(0.0)
    sigmaSel = ROOT.Double(0.0)

    All_s = htest_s.IntegralAndError(0, nbins+1, sigmaAll, "")
    All_b = htest_b.IntegralAndError(0, nbins+1, sigmaAll, "")

    eff_s = []; eff_b = []; xvalue = []; error = []
    for i in range(0, nbins+1):
        Sel_s = htest_s.IntegralAndError(i, nbins+1, sigmaSel, "")
        Sel_b = htest_b.IntegralAndError(i, nbins+1, sigmaSel, "")

        if (All_s <= 0):
            All_s = 1
            Sel_s = 0
        if (All_b <= 0):
            All_b = 1
            Sel_b = 0

        eff_s.append(Sel_s/All_s)
        eff_b.append(Sel_b/All_b)
        error.append(0)
        xvalue.append(htest_s.GetBinCenter(i))
        
    graph_s = plot.GetGraph(xvalue, eff_s, error, error, error, error)
    graph_b = plot.GetGraph(xvalue, eff_b, error, error, error, error)

    graph_s.SetLineColor(ROOT.kBlue)
    graph_s.SetMarkerColor(ROOT.kBlue)
    graph_s.SetMarkerStyle(8)
    graph_s.SetMarkerSize(0.4)
    graph_b.SetLineColor(ROOT.kRed)
    graph_b.SetMarkerColor(ROOT.kRed)
    graph_b.SetMarkerStyle(8)
    graph_b.SetMarkerSize(0.4)

    graph_s.SetMinimum(0)
    graph_b.SetMinimum(0)
    
    #
    h_signif0=ROOT.TH1F('signif0', '', nbins, 0.0, 1.)
    h_signif1=ROOT.TH1F('signif1', '', nbins, 0.0, 1.)

    sign = []
    sigmaSel_s = ROOT.Double(0.0)
    sigmaSel_b = ROOT.Double(0.0)

    for i in range(0, nbins+1):
        # Get selected events                                                                                                                                                                       
        sSel = htest_s.IntegralAndError(i, nbins+1, sigmaSel_s, "")
        bSel = htest_b.IntegralAndError(i, nbins+1, sigmaSel_b, "")
        # Calculate Significance                                                                                                                                                                    
        _sign0 = sSel/math.sqrt(sSel+bSel)
        _sign1 = 2*(math.sqrt(sSel+bSel) - math.sqrt(bSel))
        sign.append(_sign0)
        h_signif0.Fill(htest_s.GetBinCenter(i), _sign0)        
        h_signif1.Fill(htest_s.GetBinCenter(i), _sign1)        

    graph_signif = plot.GetGraph(xvalue, sign, error, error, error, error)
    graph_signif.SetLineColor(ROOT.kGreen)
    graph_signif.SetMarkerColor(ROOT.kGreen)
    graph_signif.SetMarkerStyle(8)
    graph_signif.SetMarkerSize(0.4)

    h_signif0.SetLineColor(ROOT.kGreen)
    h_signif0.SetMarkerColor(ROOT.kGreen)
    h_signif0.SetLineWidth(3)

    h_signif1.SetLineColor(ROOT.kGreen+3)
    h_signif1.SetMarkerColor(ROOT.kGreen+3)
    h_signif1.SetLineWidth(3)
    
    #=== Scale Significance
    maxSignif0 = h_signif0.GetMaximum()
    maxSignif1 = h_signif1.GetMaximum()
    maxSignif = max(maxSignif0, maxSignif1)

    graph_s.SetTitle("")
    graph_b.SetTitle("")

    
    h_signifScaled0 = h_signif0.Clone("signif0")
    h_signifScaled0.Scale(1./float(maxSignif))

    h_signifScaled1 = h_signif1.Clone("signif1")
    h_signifScaled1.Scale(1./float(maxSignif))

    ymax = max(h_signifScaled0.GetMaximum(), h_signifScaled1.GetMaximum())
    graph_s.SetMaximum(ymax*1.1)
    graph_b.SetMaximum(ymax*1.1)
    h_signifScaled0.SetMaximum(ymax*1.1)
    h_signifScaled1.SetMaximum(ymax*1.1)

    graph_s.SetMinimum(0)
    graph_b.SetMinimum(0)
    h_signifScaled0.SetMinimum(0)
    h_signifScaled1.SetMinimum(0)

    graph_s.GetXaxis().SetTitle("Output")
    graph_s.GetYaxis().SetTitle("Efficiency")    
    graph_b.GetXaxis().SetTitle("Output")
    graph_b.GetYaxis().SetTitle("Efficiency")    

    h_signifScaled0.GetXaxis().SetTitle("Output")
    h_signifScaled0.GetYaxis().SetTitle("Efficiency")
    h_signifScaled1.GetXaxis().SetTitle("Output")
    h_signifScaled1.GetYaxis().SetTitle("Efficiency")
    #Draw
    h_signifScaled0.Draw("HIST")
    h_signifScaled1.Draw("HIST SAME")
    graph_s.Draw("P SAME")
    graph_b.Draw("P SAME")

    #Legend
    leg=plot.CreateLegend(0.50, 0.25, 0.85, 0.45)    
    leg.AddEntry(graph_s, "Signal Efficiency", "l")
    leg.AddEntry(graph_b, "Bkg Efficiency", "l")
    leg.AddEntry(h_signifScaled0, "S/#sqrt{S+B}", "l")
    leg.AddEntry(h_signifScaled1, "2#times(#sqrt{S+B} - #sqrt{B})", "l")
    leg.Draw()
    # Define Right Axis (Significance)
    signifColor = ROOT.kGreen+2
    rightAxis = ROOT.TGaxis(1, 0, 1, 1.1, 0, 1.1*maxSignif, 510, "+L")
    #rightAxis.SetWmax(1.1*maxSignif)
    rightAxis.SetLineColor ( signifColor )
    rightAxis.SetLabelColor( signifColor )
    rightAxis.SetTitleColor( signifColor )
    rightAxis.SetTitleOffset(1.3)
    rightAxis.SetTitle( "Significance" )
    rightAxis.Draw()
        
    SavePlot(canvas, saveDir, saveName)
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
nepochs  = 5000
nbatch   = 100
lam = 10

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
    print("Predicted=%s (%s)"% (target_test[i], y_predD[i]))

###################################################
#Pretraining
###################################################

inputs = Input(shape=(X_train.shape[1],))

Dx = Dense(32, activation="relu")(inputs)
Dx = Dense(32, activation="relu")(Dx)
Dx = Dense(32, activation="relu")(Dx)
Dx = Dense(1, activation="sigmoid")(Dx)
D = Model(inputs=[inputs], outputs=[Dx])

#Rx = Dx
#Rx = Dense(16, activation="relu")(Rx)
#Rx = Dense(32, activation="relu")(Rx)
#Rx = Dense(32, activation="relu")(Rx)
#Rx = Dense(1, activation="sigmoid")(Rx)
#R = Model(inputs=[inputs], outputs=[Rx])

inp_target = Input(shape=(y_predD.shape[1],))
Rx = Dense(16, activation="relu")(inp_target) #epochs = 10, dense = 16
#Rx = Dense(32, activation="relu")(Rx)
#Rx = Dense(32, activation="relu")(Rx)
Rx = Dense(1, activation = "relu")(Rx)
R = Model(inputs=[inp_target], outputs=[Rx])

print "===D model"
D.summary()
print "===R model"
R.summary()
#R.compile(loss=make_loss_R(c=1), optimizer="sgd")
R.compile(loss="msle", optimizer="adam")
R.fit(Y_train, target_train, validation_data=(Y_test, target_test), epochs=10, verbose=1)

y_pred = R.predict(y_predD)
for i in range(20):
    print("Predicted=%.3f (%.3f) (%.3f) (%.3f)"% (target_test[i], y_pred[i], Y_test[i], y_predD[i]))

'''
#print "=== D model: summary"
#D.summary()
#print "=== R model: summary"
#R.summary()

opt_D = "adam"
D.compile(loss=[make_loss_D(c=1.0)], optimizer=opt_D)

opt_DRf = "adam"
DRf = Model(inputs=[inputs], outputs=[D(inputs), R(inputs)])
make_trainable(R, False)
make_trainable(D, True)
DRf.compile(loss=[make_loss_D(c=1.0), make_loss_R(c=-lam)], optimizer=opt_DRf)

opt_DfR = "adam"
DfR = Model(inputs=[inputs], outputs=[R(inputs)])
make_trainable(R, True)
make_trainable(D, False)
DfR.compile(loss=[make_loss_R(c=1.0)], optimizer=opt_DfR)

#print "=== DfR model: summary"
#DfR.summary()

#print "=== DRf model: summary"
#DRf.summary() 
#UserWarning: Discrepancy between trainable weights and collected trainable weights, did you set `model.trainable` without calling `model.compile` after ?

# Pretraining of D
make_trainable(R, False)
make_trainable(D, True)

print "==D model: Fit"
D.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, verbose=1)

y_pred = D.predict(X_test)
for i in range(20):
    print("Predicted=%s (%s)"% (target_test[i], y_pred[i]))

# Pretraining of R
make_trainable(R, True)
make_trainable(D, False)

print "=== Dfr model: Fit"
DfR.fit(X_train, target_train, validation_data=(X_test, target_test), 10, verbose=1)

y_pred = DfR.predict(X_test)
for i in range(20):
    print("Predicted=%s (%s)"% (target_test[i], y_pred[i]))


#Adversarial
losses = {"L_f": [], "L_r": [], "L_f - L_r": []}

batch_size = 500

for i in range(10):
    l = DRf.evaluate(X_test, [Y_test, target_test], verbose=1)
    losses["L_f - L_r"].append(l[0][None][0])
    losses["L_f"].append(l[1][None][0])
    losses["L_r"].append(-l[2][None][0])
    print(losses["L_r"][-1] / lam)
    
#    if i % 5 == 0:
#        plot_losses(i, losses)

    # Fit D
    make_trainable(R, False)
    make_trainable(D, True)
    indices = numpy.random.permutation(len(X_train))[:batch_size]
    print "=== Drf model: train on batch"
    DRf.train_on_batch(X_train[indices], [Y_train[indices], target_train[indices]])
        
    # Fit R
    make_trainable(R, True)
    make_trainable(D, False)
    print "=== Dfr model: Fit"
    DfR.fit(X_train, target_train, validation_data=(X_test, target_test), batch_size=batch_size, epochs=1, verbose=1)

y_pred = D.predict(X_test)
for i in range(20):
    print("Predicted=%s (%s)"% (target_test[i], y_pred[i]))

'''
