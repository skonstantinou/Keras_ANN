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

def PlotOutput(Y_train, Y_test, saveDir, saveName, isSB):
    ROOT.gStyle.SetOptStat(0)
    canvas = ROOT.TCanvas("canvas", "canvas",0,0,800,800)
    canvas.cd()
    canvas.SetLogy()
    h1=ROOT.TH1F('train', '', 50, 0.0, 1.)
    print "===PlotOutput: train"
    for r in Y_train:
        #print r
        h1.Fill(r)

    h2=ROOT.TH1F('test', '', 50, 0.0, 1.)
    print "===PlotOutput: test"
    for r in Y_test:
        #print r
        h2.Fill(r)

    if 0:
        h1.Scale(1./h1.Integral())
        h2.Scale(1./h2.Integral())
    print "=== INTEGRAL: ", h1.Integral(), h2.Integral()
    ymax = max(h1.GetMaximum(), h2.GetMaximum())

    h1.SetLineColor(ROOT.kRed)
    h1.SetMaximum(ymax*1.1)
    h1.GetXaxis().SetTitle("Output")
    h1.Draw("HIST")

    h2.SetLineColor(ROOT.kGreen)
    h2.SetMaximum(ymax*1.1)
    h2.GetXaxis().SetTitle("Output")
    h2.Draw("HIST SAME")

    graph = plot.CreateGraph([0, 0], [0, ymax*1.1])
    #graph.Draw("same")
    
    leg=plot.CreateLegend(0.6, 0.75, 0.9, 0.85)
    if isSB:
        leg.AddEntry(h1, "signal","l")
        leg.AddEntry(h2, "background","l")
    else:
        leg.AddEntry(h1, "train","l")
        leg.AddEntry(h2, "test","l")

    leg.Draw()

    SavePlot(canvas, saveDir, saveName)
    canvas.Close()
    return

def PlotTarget(Y_train, Y_test, saveDir, saveName, isSB):
    ROOT.gStyle.SetOptStat(0)
    canvas = ROOT.TCanvas("canvas", "canvas",0,0,800,800)
    canvas.cd()
    h1=ROOT.TH1F('train', '', 50, -0.1, 1.1)
    for r in Y_train:
        h1.Fill(r)

    h2=ROOT.TH1F('test', '', 50, -0.1, 1.1)
    for r in Y_test:
        h2.Fill(r)

    ymax = max(h1.GetMaximum(), h2.GetMaximum())

    h1.SetLineColor(ROOT.kRed)
    h1.SetMaximum(ymax*2.1)
    h1.GetXaxis().SetTitle("Output")
    h1.Draw("HIST")

    h2.SetLineColor(ROOT.kGreen)
    h2.SetMaximum(ymax*2.1)
    h2.GetXaxis().SetTitle("Output")
    h2.Draw("HIST SAME")

    h3=ROOT.TH1F('test', '', 50, -0.1, 1.1)
    for i in range(len(Y_test.tolist())):
        h3.Fill((Y_train[i] + Y_test[i])/2.)

    
    h3.SetLineColor(ROOT.kBlack)
    h3.Draw("same")
    graph = plot.CreateGraph([0.5, 0.5], [0, ymax*2.1])
    graph.Draw("same")

    leg=plot.CreateLegend(0.6, 0.75, 0.9, 0.85)
    if isSB:
        leg.AddEntry(h1, "signal","l")
        leg.AddEntry(h2, "background","l")
    else:
        leg.AddEntry(h1, "train","l")
        leg.AddEntry(h2, "test","l")

    leg.Draw()

    #saveName = "Output.pdf"
    SavePlot(canvas, saveDir, saveName)
    canvas.Close()
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

dirName = getDirName()
if not os.path.exists(dirName):
    os.mkdir(dirName)
    print "Directory " , dirName ,  " Created "
else:
    print "Output saved under", dirName

X_signal     = dset_signal[:nsignal, 0:nInputs]
X_background = dset_background[:nsignal, 0:nInputs]

PlotOutput(pred_signal, pred_background, dirName, "Output_SB.pdf", 1)
PlotOutput(pred_train, pred_test, dirName, "Output_pred.pdf", 0)
PlotOutput(pred_train_S, pred_train_B, dirName, "Output_SB_train.pdf", 1)
PlotOutput(pred_test_S, pred_test_B, dirName, "Output_SB_test.pdf", 1)

# Calculate efficiency
htrain_s, htest_s, htrain_b, htest_b = PlotOvertrainingTest(pred_train_S, pred_test_S, pred_train_B, pred_test_B, dirName, "OvertrainingTest.pdf")
PlotEfficiency(htest_s, htest_b, dirName, "Efficiency.pdf")



