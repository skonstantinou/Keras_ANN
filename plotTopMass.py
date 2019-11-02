#!/usr/bin/env python
#Useful regression tutorial: https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/
import numpy
import pandas
import keras
import ROOT
import array
import math
import os
import uproot
from keras.models import Sequential
from keras.layers import Dense, Dropout #https://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/
from keras.wrappers.scikit_learn import KerasRegressor
from keras.optimizers import Adam
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
# Regression predictions
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import load_model
from sklearn.externals import joblib
from keras.models import model_from_json
#from rootpy.root2array import fill_hist_with_ndarray

import plot
import tdrstyle
import func

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #Disable AVX/FMA Warning


def main():
    ROOT.gStyle.SetOptStat(0)

    style = tdrstyle.TDRStyle() 
    style.setOptStat(False) 
    style.setGridX(False)
    style.setGridY(False)

    # Definitions
    filename  = "histograms-TT_19var.root"
    debug   = 1
    nprint  = 100
    tfile    = ROOT.TFile.Open(filename)    
    
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
    
    nInputs = len(inputList)
    
    #Signal and background dataframes
    df_signal     = signal.pandas.df(inputList)
    df_background = background.pandas.df(inputList)

    nEvts = len(df_background.index)/10
    print "=== Number of signal events: ", nEvts
    
    #Signal and background datasets
    dset_signal     = df_signal.values
    dset_background = df_background.values
    
    # Concat signal, background datasets
    df_list = [df_signal, df_background]
    df_all = pandas.concat(df_list)
    dataset = df_all.values  
    dataset_target_all = pandas.concat([signal.pandas.df(["TrijetMass"]), background.pandas.df(["TrijetMass"])]).values
    dataset_target_bkg = background.pandas.df(["TrijetMass"]).values
    dataset_target_signal = signal.pandas.df(["TrijetMass"]).values

    X_signal     = dset_signal[:nEvts, 0:nInputs]
    X_background = dset_background[:nEvts, 0:nInputs]

    nEvts = nEvts
    X = dset_background[:nEvts, 0:nInputs]
    target = dataset_target_bkg[:nEvts, :]

    #Load models
    lamValues = [1, 5, 10, 50] # fixme! should be given as option
    colors = [ROOT.kBlack, ROOT.kOrange, ROOT.kBlue, ROOT.kRed, ROOT.kGreen, ROOT.kMagenta, ROOT.kOrange+7]
    canvas = plot.CreateCanvas()
    canvas.cd()
    #canvas.SetLogy()
    
    ymax = 0
    histoList = []
    for lam in lamValues:
        print "Lambda = ", lam
        loaded_model = load_model('models_30Oct/modelANN_lam%s.h5' % (lam))
        #loaded_model = load_model('modelANN_lam%s.h5' % (lam))
        loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
        Y = loaded_model.predict(X, verbose=1)
        Ymass = numpy.concatenate((Y, target), axis=1)

        # Get selected top candidates
        Ymass_sel = Ymass[Ymass[:,0] >= 0.5]
        massSel = Ymass_sel[:, 1]        

        #Plot resutls
        nbins = 100
        xmax  = 1000
        width = float(xmax)/nbins
        histo = ROOT.TH1F('histo_lam%s' %(lam), '', nbins, 0, xmax)
        mass_sel = []
        print "selected entries:", len(massSel)
        for mass in massSel:
            #print "test: %s / %s: m = %s" % (i, len(massSel), mass)
            histo.Fill(mass)

        histoList.append(histo.Clone("histoClone_lam%s" % lam))
        
        #histo.Delete()
        del loaded_model
    
    leg=plot.CreateLegend(0.6, 0.67, 0.9, 0.85)
    
    for i in range(len(histoList)):
        leg.AddEntry(histoList[i], "t#bar{t} (#lambda = %.0f)" %(lamValues[i]) ,"f")
        print "=== Lambda =  %.0f" % lamValues[i]
        histoList[i].Scale(1./(histoList[i].Integral()))
        ymax = max(ymax, histoList[i].GetMaximum())
        histoList[i].GetXaxis().SetTitle("m_{top} (GeV)")
        histoList[i].GetYaxis().SetTitle("Arbitrary Units / %.0f GeV" % (width))
        plot.ApplyStyle(histoList[i], colors[i])
        histoList[i].Draw("HIST same")
    for i in range(len(histoList)):
        histoList[i].SetMaximum(ymax*1.1)
    leg.Draw("same")
    tex1 = plot.Text(" unmatched top candidates",0.85,0.5)
    tex2 = plot.Text(" with output value > 0.50",0.82,0.45)
    tex1.Draw()
    tex2.Draw()
    graph = plot.CreateGraph([173., 173.], [0, ymax*1.1])
    graph.Draw("same")
    dirName = plot.getDirName("TopTag")
            
    plot.SavePlot(canvas, dirName, "TopMassANN")
    
main()
