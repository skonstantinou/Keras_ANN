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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #Disable AVX/FMA Warning


def SavePlot(canvas, saveDir, saveName):
    savePath = "%s/%s" % (saveDir, saveName)
    print "savepath = ", savePath
    saveURL  = savePath.replace("/afs/cern.ch/user/s/","https://cmsdoc.cern.ch/~")
    saveURL  = saveURL.replace("/public/html/","/")
    canvas.SaveAs(savePath)
    savePath = savePath.replace("pdf","root")
    canvas.SaveAs(savePath)
    print "=== ", saveURL
    return

def getDirName():
    dirName = "TopTag"
    dirName = dirName.replace(".", "p")
    dirName = "/afs/cern.ch/user/s/skonstan/public/html/"+dirName
    return dirName

def main():
    ROOT.gStyle.SetOptStat(0)
    # Definitions

    #filename  = "TopRecoTree_QCD/QCD_HT200to300_ext1/res/histograms-QCD_HT200to300_ext1.root"
    filename  = "histograms-TT_19var.root"
    debug   = 1
    nprint  = 100
    tfile    = ROOT.TFile.Open(filename)    
    
    #Signal and background branches
    signal     = uproot.open(filename)["treeS"]
    background = uproot.open(filename)["treeB"]
    #signal     = tfile.Get("treeS")
    #background = tfile.Get("treeB")
    
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
    
    '''
    ###########################################################
    # Activate interesting branches (Speed things up !)
    ###########################################################
    print "Activate interesting branches (Speed things up !)"
    signal.SetBranchStatus("*",0) 
    background.SetBranchStatus("*",0) 
    for inp in inputList:        
        signal.SetBranchStatus(inp,1)
        background.SetBranchStatus(inp,1)
    signal.SetBranchStatus("TrijetMass", 1)
    background.SetBranchStatus("TrijetMass", 1)
    ###########################################################
    '''
    #Signal and background dataframes
    df_signal     = signal.pandas.df(inputList)#.assign(signal=1)
    df_background = background.pandas.df(inputList)#.assign(signal=0)

    nEvts = len(df_background.index)/10
    #nEvts = len(df_signal.index)
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
    #X            = dataset[:nEvts, 0:nInputs]
    #target       = dataset_target_all[:nEvts, :] 
    X = dset_background[:nEvts, 0:nInputs]
    target = dataset_target_bkg[:nEvts, :]
    #X = dset_signal[:nEvts, 0:nInputs]
    #target = dataset_target_signal[:nEvts, :]
    #Load model

    '''
    Models: modelANN_lamX.h5
    '''

    lamValues = [0, 1, 5, 10, 50]
    colors = [ROOT.kBlack, ROOT.kOrange, ROOT.kBlue, ROOT.kRed, ROOT.kGreen]
    canvas = ROOT.TCanvas("canvas", "canvas",0,0,800,800)
    canvas.cd()
    #canvas.SetLogy()
    
    icol=-1
    ymax = 0
    histoList = []
    for lam in lamValues:
        print "Lambda = ", lam
        loaded_model = load_model('modelANN_lam%s.h5' % (lam))
        loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
        Y = loaded_model.predict(X, verbose=1)
        Ymass = numpy.concatenate((Y, target), axis=1)
        Ymass_sel = Ymass[Ymass[:,0] >= 0.5]
        massSel = Ymass_sel[:, 1]        

        #Plot resutls
        histo = ROOT.TH1F('histo_lam%s' %(lam), '', 500, 0, 1000)
        mass_sel = []
        print "selected entries:", len(massSel)
        for mass in massSel:
            #print "test: %s / %s: m = %s" % (i, len(massSel), mass)
            histo.Fill(mass)

        icol+=1
        histo.SetLineColor(colors[icol])
        histoList.append(histo.Clone("histoClone_lam%s" % lam))
        
        #histo.Delete()
        del loaded_model
    
    leg=plot.CreateLegend(0.6, 0.67, 0.9, 0.85)
    
    for i in range(len(histoList)):
        leg.AddEntry(histoList[i], "t#bar{t} (#lambda = %s)" %(lamValues[i]) ,"f")
        #histoList[i].Scale(1./(histoList[i].Integral()))
        ymax = max(ymax, histoList[i].GetMaximum())
        histoList[i].Draw("HIST same")
        histoList[i].GetXaxis().SetTitle("m_{top}")
        histoList[i].GetYaxis().SetTitle("Entries")

    leg.Draw("same")
    #tex1 = plot.TFtext("t#bar{t}",0.63,0.5)
    tex2 = plot.TFtext(" unmatched top candidates",0.85,0.5)
    tex3 = plot.TFtext(" with output value > 0.50",0.82,0.45)
    #tex1.SetTextSize(0.045)
    tex2.SetTextSize(0.030)
    tex3.SetTextSize(0.030)
    #tex1.Draw()
    tex2.Draw()
    tex3.Draw()
    graph = plot.CreateGraph([173., 173.], [0, ymax*1.1])
    graph.Draw("same")
    dirName = getDirName()
    if not os.path.exists(dirName):
        os.mkdir(dirName)
        print "Directory " , dirName ,  " Created "
    else:
        print "Output saved under", dirName
        
    SavePlot(canvas, dirName, "TopMassANN_nEvts.pdf")
    #canvas.SaveAs("TopMass.pdf")

    
    
main()
