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
from optparse import OptionParser
#from rootpy.root2array import fill_hist_with_ndarray

import plot
import tdrstyle
import func

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #Disable AVX/FMA Warning
# Do not display canvases
ROOT.gROOT.SetBatch(ROOT.kTRUE)
# Disable screen output info
ROOT.gROOT.ProcessLine( "gErrorIgnoreLevel = 1001;")

def main():
    ROOT.gStyle.SetOptStat(0)

    # Apply tdr style
    style = tdrstyle.TDRStyle() 
    style.setOptStat(False) 
    style.setGridX(False)
    style.setGridY(False)

    # Definitions
    filename  = opts.filename
    tfile     = ROOT.TFile.Open(filename)    
    
    dName = ""
    if ("TT" in filename):
        dName = "TT"
    elif ("QCD" in filename):
        dName = "QCD"

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

    # Number of events to predict the output and plot the top-quark mass
    if (opts.plotSignal):
        nEvts = len(df_signal.index)
    else:
        nEvts = len(df_background.index)

    # User few events for testing
    if (opts.test):
        nEvts = 1000

    print "=== Number of events: ", nEvts    
    #Signal and background datasets
    dset_signal     = df_signal.values
    dset_background = df_background.values
    
    # Concat signal, background datasets
    df_list = [df_signal, df_background]
    df_all = pandas.concat(df_list)
    dataset = df_all.values
    
    # Target (top-quark mass) datasets
    dset_target_all    = pandas.concat([signal.pandas.df(["TrijetMass"]), background.pandas.df(["TrijetMass"])]).values
    dset_target_bkg    = background.pandas.df(["TrijetMass"]).values
    dset_target_signal = signal.pandas.df(["TrijetMass"]).values

    # Signal and background inputs
    X_signal     = dset_signal[:nEvts, 0:nInputs]
    X_background = dset_background[:nEvts, 0:nInputs]

    if (opts.plotSignal):
        X = dset_signal[:nEvts, 0:nInputs]
        target = dset_target_signal[:nEvts, :]
    else:
        X = dset_background[:nEvts, 0:nInputs]
        target = dset_target_bkg[:nEvts, :]

    #Load models
    lamValues = [0, 1, 5, 10, 20]#, 100, 500] # fixme! should be given as option
    colors = [ROOT.kBlack, ROOT.kBlue, ROOT.kMagenta, ROOT.kOrange, ROOT.kRed, ROOT.kGreen, ROOT.kOrange+7]
    canvas = plot.CreateCanvas()
    canvas.cd()
    
    ymaxFactor = 1.1
    ymax = 0    
    histoList = []

    if (opts.setLogY):
        canvas.SetLogy()
        ymaxFactor = 2
        

    # load the models with different lambda
    for lam in lamValues:
        print "Lambda = ", lam
        if (lam == 0):
            # For labda = 0 load the simple classification (sequential) model
            loaded_model = load_model('Model_relu_relu_relu_sigmoid.h5')
        else:
            loaded_model = load_model('modelANN_32_Adam_0p0008_500_tanh_relu_msle_lam%s.h5' % (lam))

        # Compile the model
        loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
        Y = loaded_model.predict(X, verbose=1)
        
        # Concatenate Y (predicted output) and target (top-quark mass)
        # Ymass 0 column: output
        # Ymass 1st column: top-quark mass
        Ymass = numpy.concatenate((Y, target), axis=1)
        
        # Get selected top candidates (pass the output working point)
        Ymass_sel = Ymass[Ymass[:,0] >= opts.wp] # Select samples with y > WP (col 0)
        massSel   = Ymass_sel[:, 1]              # Get the top-quark mass (col 1) for the selected samples.

        #Plot resutls
        nbins = 100
        xmin  = 0
        xmax  = 1000
        
        # Change x-axis range when plotting the signal
        if (opts.plotSignal):
            nbins = 45
            xmax  = 450

        width = float(xmax)/nbins
        histo = ROOT.TH1F('histo_lam%s' %(lam), '', nbins, xmin, xmax)
        mass_sel = []
        print "selected entries:", len(massSel)
        
        for mass in massSel:
            histo.Fill(mass)

        histoList.append(histo.Clone("histoClone_lam%s" % lam))
        
        del loaded_model
    
    # Create legend
    leg=plot.CreateLegend(0.6, 0.67, 0.9, 0.85)
    dText = dName
    dText = dText.replace("TT", "t#bar{t}") #Legend text

    # Loop over the histograms in the histoList
    for i in range(len(histoList)):        
        leg.AddEntry(histoList[i], "%s (#lambda = %.0f)" %(dText, lamValues[i]) ,"f")
        # Normalize histograms to unity
        histoList[i].Scale(1./(histoList[i].Integral()))
        ymax = max(ymax, histoList[i].GetMaximum())
        histoList[i].GetXaxis().SetTitle("m_{top} (GeV)")
        histoList[i].GetYaxis().SetTitle("Arbitrary Units / %.0f GeV" % (width))
        plot.ApplyStyle(histoList[i], colors[i])
        histoList[i].Draw("HIST same")

    for i in range(len(histoList)):
        histoList[i].SetMaximum(ymax*ymaxFactor)
    leg.Draw("same")
    
    if (opts.plotSignal):
        sbText = "truth-matched"
    else:
        sbText = "unmatched"

    # Additional text
    tex1 = plot.Text(" %s top candidates" % sbText,0.85,0.5)
    tex2 = plot.Text(" with output value > %s" % (opts.wp),0.82,0.45)
    
    #Draw text
    tex1.Draw()
    tex2.Draw()
    
    # Draw line to indicate the real value of the top-quark mass
    graph = plot.CreateGraph([173., 173.], [0, ymax*ymaxFactor])
    graph.Draw("same")

    # CMS extra text and lumi text
    #plot.CMSText("CMS Preliminary") #Fixme! cmsExtra doesn't work
    #cmsText.Draw()

    # Output directory
    dirName = plot.getDirName(opts.saveDir)
        
    # Save the plot
    saveName = opts.saveName
    if (opts.saveName == "TopMassANN"):
        saveName = "TopMassANN_%s_%s" % (dName, sbText.replace("-","_"))

    # Save the histogram
    plot.SavePlot(canvas, dirName, saveName)



#================================================================================================
# Main
#================================================================================================
if __name__ == "__main__":
    '''
    https://docs.python.org/3/library/argparse. html

    name or flags...: Either a name or a list of option strings, e.g. foo or -f, --foo.
action..........: The basic type of action to be taken when this argument is encountered at the command line.
    nargs...........: The number of command-line arguments that should be consumed.
    const...........: A constant value required by some action and nargs selections.
    default.........: The value produced if the argument is absent from the command line.
    type............: The type to which the command-line argument should be converted.
    choices.........: A container of the allowable values for the argument.
    required........: Whether or not the command-line option may be omitted (optionals only).
    help............: A brief description of what the argument does.
    metavar.........: A name for the argument in usage messages.
    dest............: The name of the attribute to be added to the object returned by parse_args().
    '''
    
    # Default Settings
    FILENAME   = "histograms-TT_19var.root"
    PLOTSIGNAL = False
    WP         = 0.5
    SAVEDIR    = "TopTag"
    SAVENAME   = "TopMassANN" #"TopMassANN"
    SETLOGY    = False
    TEST       = False

   # Define the available script options
    parser = OptionParser(usage="Usage: %prog [options]")

    parser.add_option("--filename", dest="filename", type="string", default=FILENAME, 
                      help="Input ROOT file containing the signal and backbground TTrees with the various TBranches *variables) [default: %s]" % FILENAME)
    parser.add_option("--plotSignal", dest="plotSignal", action="store_true", default=PLOTSIGNAL, 
                      help="Plot the top-quark mass for real top-candidates (signal) [default: %s]" % PLOTSIGNAL)
    parser.add_option("--wp", dest="wp", type=float, default=WP,
                      help="Neural Network output working point [default: %s]" % WP)
    parser.add_option("--saveDir", dest="saveDir", type="string", default=SAVEDIR,
                      help="Output name directory [default: %s]" % SAVEDIR)
    parser.add_option("--saveName", dest="saveName", type="string", default=SAVENAME,
                      help="Name of the output histogram")
    parser.add_option("--setLogY", dest="setLogY", action="store_true", default=SETLOGY,
                      help="Set logarithmic y-axis [default: %s]" % SETLOGY)
    parser.add_option("--test", dest="test", action="store_true", default=TEST,
                      help="Test mode (Use few events) [default: %s]" % TEST)
    (opts, parseArgs) = parser.parse_args()

main()
