#!/usr/bin/env python
'''
DESCRIPTION:
This script is used to develop a mass-decorrelated top-quark tagger using a Neural Network architecture. 
The model identifies hadronically decaying top-quarks using as input a list of discriminating variables of 
truth-matched (signal) or unmatched (background) trijet combinations. 
For the mass decorrelated tagger we develop an Adversarial Neural Network that consists of two models:
 1. Classifer: The nominal model that takes as input the set of input variables and gives as output 
               a value from 0 to 1 that characterizes an object as signal or background. 
               Loss function: L_c
 2. Regressor: This model is introduced in order to prevent the classifier from learning the top-quark mass.
               It takes as input the output of the classifier and its output is a prediction of the top-quark mass.
               Loss function: L_r
Combining the two models, we get a new Neural Network that minimizes a joint Loss functions: L = L_c - lambda*L_r
The factor lambda controls the mass independence and the performance of the classifier. 
During a training epoch, both classifier and regressor are trained and after each iteration, the models use 
the updated weights.
During the training, we expect an increase of the loss value of the classifier and the regressor, and at the same
time the minimization of the joint loss function.

LAST USED: 
./trainTopTagger_decorrelated.py --neurons_clf 32,32,32,1 --neurons_reg 16,1 --activ_clf relu,relu,relu,sigmoid --activ_reg tanh,relu --lam 5
'''

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
from sklearn.model_selection import train_test_split
import keras.backend as K
from keras.optimizers import SGD, Adam

import plot
import func
import tdrstyle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #Disable AVX/FMA Warning

#Run in single CPU: this ensures reproducible results!
config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1,
                        allow_soft_placement=True, device_count = {'CPU': 1})
session = tf.Session(config=config)
K.set_session(session)

###########################################################
# Print
###########################################################

def Print(msg, printHeader=False):
    fName = __file__.split("/")[-1]
    if printHeader==True:
        print "=== ", fName
        print "\t", msg
    else:
        print "\t", msg
    return


###########################################################
# Plot Loss Function
###########################################################

def PlotLossFunction(losses, nepochs, saveDir):
    '''
    Plot the loss function vs the number of the epoch
    '''
    def ApplyStyle(h, ymin, ymax, color):
        h.SetLineColor(color)
        h.SetMarkerColor(color)
        h.SetMarkerStyle(8)
        h.SetMarkerSize(0.6)
        h.SetMaximum(ymax)
        h.SetMinimum(ymin)
        h.GetXaxis().SetTitle("# epoch")
        h.GetYaxis().SetTitle("Loss")

    ROOT.gStyle.SetOptStat(0)
    canvas = ROOT.TCanvas()
    canvas.cd()
    leg=plot.CreateLegend(0.6, 0.75, 0.9, 0.85)

    xarr = []
    for i in range(1, nepochs+1):
        xarr.append(i)


    if (opt.lam > 0):
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
        
        leg.AddEntry(hLf, "L_{c}","pl")
        leg.AddEntry(hLr,"L_{r}","pl")
        leg.AddEntry(hLfr,"L_{c} - L_{r}","pl")

    else:
        h_loss = ROOT.TGraph(len(xarr), array('d', xarr), array('d', losses["loss"]))
        ymax = h_loss.GetHistogram().GetMaximum()
        ymin = h_loss.GetHistogram().GetMinimum()
        ApplyStyle(h_loss, ymin, ymax,  ROOT.kGreen +1)
        h_loss.Draw("pa")
        leg.AddEntry(h_loss, "L_{c}","pl")

    leg.Draw()
    saveName = "Loss_lam%s_opt%s" %(opt.lam, opt.opt)
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

###########################################################
# Get Classifier
###########################################################

def getClassifier(model_clf, nInputs):        
    # Add classifier layers
    for iLayer, n in enumerate(opt.neurons_clf, 0):
        # First layer demands input_dim
        if (iLayer == 0):
            model_clf.add(Dense(opt.neurons_clf[iLayer], input_dim =  nInputs)) 
            model_clf.add(Activation(opt.activ_clf[iLayer])) 
        else:
            model_clf.add(Dense(opt.neurons_clf[iLayer]))
            model_clf.add(Activation(opt.activ_clf[iLayer]))

    return model_clf

###########################################################
# Get Regressor
###########################################################

def getRegressor(model_clf, model_reg):    
    # Add classifier
    model_reg.add(model_clf)

    # Add regressor layers
    for iLayer, n in enumerate(opt.neurons_reg, 0):
        model_reg.add(Dense(opt.neurons_reg[iLayer]))
        model_reg.add(Activation(opt.activ_reg[iLayer]))

    return model_reg

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
        if opt.loss_reg == "msle":
            loss =  c * keras.losses.mean_squared_logarithmic_error(z_true, z_pred)
        elif opt.loss_reg == "mse":
            loss =  c * keras.losses.mean_squared_error(z_true, z_pred)
        return loss            
    return loss_R

def main():
    
    # Do not display canvases
    ROOT.gROOT.SetBatch(ROOT.kTRUE)
    # Disable screen output info
    ROOT.gROOT.ProcessLine( "gErrorIgnoreLevel = 1001;")

    # Apply tdr style
    style = tdrstyle.TDRStyle() 
    style.setOptStat(False) 
    style.setGridX(opt.gridX)
    style.setGridY(opt.gridY)

    # Definitions
    filename   = opt.filename
    tfile      = ROOT.TFile.Open(filename)
    seed       = 1234
    nepochs    = opt.nepochs
    lam        = opt.lam
    batch_size = opt.batch_size
    
    # Extra text: 
    _lr        = str(opt.lr)
    extra_text = "batch%s_Opt%s_LR%s_Epoch%s_NLayerClf%s_NLayerReg%s" % (opt.batch_size, opt.opt, _lr.replace(".","p"), opt.nepochs, len(opt.activ_clf), len(opt.activ_reg))
    saveDir    = plot.getDirName("TopTag_%s" % extra_text)

    if (opt.debug):
        print "extra_text: %s" % extra_text

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
    
    # Number of signal events
    nsignal = len(df_signal.index)
    print "=== Number of signal events: ", nsignal
    
    #Get signal and background datasests
    dset_signal     = df_signal.values
    dset_background = df_background.values

    # Get inputs for signal and background 
    X_signal     = dset_signal[:nsignal, 0:nInputs]
    X_background = dset_background[:nsignal, 0:nInputs]
    
    # Concat signal, background datasets
    dataset            = pandas.concat([df_signal, df_background]).values
    dataset_target_all = pandas.concat([signal.pandas.df(["TrijetMass"]), background.pandas.df(["TrijetMass"])]).values

    # Define Callbacks. Early stop: Stop after 100 iterations if the val loss function converges (no change > min_delta)
    earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=100)
    callbacks_list = [earlystop]

    #Split data set into input (X) and output (Y)
    X = dataset[:2*nsignal,0:nInputs] # inputs: columns 0-19
    Y = dataset[:2*nsignal,nInputs:]  # output: column 19

    # Get target variable (top-quark mass)
    target = dataset_target_all[:2*nsignal, :]

    # Is this needed?
    #numpy.random.seed(seed)

    # Plot input variables
    if (opt.plotInputs):
        for i, var in enumerate(inputList, 0):
            func.PlotInputs(dset_signal[:, i:i+1], dset_background[:, i:i+1], var, "%s/%s" % (saveDir, "inputs"), opt.saveFormats)

    # Get training and test samples
    X_train, X_test, Y_train, Y_test, target_train, target_test = train_test_split(
        X, Y, target, test_size=0.5, random_state=seed, shuffle=True)

    ###################################################
    # Define classifier
    ###################################################
    
    model_clf = Sequential()
    model_clf = getClassifier(model_clf, nInputs)

    # Compile classifier (loss function: binary crossentropy, optimizer: adam)
    model_clf.compile(loss="binary_crossentropy", optimizer="adam")

    print "=== Classifier: Fit"
    model_clf.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, verbose=opt.debug)

    ###################################################
    # Define Models
    ###################################################
    '''
    Classifier (characterizes events as signal or background)
    Inputs: List of input variables
    Output: classifier
    Regressor: Predicts the top-quark mass from the classifiers output
    '''
    model_clf = Sequential()
    model_clf = getClassifier(model_clf, nInputs)
    model_reg = Sequential()
    model_reg = getRegressor(model_clf, model_reg)

    # Print the summary of classifier, regressor
    if opt.debug:
        model_clf.summary()
        model_reg.summary()

    # Compile classifier (loss function: binary crossentropy, optimizer: adam)
    model_clf.compile(loss="binary_crossentropy", optimizer="adam")
    
    ###################################################
    # Combined models
    ###################################################
    inputs   = Input(shape=(X_train.shape[1],))

    # Optimizer
    opt_comb = opt.opt
    opt_adv  = opt.opt

    if opt.opt == "SGD":    
        opt_comb = SGD(momentum=0.0, lr = opt.lr)
        opt_adv = SGD(momentum=0.0, lr = opt.lr)
    elif opt.opt == "Adam":
        opt_comb = Adam(lr = opt.lr)
        opt_adv = Adam(lr = opt.lr)

    # Combined network: updates the classifier's weights
    # Input: list of discriminating variables
    # Output: Classifier's output, top-qark mass prediction
    model_comb = Model(inputs=[inputs], outputs=[model_clf(inputs), model_reg(inputs)])
    
    # Regression layers are not trainable !
    make_trainable(model_reg, False)
    make_trainable(model_clf, True)

    # Compile the combined model
    # L_comb = L_clf - lambda*L_reg
    model_comb.compile(loss=[getLoss_clf(c=1.0), getLoss_reg(c=-lam)], optimizer=opt_comb)

    # Adversary network: predicts top-quark mass 
    model_adv = Model(inputs=[inputs], outputs=[model_reg(inputs)])

    # Classification layers are not trainable !
    make_trainable(model_reg, True)
    make_trainable(model_clf, False)
    
    # Compile the model
    # L_adv = L_reg
    model_adv.compile(loss=[getLoss_reg(c=1.0)], optimizer=opt_adv)

    ###################################################
    # Pretrain the Classifier
    ###################################################
    make_trainable(model_reg, False)
    make_trainable(model_clf, True)
    
    print "=== Classifier: Fit"
    model_clf.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, verbose = opt.debug)
    ###################################################
    # Pretrain  the Regressor
    # Here the output is the top-quark mass (target)
    ###################################################
    make_trainable(model_reg, True)
    make_trainable(model_clf, False)
    
    print "=== Adversarial Network: Fit"
    model_adv.fit(X_train, target_train, validation_data=(X_test, target_test), epochs=10, verbose = opt.debug)
    
    ###################################################
    # Train the combined model
    ###################################################

    # Dictionary with loss functions
    losses = {"L_f": [], "L_r": [], "L_f - L_r": []}    

    if (lam > 0):
        for i in range(nepochs):
            # Evaluate the loss function after each epoch
            l = model_comb.evaluate(X_test, [Y_test, target_test], verbose=opt.debug)
            
            lf = l[1][None][0]
            lr = -l[2][None][0]
            lfr = l[0][None][0]
            
            # Store the value of the loss function after each epoch
            losses["L_f - L_r"].append(lfr)
            losses["L_f"].append(lf)
            losses["L_r"].append(lr)
            
            if (nepochs < 10) or (i % (nepochs/10) == 0):
                print "=== Epoch: %s / %s. Losses: Lf = %.3f, Lr = %.3f, (L_f - L_r) = %.3f" % (i, nepochs, lf, lr, lfr)
           
            # if the loss function does not change after 100 epochs, return
            if (i > 100):
                if ((losses["L_r"][0] == losses["L_r"][50]) and (losses["L_r"][0] ==losses["L_r"][100])):
                    print "failure!"
                    return
                if ((losses["L_f"][0] == losses["L_f"][50]) and (losses["L_f"][0] ==losses["L_f"][100])):
                    print "failure!"
                    return

            # Fit Classifier (with updated weights) to minimize joint loss function
            make_trainable(model_reg, False)
            make_trainable(model_clf, True)
            indices = numpy.random.permutation(len(X_train))[:batch_size]

            # Train and test on a batch
            model_comb.train_on_batch(X_train[indices], [Y_train[indices], target_train[indices]])
            model_comb.test_on_batch(X_test[indices], [Y_test[indices], target_test[indices]])
            
            # Fit Regressor (with updated weights) to minimize Lr
            make_trainable(model_reg, True)
            make_trainable(model_clf, False)
            model_adv.fit(X_train, target_train, validation_data=(X_test, target_test), batch_size=batch_size, epochs=opt.nepochs_adv, callbacks=callbacks_list, verbose=opt.debug)
    else: # Lambda == 0
        make_trainable(model_clf, True)
        hist = model_clf.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=batch_size, epochs=nepochs, callbacks=callbacks_list, verbose=opt.debug)

    ###################################################
    # Test classifier's performance
    ################################################### 
    if opt.debug:
        y_pred = model_clf.predict(X_test)
        for i in range(50):
            print("Predicted=%s (%s) (%s)"% (target_test[i], y_pred[i], Y_test[i]))

    ###################################################
    # Calculate the output
    ###################################################
    XY_train     = numpy.concatenate((X_train, Y_train), axis=1)
    XY_test      = numpy.concatenate((X_test, Y_test), axis=1)
    
    x_train_S = XY_train[XY_train[:,nInputs] == 1]; x_train_S = x_train_S[:,0:nInputs]
    x_train_B = XY_train[XY_train[:,nInputs] == 0]; x_train_B = x_train_B[:,0:nInputs]
    x_test_S  = XY_test[XY_test[:,nInputs] == 1];   x_test_S  = x_test_S[:,0:nInputs]
    x_test_B  = XY_test[XY_test[:,nInputs] == 0];   x_test_B  = x_test_B[:,0:nInputs]
    
    pred_train_S =  model_clf.predict(x_train_S)
    pred_train_B =  model_clf.predict(x_train_B)
    pred_test_S  =  model_clf.predict(x_test_S)
    pred_test_B  =  model_clf.predict(x_test_B)
    
    ###################################################
    # Plot results:
    ###################################################

    # === Loss function vs # epoch
    if (lam > 0):
        last_epoch = len(losses["L_f - L_r"])
        PlotLossFunction(losses, last_epoch, saveDir)
    else:
        last_epoch = earlystop.stopped_epoch
        if (last_epoch == 0):
            # last_epoch = nepoch if the training does not stop before the last epoch
            last_epoch = nepochs
        print "last epoch", last_epoch
        PlotLossFunction(hist.history, last_epoch, saveDir)
        
    # === Overtraining test
    htrain_s, htest_s, htrain_b, htest_b = func.PlotOvertrainingTest(pred_train_S, pred_test_S, pred_train_B, pred_test_B, saveDir, "OvertrainingTest_lam%s" % lam, opt.saveFormats)
    # === Efficiency
    func.PlotEfficiency(htest_s, htest_b, saveDir, "Efficiency_lam%s" % lam, opt.saveFormats)

    ###################################################
    # Save the model
    ###################################################
    
    saveModelDir = "models_%s" % extra_text
    # Create output directory if it does not exist
    plot.CreateDir(saveModelDir)

    # serialize weights to HDF5
    model_clf.save_weights('%s/modelANN_weights_lam%s.h5' % (saveModelDir, opt.lam), overwrite=True)
    model_clf.save("%s/modelANN_lam%s.h5" % (saveModelDir, opt.lam))
    
    # serialize architecture to JSON
    model_json = model_clf.to_json()
    with open("%s/modelANN_architecture_lam%s.json" % (saveModelDir, opt.lam), "w") as json_file:
        json_file.write(model_json)

    # write weights and architecture in txt file
    func.WriteModel(model_clf, model_json, inputList, "%s/modelANN_lam%s.txt" % (saveModelDir, opt.lam))


#================================================================================================ 
# Main
#================================================================================================ 
if __name__ == "__main__":
    '''
    https://docs.python.org/3/library/argparse.html
 
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

    #Default values:
    LAMBDA       = 10
    NEPOCHS      = 1000
    NEPOCHS_ADV  = 1
    OPTIMIZER    = "Adam"
    GRIDX        = False
    GRIDY        = False
    DEBUG        = False
    FILENAME     = "histograms-TT_19var.root"
    BATCH_SIZE   = 128
    LR           = 0.001 #Default for Adam Optomizer (For SGD the default value is 0.01)
    ACTIV_CLF    = 'relu,relu,relu,sigmoid'
    ACTIV_REG    = 'tanh,relu'
    NEURONS_CLF  = '32,32,32,1'
    NEURONS_REG  = '16,1'
    LOSS_REG     = 'msle'
    SAVEFORMATS  = ["pdf"]
    PLOTINPUTS   = False

    from optparse import OptionParser
    parser = OptionParser(usage="Usage: %prog [options]")
    
    parser.add_option("--filename",dest="filename",  default=FILENAME,  type = str,  help="Input root file (Default: %s)" % FILENAME)
    parser.add_option("--lr",dest="lr",  default=LR,  type = float,   help="Learning Rate of the optimizer (Default: %s)" % LR)
    parser.add_option("--batch_size",dest="batch_size",  default=BATCH_SIZE,  type = int,   help="Batch Size (Default: %s)" % BATCH_SIZE)
    parser.add_option("--opt",dest="opt",  default=OPTIMIZER,   help="Optimizer (Default: %s)" % OPTIMIZER)
    parser.add_option("--lam",dest="lam",  default=LAMBDA,  type = int,   help="Lambda (Default: %s)" % LAMBDA)
    parser.add_option("--nepochs", dest="nepochs", default = NEPOCHS, type = int, help ="Number of epochs (Default: %s)" % NEPOCHS)
    parser.add_option("--nepochs_adv", dest="nepochs_adv", default = NEPOCHS_ADV, type = int, help ="Number of epochs for adversarial network (Default: %s)" % NEPOCHS_ADV)
    parser.add_option("--gridX", dest="gridX", action="store_true", default=GRIDX, help="Enable the x-axis grid lines [default: %s]" % GRIDX)
    parser.add_option("--gridY", dest="gridY", action="store_true", default=GRIDY, help="Enable the y-axis grid lines [default: %s]" % GRIDY)
    parser.add_option("--debug", dest="debug", action="store_true", default=DEBUG, help="Enable debugging mode [default: %s]" % DEBUG)
    parser.add_option("--loss_reg", dest="loss_reg",  default=LOSS_REG,   help="Regressor loss function (Default: %s)" % LOSS_REG)
    parser.add_option("--activ_clf",dest="activ_clf", type="string", default=ACTIV_CLF,
                      help="List of activation functions of the classifier (comma-separated) [default: %s]" % ACTIV_CLF)
    parser.add_option("--activ_reg", dest="activ_reg", type="string", default=ACTIV_REG,
                      help="List of activation functions of the regressor (comma-separated) [default: %s]" % ACTIV_REG)    
    parser.add_option("--neurons_clf", dest="neurons_clf", type="string", default=NEURONS_CLF,
                      help="List of neurons to use for each classification layer (comma-separated integers)  [default: %s]" % NEURONS_CLF)
    parser.add_option("--neurons_reg", dest="neurons_reg", type="string", default=NEURONS_REG,
                      help="List of neurons to use for each regression layer (comma-separated integers)  [default: %s]" % NEURONS_REG)
    parser.add_option("-s", "--saveFormats", dest="saveFormats", default=SAVEFORMATS, help="Save formats for all plots [default: %s]" % SAVEFORMATS)
    parser.add_option("--plotInputs", dest="plotInputs", action="store_true", default=PLOTINPUTS, help="Plot the distributions of the input variables [default: %s]" % PLOTINPUTS)

    opt, args = parser.parse_args()
    
    # Define colors
    c_green = "\033[92m"
    c_white = "\033[0;0m"

    if (opt.opt == "SGD"):
        lr_def = 0.01
    elif (opt.opt == "Adam"):
        lr_def = 0.001
    
    if (opt.lr != lr_def):
        Print("%sYou selected to use the %s optimizer with learing rate %s. The default value of the learing rate is %s.%s" % (c_green, opt.opt, opt.lr, lr_def, c_white), True)
        
    # Create specification lists
    print "=== Classifier"
    if "," in opt.activ_clf:
        opt.activ_clf = opt.activ_clf.split(",")
    else:
        opt.activ_clf = [opt.activ_clf]
    Print("Activation = %s" % (opt.activ_clf), True)
    if "," in opt.neurons_clf:
        opt.neurons_clf = list(map(int, opt.neurons_clf.split(",")) )
    else:
        opt.neurons_clf = list(map(int, [opt.neurons_clf]))
    Print("Neurons = %s" % (opt.neurons_clf), True)

    print "=== Regressor"
    if "," in opt.activ_reg:
        opt.activ_reg = opt.activ_reg.split(",")
    else:
        opt.activ_reg = [opt.activ_reg]
    Print("Activation = %s" % (opt.activ_reg), True)
    if "," in opt.neurons_reg:
        opt.neurons_reg = list(map(int, opt.neurons_reg.split(",")) )
    else:
        opt.neurons_reg = list(map(int, [opt.neurons_reg]))
    Print("Neurons = %s" % (opt.neurons_reg), True)

    # Check if the number of layers in equal with the number of activation layers
    if len(opt.neurons_clf) != len(opt.activ_clf):
        msg = "== Classifier: The list of neurons (size=%d) is not the same size as the list of activation functions (=%d)" % (len(opt.neurons_clf), len(opt.activ_clf))
        raise Exception(c_green + msg + c_white)
    if len(opt.neurons_reg) != len(opt.activ_reg):
        msg = "== Regressor: The list of neurons (size=%d) is not the same size as the list of activation functions (=%d)" % (len(opt.neurons_reg), len(opt.activ_reg))
        raise Exception(c_green + msg + c_white)

    main()                                  
                                    
    # Problems:
    # 1. Check why we need the classifier's PRE pre-training
