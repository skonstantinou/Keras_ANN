'''
DESCRIPTION:
Tools for plotting Keras output in the form of JSON files

'''

#================================================================================================ 
# Import modules
#================================================================================================ 
import sys
import os
import math
import json
import array

import ROOT
ROOT.gROOT.SetBatch(True)

import HiggsAnalysis.NtupleAnalysis.tools.statisticalFunctions as statisticalFunctions
import HiggsAnalysis.NtupleAnalysis.tools.plots as plots
import HiggsAnalysis.NtupleAnalysis.tools.histograms as histograms
import HiggsAnalysis.NtupleAnalysis.tools.styles as styles
import HiggsAnalysis.NtupleAnalysis.tools.ShellStyles as ShellStyles
import HiggsAnalysis.NtupleAnalysis.tools.aux as aux


#================================================================================================ 
# Shell Type
#================================================================================================ 
sh_e = ShellStyles.ErrorStyle()
sh_s = ShellStyles.SuccessStyle()
sh_h = ShellStyles.HighlightStyle()
sh_a = ShellStyles.HighlightAltStyle()
sh_l = ShellStyles.AltStyle()
sh_t = ShellStyles.NoteStyle()
sh_n = ShellStyles.NormalStyle()
sh_w = ShellStyles.WarningStyle()

#================================================================================================ 
# Function definitions
#================================================================================================ 
def cleanGraph(graph, massPoint):
    '''
    Remove mass points lower than 100
    
    \param graph   TGraph to operate
    \param minX    Minimum value of mass hypotheses to keep
    
    Remove mass points lower than 100 since
    statisticalFunctions.tanbForBR cannot handle them (they are unphysical)
    also remove points lower than 115 since excluded by LEP    
    '''
    i=0
    while (i<graph.GetN()):
        if (graph.GetX()[i] > massPoint-0.01 and graph.GetX()[i] < massPoint+0.01):
            graph.RemovePoint(i)
        else:
            i=i+1
    return

def divideGraph(num, denom):
    '''
    Divide two TGraphs
    
     \param num    Numerator TGraph
     \param denom  Denominator TGraph
     
     \return new TGraph as the ratio of the two TGraphs
     '''
    gr = ROOT.TGraph(num)
    for i in xrange(gr.GetN()):
        y = denom.GetY()[i]
        val = 0
        if y != 0:
            val = gr.GetY()[i]/y
        gr.SetPoint(i, gr.GetX()[i], val)
    return gr

def subtractGraph(minuend, subtrahend):
    '''
    Subtract two TGraphs
    
    \param minuend     Minuend TGraph
    \param subtrahend  Subtrahend TGraph
    
    \return new TGraph as the difference of the two TGraphs
    '''
    gr = ROOT.TGraph(minuend)
    for i in xrange(gr.GetN()):
        val = gr.GetY() - subtrahend.GetY()[i]
        gr.SetPoint(i, gr.GetX()[i], val)
    return gr

#================================================================================================ 
# Class Definition
#================================================================================================ 
class Output:
    def __init__(self, directory=".", analysisType="HToTauNu", excludePoints=[], resultsFile="results.json", cfgFile="config.json", verbose=False):
        '''
        Constructor

        \param directory          Path to the multicrab task directory with the JSON files
         
        \param excludePoints  List of strings for points to exclude
        '''
        self.resultsX    = {} # from resuits.json
        self.resultsY    = {} # from resuits.json
        self.directory   = directory
        self.resultsFile = resultsFile
        self.resultsPath = os.path.join(self.directory, self.resultsFile)
        self.cfgFile     = cfgFile
        self.cfgFilePath = os.path.join(self.directory, self.cfgFile)
        self.verbose = verbose
        self.storeCfgFiles()
        self.storeAnalysisType(analysisType)
        return

    def Print(self, msg, printHeader=False):
        fName = __file__.split("/")[-1]
        if printHeader==True:
            print "=== ", fName
            print "\t", msg
        else:
            print "\t", msg
            return

    def Verbose(self, msg, printHeader=True, verbose=False):
        if not self.verbose:
            return
        self.Print(msg, printHeader)
        return

    def printAttributes(self):
        self.Print("Printing all object attributes:", True)
        for i,o in enumerate(self.__dict__, 0):
            self.Print("self.%s = %s" % (o, getattr(self, o)), i==-1)
        return

    def storeAnalysisType(self, analysisType):
        self.analysisType = analysisType
        myAnalyses = ["HToTauNu", "HToTB", "HToHW"]
        if analysisType in myAnalyses:
            self.analysisType = analysisType
        else:
            msg = "Invalid analysis type \"%s\". Please selected one of the following: \"%s" % (self.analysisType, "\", \"".join(myAnalyses) + "\"")
            raise Exception(sh_e + msg + sh_n)
        return

    def storeCfgFiles(self):
        
        # Open & load configuration json file
        msg = "Opening file %s" % (self.cfgFilePath)
        self.Verbose(msg, True)
        f = open(self.cfgFilePath, "r")
        config = json.load(f)
        f.close()

        # Create list of variable to retrieve    
        vList = ["timestamp", "layers", "optimizer", "loss function", "epochs", "neurons", "hidden layers", "batch size",  "model", "activation functions"]
        iErr  = -1
        for v in vList:
            if v in config:
                if hasattr(self, v):
                    raise Exception("Output instance already has attribute %s" % (v))
                else:
                    aName = v.title().replace(" ", "")
                    aName = aName[0].lower() + aName[1:]
                    setattr(self, "%s" % aName, config[v])
            else:
                iErr+=1
                msg = "Cannot find variable %s in file %s." % (sh_h + v + sh_n, sh_a + self.cfgFile + sh_n)
                self.Print(msg, iErr == 0)

        # Print all object attributes?
        if self.verbose:
            self.printAttributes()

        # Open results file
        msg = "Opening file '%s'" % (self.resultsFile)
        self.Verbose(msg, True)
        f = open(self.resultsPath, "r")
        results = json.load(f)
        f.close()

        # For-loop: All keys in json file
        self.Verbose("Reading results from %s:" % (sh_h + self.resultsFile + sh_n), True)
        for i, k in enumerate(results.keys(), 0):
            if hasattr(self, k):
                msg = "Output instance already has attribute %s. Skipping" % (sh_a + k + sh_n)
                self.Verbose(msg, True)
                #raise Exception(sh_e + msg + sh_n)
            else:
                attrName  = k
                attrValue = results[k]
                attrType  = "%s" % type(attrValue)
                #if "unicode" in attrType:
                #    self.Print("Converting \"unicode\" to list (name = %s, value = %s)" % (sh_h + attrName + sh_n, sh_a + attrValue + sh_n), True)
                #    attrValue = [x.encode('UTF8') for x in attrValue]
                #else:
                #    self.Print("AttrName = %s, AttrType = %s" % (sh_h + attrName + sh_n, sh_l + str(attrValue) + sh_n), True)

                setattr(self, "%s" % attrName, attrValue)
                # key = "%s-%s" % (os.path.basename(self.directory), k) #iro
                key = k
                self.resultsX[key] = [d["x"] for d in getattr(self, k)]
                self.resultsY[key] = [d["y"] for d in getattr(self, k)]
                self.Verbose("%s = %s" % (key, results[k]), False)

        if self.verbose:
            for i, k in enumerate(self.resultsX.keys(), 0):
                for j in range(0, len(self.resultsX[k])):
                    self.Print("%s: x = %s, y = %s" % (k, self.resultsX[k][j], self.resultsY[k][j]), j==0)
            self.printAttributes()
        return

    def getYMax(self):
        yMax = -1

        # For-loop: All mass points
        self.Print("FIXME Please", True)
        return 1e6
        # for y in self.expectedPlus2:
        #     if y > yMax:
        #         yMax = y
        # return yMax

    def getYMin(self):
        yMin = 1e6

        # For-loop: All mass points
        self.Print("FIXME Please", True)
        return 1e-3
#        for y in self.expectedMinus2:
#            if y < yMin:
#                yMin = y
#        return yMin

    def getYMinMedian(self):
        yMin = 1e6

        # For-loop: All mass points
        for y in self.expectedMedian:
            if y < yMin:
                yMin = y
        return yMin

    def getResultsTable(self, unblindedStatus=False, nDigits=5):
        '''
        Returns a table (list) with the results
        '''
        width  = nDigits + 6
        align  = "{:<8} {:>%s} {:>%s} {:>%s} {:>%s} {:>%s} {:>%s}" % (width, width, width, width, width, width)
        header = align.format("Mass", "Observed", "Median", "-2sigma", "-1sigma", "+1sigma", "+2sigma")
        hLine  = "="*len(header)

        # Define precision
        precision = "%%.%df" % nDigits

        # Create the results table
        table  = []
        table.append(hLine)
        table.append(header)
        table.append(hLine)
        self.Print("PLEASE FIXME", True)
        return
        for i in xrange(len(self.mass_string)):
            mass = self.mass_string[i]
            if unblindedStatus:
                #observed = self.observed_string[i]
                observed = precision % (self.observed_string[i])
            else:
                observed = "BLINDED"
            median       = precision % (self.expectedMedian_string[i])
            sigma2minus  = precision % (self.expectedMinus2_string[i])
            sigma1minus  = precision % (self.expectedMinus1_string[i])
            sigma1plus   = precision % (self.expectedPlus1_string[i])
            sigma2plus   = precision % (self.expectedPlus2_string[i])

            # Append results
            row = align.format(mass, observed, median, sigma2minus, sigma1minus, sigma1plus, sigma2plus)
            table.append(row)
        table.append(hLine)
        return table


    def printResults(self, unblindedStatus=False, nDigits=5):
        '''
        Print the results
        '''
        table = self.getLimitsTable(unblindedStatus, nDigits)
        # Print limits (row by row)
        for i, row in enumerate(table,1):
            Print(row, i==1)
        return

    def saveAsLatexTable(self, unblindedStatus=False, nDigits=3, savePath=None, HToTB=False):
        '''
        Save the table as tex format
        '''        
        myDigits = nDigits

        # Define precision of results
        precision = "%%.%df" % myDigits
        format    = "%3s "

        # Five columns (+/-2sigma, +/-1sigma, median)
        for i in range(0,5):
	    format += "& %s "%precision 
	
        # Blinded column
        if not unblindedStatus:
	    format += "& Blinded "
	else:
	    format += "& %s " % precision 

        # End-line character (\\)
        format += "\\\\ \n"

        # Add the LaTeX table contents
        s  = "% Table autocreated by HiggsAnalysis.LimitCalc.limit.saveAsLatexTable() \n"
        s += "\\begin{tabular}{ c c c c c c c } \n"
        s += "\\hline \n"
        if HToTB:
	    s += "\\multicolumn{7}{ c }{95\\% CL upper limit on $\\BRtH\\times\\BRHtb$}\\\\ \n"
        else:
            s += "\\multicolumn{7}{ c }{95\\% CL upper limit on $\\sigmaHplus\\times\\BRHtau$}\\\\ \n"

	s += "\\hline \n"
	s += "\\mHpm & \\multicolumn{5}{ c }{Expected limit} & Observed \\\\ \\cline{2-6} \n"
	s += "(GeV)   & $-2\\sigma$  & $-1\\sigma$ & median & +1$\\sigma$ & +2$\\sigma$  & limit \\\\ \n"
	s += "\\hline \n"

        # Get the limit values
        for i in xrange(len(self.mass_string)):
            mass     = self.mass_string[i]
            eMinus2  = float( precision % (self.expectedMinus2_string[i]) )
            eMinus1  = float( precision % (self.expectedMinus1_string[i]) )
            eMedian  = float( precision % (self.expectedMedian_string[i]) )
            ePlus1   = float( precision % (self.expectedPlus1_string[i]) )
            ePlus2   = float( precision % (self.expectedPlus2_string[i]) )
            observed = float( self.observed_string[i]) 
            if unblindedStatus:
                s += format % (mass, eMinus2, eMinus1, eMedian, ePlus1, ePlus2, observed)
            else:
                s += format % (mass, eMinus2, eMinus1, eMedian, ePlus1, ePlus2)
	s += "\\hline \n"
        s += "\\end{tabular} \n"

        fileName = "resultsTable.tex"
        openMode = "w"  
        Verbose("Opening file '%s' in mode '%s'" % (fileName, openMode), True)
        if savePath == None:
            f = open(fileName, openMode)
        else:
            if not os.path.isdir(savePath):
                raise Exception("Cannot save LaTeX table! The path provided (%s) is not a directory!" % (savePath))
            f = open(os.path.join(savePath, fileName), openMode)
        f.write(s)
        f.close()
        Print("Wrote LaTeX table in file '%s'" % (fileName), True)
        return

    def getLegendLabel(self):
        label = "%sL " % (self.layers)
        #label+= "%s Opt " % (self.optimizer)
        label+= " %sb" % (self.batchSize)
        label+= " %se" % (self.epochs)
        # Dirty trick to convert to list the activationFunctions variable (unicode)
        self.activationList = self.activationFunctions.encode('UTF8').replace("[", "").replace("]", "").replace("'", "").split(",")
        self.neuronsList    = self.neurons.encode('UTF8').replace("[", "").replace("]", "").replace("'", "").replace(" ", "").split(",")

        # For-loop: All actication functions
        for i, a in enumerate(self.activationList, 0):
            self.Verbose("%s, type(%s) = %s" % (a, a, type(a)), True)
            label+= " %s%s" % (self.activationList[i].replace("sigmoid", "sig"), self.neuronsList[i])
        # self.hiddenLayers
        # self.lossFunction
        # self.model
        return label

    def _getGraphs(self, keyword=None):

        graphList = []
        legList   = []
        # For-loop: All x-results (should be same size as y-results)
        for i, k in enumerate(self.resultsX.keys(), 0):
            #self.Print("Setting graph name to %s" % (sh_a + k + sh_n), i==0)
            if keyword!=None:
                if keyword not in k:
                    continue 
            xArray = array.array("d", self.resultsX[k])
            yArray = array.array("d", self.resultsY[k])
            graph  = ROOT.TGraph(len(xArray), xArray, yArray)
            graph.SetName(k) # + "_%d" % (i))
            styles.applyStyle(graph, i)
            # setExpectedStyle(gr) #fixme
            graphList.append(graph)
            legList.append(self.getLegendLabel())
        return graphList, legList

    def getGraphs(self, keyword=None):
        return self._getGraphs(keyword)

    def expectedErrorGraph(self, sigma=0):
        '''
        Construct TGraph for the expected limit toy MC stat error
        
        \param sigma   Integer for the sigma band (0 for median, 1,-1, 2,-2)
        
        \return TGraph of the expexted limit stat error
        '''
        if not hasattr(self, "expectedMedianError"):
            return None
        return self._getGraph("Error", sigma)


    def expectedBandGraph(self, sigma):
        '''
        Construct TGraph for the expected +-1/2 sigma bands
        
        \param sigma   Integer for the sigma bands (1, 2)
        
        \return TGraph for the expected sigma bands
        
        The TGraph holds the sigma bands as the values. The values go
        first through the lower band in the increasing mass order, then
        the upper band in the decreasing mass order
        '''
        xArray = array.array("d", self.mass)
        massErr = array.array("d", [0]*len(self.mass))
        if sigma == 1:
            tmp1 = self.mass[:]
            tmp1.reverse()
            tmp2 = self.expectedPlus1[:]
            tmp2.reverse()

            gr = ROOT.TGraph(2*len(self.mass),
                             array.array("d", self.mass+tmp1),
                             array.array("d", self.expectedMinus1 + tmp2))

            setExpectedGreenBandStyle(gr)
            
            gr.SetName("Expected1Sigma")
        elif sigma == 2:
            tmp1 = self.mass[:]
            tmp1.reverse()
            tmp2 = self.expectedPlus2[:]
            tmp2.reverse()

#            print self.mass+tmp1
#            print self.expectedMinus2+tmp2

            gr = ROOT.TGraph(2*len(self.mass),
                             array.array("d", self.mass+tmp1),
                             array.array("d", self.expectedMinus2 + tmp2))

            setExpectedYellowBandStyle(gr)
            gr.SetName("Expected2Sigma")
        else:
            raise Exception("Invalid value of sigma '%d', valid values are 0,1,2" % sigma)
        return gr
