#================================================================================================
# Import modules
#================================================================================================   
import ROOT
import plot
import math
import array
import json

#================================================================================================   
# Function definition
#================================================================================================   
def Print(msg, printHeader=False):
    fName = __file__.split("/")[-1]
    if printHeader==True:
        print "=== ", fName
        print "\t", msg
    else:
        print "\t", msg
    return
   
def convertHistoToGaph(histo, verbose=False):
                                                                                                                                                                                                            
    # Lists for values
    x     = []
    y     = []
    xerrl = []
    xerrh = []
    yerrl = []
    yerrh = []
    nBinsX = histo.GetNbinsX()
    
    for i in range (0, nBinsX+1):
        # Get values
        xVal  = histo.GetBinCenter(i)
        xLow  = histo.GetBinWidth(i)
        xHigh = xLow
        yVal  = histo.GetBinContent(i)
        yLow  = histo.GetBinError(i)
        yHigh = yLow
                                                                                                                                                                                                            
        # Store values                                                                                                                                                                                      
        x.append(xVal)
        xerrl.append(xLow)
        xerrh.append(xHigh)
        y.append(yVal)
        yerrl.append(yLow)
        yerrh.append(yHigh)
        
    # Create the TGraph with asymmetric errors                                                                                                                                                              
    tgraph = ROOT.TGraphAsymmErrors(len(x),
                                    array.array("d",x),
                                    array.array("d",y),
                                    array.array("d",xerrl),
                                    array.array("d",xerrh),
                                    array.array("d",yerrl),
                                    array.array("d",yerrh))
    if verbose:
        tgraph.GetXaxis().SetLimits(-0.05, 1.0)
    tgraph.SetName(histo.GetName())

    # Construct info table (debugging)
    table  = []
    align  = "{0:>6} {1:^10} {2:>10} {3:>10} {4:>10} {5:^3} {6:<10}"
    header = align.format("#", "x-", "x", "x+", "y", "+/-", "Error")
    hLine  = "="*70
    table.append("")
    table.append(hLine)
    table.append("{0:^70}".format(histo.GetName()))
    table.append(header)
    table.append(hLine)
                                                                                                                                                                                                            
    # For-loop: All values x-y and their errors
    for i, xV in enumerate(x, 0):
        row = align.format(i+1, "%.4f" % xerrl[i], "%.4f" %  x[i], "%.4f" %  xerrh[i], "%.5f" %  y[i], "+/-", "%.5f" %  yerrh[i])
        table.append(row)
    table.append(hLine)
                                                                                                                                                                                                            
    if 0:
        for i, line in enumerate(table, 1):
            print line
    return tgraph                              
        
def PlotOutput(Y_train, Y_test, saveDir, saveName, isSB, saveFormats):
    
    ROOT.gStyle.SetOptStat(0)

    # Create canvas
    canvas = plot.CreateCanvas()
    canvas.cd()
    canvas.SetLogy()

    # Create histograms
    h1 = ROOT.TH1F('train', '', 50, 0.0, 1.0)
    h2 = ROOT.TH1F('test' , '', 50, 0.0, 1.0)

    # Fill histograms
    for r in Y_train:
        h1.Fill(r)

    for r in Y_test:
        h2.Fill(r)

    if 0:
        h1.Scale(1./h1.Integral())
        h2.Scale(1./h2.Integral())

    ymax = max(h1.GetMaximum(), h2.GetMaximum())

    plot.ApplyStyle(h1, ROOT.kMagenta+1)
    plot.ApplyStyle(h2, ROOT.kGreen+2)

    for h in [h1,h2]:
        h.SetMinimum(100)
        h.SetMaximum(ymax*1.1)
        h.GetXaxis().SetTitle("Output")
        h.GetYaxis().SetTitle("Entries")
        h.Draw("HIST SAME")
    
    # What is this for? Ask soti
    graph = plot.CreateGraph([0.5, 0.5], [0, ymax*1.1])
    graph.Draw("same")
    
    # Create legend
    leg=plot.CreateLegend(0.6, 0.75, 0.9, 0.85)
    if isSB:
        leg.AddEntry(h1, "signal","l")
        leg.AddEntry(h2, "background","l")
    else:
        leg.AddEntry(h1, "train","l")
        leg.AddEntry(h2, "test","l")
    leg.Draw()

    plot.SavePlot(canvas, saveDir, saveName, saveFormats)
    canvas.Close()
    return
        
def PlotAndWriteJSON(signal, bkg, saveDir, saveName, jsonWr, saveFormats):

    resultsDict = {}
    resultsDict["signal"]     = signal
    resultsDict["background"] = bkg
 
   # Create canvas
    ROOT.gStyle.SetOptStat(0)
    canvas = plot.CreateCanvas()
    canvas.cd()

    hList = []
    gList = []
    yMin  = 100000
    yMax  = -1

    # For-loop: 
    for i, key in enumerate(resultsDict.keys(), 0):

        h = ROOT.TH1F(key, '', 50, 0.0, 1.0)             
        for j, x in enumerate(resultsDict[key], 0):
            h.Fill(x)
            try:
                yMin = min(x[0], yMin)
            except:
                pass
                
        # Save maximum
        yMax = max(h.GetMaximum(), yMax)

        # Customise & append to list
        plot.ApplyStyle(h, i+1)
        hList.append(h)

    if yMin <= 0.0:
        yMin = 100
    canvas.SetLogy()

    # For-loop: All histograms
    for i, h in enumerate(hList, 0):
        h.SetMinimum(yMin*0.85)        
        h.SetMaximum(yMax*1.15)
        h.GetXaxis().SetTitle("DNN output")
        if "output" in saveName.lower():
            h.GetYaxis().SetTitle("Entries")
        elif "efficiency" in saveName.lower():
            h.GetYaxis().SetTitle("Efficiency")
        elif "significance" in saveName.lower():
            h.GetYaxis().SetTitle("Significance")
        else:
            pass
            
        if i==0:
            h.Draw("HIST")
        else:
            h.Draw("HIST SAME")
    
    # Create legend
    leg=plot.CreateLegend(0.6, 0.75, 0.9, 0.85)
    for h in hList:
        leg.AddEntry(h, h.GetName(),"l")
    leg.Draw()

    plot.SavePlot(canvas, saveDir, saveName, saveFormats)
    canvas.Close()

    # Create TGraph
    for h in hList:
        gList.append(convertHistoToGaph(h))

    # Write the Tgraph into the JSON file
    for gr in gList:
        gName = "%s_%s" % (saveName, gr.GetName())
        jsonWr.addGraph(gName, gr)
    return

def PlotTGraph(xVals, xErrs, yVals, yErrs, saveDir, saveName, jsonWr, saveFormats):

    # Create a TGraph object
    graph = plot.GetGraph(xVals, yVals, xErrs, xErrs, yErrs, yErrs)
 
    # Create a TCanvas object
    ROOT.gStyle.SetOptStat(0)
    canvas = plot.CreateCanvas()
    canvas.cd()
    #canvas.SetLogy()
    
    # Create the TGraph with asymmetric errors                                                                                                                                                              
    tgraph = ROOT.TGraphAsymmErrors(len(xVals),
                                    array.array("d", xVals),
                                    array.array("d", yVals),
                                    array.array("d", xErrs),
                                    array.array("d", xErrs),
                                    array.array("d", yErrs),
                                    array.array("d", yErrs))
    tgraph.SetName(saveName)
    if "efficiency" in saveName.lower():
        legName = "efficiency"
        if saveName.lower().endswith("sig"):
            plot.ApplyStyle(tgraph, ROOT.kBlue)
        else:
            plot.ApplyStyle(tgraph, ROOT.kRed)
    elif "significance" in saveName.lower():
        legName = "significance"
        if saveName.lower().endswith("sig"):
            plot.ApplyStyle(tgraph, ROOT.kGreen)
        else:
            plot.ApplyStyle(tgraph, ROOT.kGreen+3)
    else:
        plot.ApplyStyle(tgraph, ROOT.kOrange)
        
    # Draw the TGraph
    tgraph.GetXaxis().SetLimits(-0.05, 1.0)
    #tgraph.SetMaximum(1.1)
    #tgraph.SetMinimum(0)
    tgraph.Draw("AC")
        
    # Create legend
    leg = plot.CreateLegend(0.60, 0.70, 0.85, 0.80)
    leg.AddEntry(tgraph, legName, "l")
    leg.Draw()

    plot.SavePlot(canvas, saveDir, saveName, saveFormats)
    canvas.Close()

    # Write the Tgraph into the JSON file
    jsonWr.addGraph(saveName, tgraph)
    return

def GetEfficiency(histo):

    # Initialize sigma variables
    nbins    = histo.GetNbinsX()
    intErrs  = ROOT.Double(0.0)
    intVals  = histo.IntegralAndError(0, nbins+1, intErrs, "")
    xVals    = []
    xErrs    = []
    yVals    = []
    yErrs    = []
    yTmp     = ROOT.Double(0.0)
    
    # For-loop: All bins
    for i in range(0, nbins+1):
        xVal = histo.GetBinCenter(i)
        if xVal < 0.0:
            continue
        xErr = histo.GetBinWidth(i)*0.5
        intBin = histo.IntegralAndError(i, nbins+1, yTmp, "")
        yVals.append(intBin/intVals)
        xVals.append(xVal)
        xErrs.append(xErr)
        yErrs.append(yTmp/intVals)
    return xVals, xErrs, yVals, yErrs


def CalcEfficiency(htest_s, htest_b):

    # Initialize sigma variables
    nbins    = htest_s.GetNbinsX()
    sigmaAll = ROOT.Double(0.0)
    sigmaSel = ROOT.Double(0.0)
    All_s    = htest_s.IntegralAndError(0, nbins+1, sigmaAll, "")
    All_b    = htest_b.IntegralAndError(0, nbins+1, sigmaAll, "")
    eff_s    = []
    eff_b    = [] 
    xvalue   = []
    error    = []
    
    # For-loop: All bins
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
    
    #print "%d: %s" % (len(xvalue), xvalue)
    #print "%d: %s" % (len(eff_s), eff_s)
    return xvalue, eff_s, eff_b, error

def CalcSignificance(htest_s, htest_b):
    nbins = htest_s.GetNbinsX()
    h_signif0=ROOT.TH1F('signif0', '', nbins, 0.0, 1.)
    h_signif1=ROOT.TH1F('signif1', '', nbins, 0.0, 1.)
    
    sigmaSel_s = ROOT.Double(0.0)
    sigmaSel_b = ROOT.Double(0.0)
    
    for i in range(0, nbins+1):
        # Get selected events                                                                                                                                                                       
        sSel = htest_s.IntegralAndError(i, nbins+1, sigmaSel_s, "")
        bSel = htest_b.IntegralAndError(i, nbins+1, sigmaSel_b, "")
        # Calculate Significance
        _sign0 = 0
        if (sSel+bSel > 0):
            _sign0 = sSel/math.sqrt(sSel+bSel)

        _sign1 = 2*(math.sqrt(sSel+bSel) - math.sqrt(bSel))
        h_signif0.Fill(htest_s.GetBinCenter(i), _sign0)        
        h_signif1.Fill(htest_s.GetBinCenter(i), _sign1)        
    return h_signif0, h_signif1

def GetSignificance(htest_s, htest_b):
    nbinsX     = htest_s.GetNbinsX()
    sigmaSel_s = ROOT.Double(0.0)
    sigmaSel_b = ROOT.Double(0.0)
    xVals      = []
    xErrs      = []
    signif_def = [] # S/sqrt(S+B) - same definition as TMVA
    signif_alt = [] # 2[sqrt(S+B) -sqrt(B)]

    # For-loop: All histogram bins
    for i in range(0, nbinsX+1):
        xVal = htest_s.GetBinCenter(i)
        if xVal < 0.0:
            continue
        xErr = htest_s.GetBinWidth(i)*0.5

        # Get selected events                                                                                                                                                                       
        sSel = htest_s.IntegralAndError(i, nbinsX+1, sigmaSel_s, "")
        bSel = htest_b.IntegralAndError(i, nbinsX+1, sigmaSel_b, "")

        # Calculate Significance
        _sign0 = 0
        if (sSel+bSel > 0):
            _sign0 = sSel/math.sqrt(sSel+bSel)
        _sign1 = 2*(math.sqrt(sSel+bSel) - math.sqrt(bSel))

        # Append values
        xVals.append(xVal)
        xErrs.append(xErr)
        signif_def.append(_sign0)
        signif_alt.append(_sign1)
    return xVals, xErrs, signif_def, signif_alt

def PlotEfficiency(htest_s, htest_b, saveDir, saveName, saveFormats):
    ROOT.gStyle.SetOptStat(0)
    canvas = plot.CreateCanvas()
    canvas.cd()
    canvas.SetLeftMargin(0.145)
    canvas.SetRightMargin(0.11)

    # Calculate signal and background efficiency vs output
    xvalue, eff_s, eff_b, error = CalcEfficiency(htest_s, htest_b)
    graph_s = plot.GetGraph(xvalue, eff_s, error, error, error, error)
    graph_b = plot.GetGraph(xvalue, eff_b, error, error, error, error)
    
    plot.ApplyStyle(graph_s, ROOT.kBlue)
    plot.ApplyStyle(graph_b, ROOT.kRed)
    
    # Calculate significance vs output
    h_signif0, h_signif1 = CalcSignificance(htest_s, htest_b)
    
    plot.ApplyStyle(h_signif0, ROOT.kGreen)
    plot.ApplyStyle(h_signif1, ROOT.kGreen+3)
    
    #=== Get maximum of significance
    maxSignif0 = h_signif0.GetMaximum()
    maxSignif1 = h_signif1.GetMaximum()
    maxSignif = max(maxSignif0, maxSignif1)
    
    # Normalize significance
    h_signifScaled0 = h_signif0.Clone("signif0")
    h_signifScaled0.Scale(1./float(maxSignif))

    h_signifScaled1 = h_signif1.Clone("signif1")
    h_signifScaled1.Scale(1./float(maxSignif))

    #Significance: Get new maximum
    ymax = max(h_signifScaled0.GetMaximum(), h_signifScaled1.GetMaximum())

    for obj in [graph_s, graph_b, h_signifScaled0, h_signifScaled1]:
        obj.GetXaxis().SetTitle("Output")
        obj.GetYaxis().SetTitle("Efficiency")
        obj.SetMaximum(ymax*1.1)
        obj.SetMinimum(0)
    #Draw    
    h_signifScaled0.Draw("HIST")
    h_signifScaled1.Draw("HIST SAME")
    graph_s.Draw("PL SAME")
    graph_b.Draw("PL SAME")

    graph = plot.CreateGraph([0.5, 0.5], [0, ymax*1.1])
    graph.Draw("same")

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
    rightAxis.SetLineColor ( signifColor )
    rightAxis.SetLabelColor( signifColor )
    rightAxis.SetTitleColor( signifColor )
    rightAxis.SetTitleOffset(1.25)
    rightAxis.SetLabelOffset(0.005)
    rightAxis.SetLabelSize(0.04)
    rightAxis.SetTitleSize(0.045)
    rightAxis.SetTitle( "Significance" )
    rightAxis.Draw()
        
    plot.SavePlot(canvas, saveDir, saveName, saveFormats)
    canvas.Close()
    return

def GetROC(htest_s, htest_b):
    '''
    Get ROC curve (signal efficiency vs bkg efficiency)
    '''
    # Calculate signal and background efficiency vs output
    xvalue, eff_s, eff_b, error = CalcEfficiency(htest_s, htest_b)
    graph_roc = plot.GetGraph(eff_s, eff_b, error, error, error, error)
    return graph_roc

def PlotROC(graphMap, saveDir, saveName, saveFormats):
    '''
    Plot ROC curves (signal efficiency vs bkg efficiency)
    '''
    ROOT.gStyle.SetOptStat(0)
    canvas = plot.CreateCanvas()
    canvas.cd()

    leg=plot.CreateLegend(0.50, 0.25, 0.85, 0.45)    

    for i in range(len(graphMap)):
        gr = graphMap["graph"][i]
        gr_name = graphMap["name"][i]
        plot.ApplyStyle(gr, i+2)
        gr.GetXaxis().SetTitle("Signal Efficiency")
        gr.GetYaxis().SetTitle("Misidentification rate")
        if i == 0:
            gr.Draw("apl")
        else:
            gr.Draw("pl same")
        leg.AddEntry(gr, gr_name, "l")

    leg.Draw("same")
    
    plot.SavePlot(canvas, saveDir, saveName, saveFormats)
    canvas.Close()
    return

def PlotOvertrainingTest(Y_train_S, Y_test_S, Y_train_B, Y_test_B, saveDir, saveName, saveFormats):
    def ApplyStyle(histo):
        if "_s" in histo.GetName():
            color = ROOT.kBlue
        else:
            color = ROOT.kRed
            
        plot.ApplyStyle(h, color)
        histo.SetMarkerSize(0.5)
        histo.GetXaxis().SetTitle("Output")
        histo.GetYaxis().SetTitle("Entries")
        histo.SetMinimum(10)
        return
        
    def GetLegendStyle(histoName):
        if "test" in histoName:
            legStyle = "f"
            if "_s" in histoName:
                legText = "signal (test)"
            else:
                legText = "background (test)"
        elif "train" in histoName:
            legStyle = "p"
            if "_s" in histoName:
                legText = "signal (train)"
            else:
                legText = "background (train)"
        return legText, legStyle

    def DrawStyle(histoName):
        if "train" in histoName:
            _style = "P"
        else:
            _style = "HIST"
        return _style

    ROOT.gStyle.SetOptStat(0)
    canvas = plot.CreateCanvas()
    canvas.cd()
    canvas.SetLogy()

    hList     = []
    DataList = [Y_train_S, Y_test_S, Y_train_B, Y_test_B]
    ymax     = 0
    nbins    = 500
    
    # Create the histograms
    htrain_s = ROOT.TH1F('train_s', '', nbins, 0.0, 1.0)
    htest_s  = ROOT.TH1F('test_s' , '', nbins, 0.0, 1.0)
    htrain_b = ROOT.TH1F('train_b', '', nbins, 0.0, 1.0)
    htest_b  = ROOT.TH1F('test_b' , '', nbins, 0.0, 1.0)

    # Append to list
    hList.append(htrain_s)
    hList.append(htest_s)
    hList.append(htrain_b)
    hList.append(htest_b)
    
    for i in range(len(DataList)):
        for r in DataList[i]:
            hList[i].Fill(r)            

    # Clone the histograms
    htrain_s1 = htrain_s.Clone("train_s")
    htrain_b1 = htrain_b.Clone("train_b")
    htest_s1  = htest_s.Clone("test_s")
    htest_b1  = htest_b.Clone("test_b")
    drawStyle = "HIST SAME"
    leg=plot.CreateLegend(0.55, 0.68, 0.85, 0.88)

    for h in hList:
        h.Rebin(10)
        # Legend
        legText, legStyle = GetLegendStyle(h.GetName())
        leg.AddEntry(h, legText, legStyle)
        ApplyStyle(h)

    ymax = max(htrain_s.GetMaximum(), htest_s.GetMaximum(), htrain_b.GetMaximum(), htest_b.GetMaximum())
    for h in hList:
        h.SetMaximum(ymax*2)        
        h.Draw(DrawStyle(h.GetName())+" SAME")

    #graph = plot.CreateGraph([0.5, 0.5], [0, ymax*2])
    #graph.Draw("same")
    #leg.Draw()
    
    # Save & close canvas
    plot.SavePlot(canvas, saveDir, saveName, saveFormats)
    canvas.Close()
    return htrain_s1, htest_s1, htrain_b1, htest_b1
            
def WriteModel(model, model_json, output):
    '''
    Write model weights and architecture in txt file
    '''
    arch = json.loads(model_json)
    with open(output, 'w') as fout:
        # Store number of layers
        fout.write( 'layers ' + str(len(model.layers)) + '\n')
        layers = []
        
        # Iterate over each layer
        for index, l in enumerate(arch["config"]):

            # Store type of layer
            fout.write(l['class_name'] + '\n')
            #layers += [l['class_name']]

            # Convolution2D layer
            if l['class_name'] == 'Convolution2D':
                # Get weights of layer
                W = model.layers[index].get_weights()[0]
                fout.write(str(W.shape[0]) + ' ' + str(W.shape[1]) + ' ' + str(W.shape[2]) + ' ' + str(W.shape[3]) + ' ' + l['config']['border_mode'] + '\n')

                for i in range(W.shape[0]):
                    for j in range(W.shape[1]):
                        for k in range(W.shape[2]):
                            fout.write(str(W[i,j,k]) + '\n')
                fout.write(str(model.layers[index].get_weights()[1]) + '\n')

            # Activation layer
            if l['class_name'] == 'Activation':
                # Store activation function
                fout.write(l['config']['activation'] + '\n')

            # MaxPooling2D layer
            if l['class_name'] == 'MaxPooling2D':
                fout.write(str(l['config']['pool_size'][0]) + ' ' + str(l['config']['pool_size'][1]) + '\n')

            # Dense layer
            if l['class_name'] == 'Dense':
                # Store number of inputs, outputs for each layer
                W = model.layers[index].get_weights()[0]                
                fout.write(str(W.shape[0]) + ' ' + str(W.shape[1]) + '\n')
                
                for w in W:
                    # Store weights
                    fout.write(str(w) + '\n')
                # Store bias values (shifts the activation function : output[i] = (Sum(weights[i,j]*inputs[j]) + bias[i]))
                biases = model.layers[index].get_weights()[1]
                fout.write(str(biases) + '\n')
        Print('Writing model in file %s' % (output), True)
        return
