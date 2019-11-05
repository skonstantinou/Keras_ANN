import ROOT
import plot
import math
import array
import json

###########################################################
# Plot Output
###########################################################

def PlotOutput(Y_train, Y_test, saveDir, saveName, isSB, saveFormats):
    
    ROOT.gStyle.SetOptStat(0)
    canvas = plot.CreateCanvas()
    canvas.cd()
    canvas.SetLogy()
    h1=ROOT.TH1F('train', '', 50, 0.0, 1.)
    for r in Y_train:
        h1.Fill(r)

    h2=ROOT.TH1F('test', '', 50, 0.0, 1.)
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

    graph = plot.CreateGraph([0.5, 0.5], [0, ymax*1.1])
    graph.Draw("same")
    
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

###########################################################
# Canculate Efficiency
###########################################################

def CalcEfficiency(htest_s, htest_b):
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
    return xvalue, eff_s, eff_b, error

###########################################################
# Canculate Significance
###########################################################
    
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

###########################################################
# Plot Efficiency
###########################################################

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

###########################################################
# Get ROC curve (signal efficiency vs bkg efficiency)
###########################################################

def GetROC(htest_s, htest_b):
    # Calculate signal and background efficiency vs output
    xvalue, eff_s, eff_b, error = CalcEfficiency(htest_s, htest_b)
    graph_roc = plot.GetGraph(eff_s, eff_b, error, error, error, error)
    return graph_roc

###########################################################
# Plot ROC curves (signal efficiency vs bkg efficiency)
###########################################################

def PlotROC(graphMap, saveDir, saveName, saveFormats):

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


###########################################################
# Plot Overtraining test
###########################################################

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

    htrain_s1 = htrain_s.Clone("train_s")
    htrain_b1 = htrain_b.Clone("train_b")
    htest_s1  = htest_s.Clone("test_s")
    htest_b1  = htest_b.Clone("test_b")

    drawStyle = "HIST SAME"
    leg=plot.CreateLegend(0.55, 0.18, 0.85, 0.38)
    for h in hList:
        h.Rebin(10)
        ymax = max (htrain_s.GetMaximum(), htest_s.GetMaximum(), htrain_b.GetMaximum(), htest_b.GetMaximum())
        h.SetMaximum(ymax*2)
        # Legend
        legText, legStyle = GetLegendStyle(h.GetName())
        leg.AddEntry(h, legText, legStyle)
        
        ApplyStyle(h)
        h.Draw(DrawStyle(h.GetName())+" SAME")

    graph = plot.CreateGraph([0.5, 0.5], [0, ymax*2])
    graph.Draw("same")
    leg.Draw()

    plot.SavePlot(canvas, saveDir, saveName, saveFormats)
    canvas.Close()
    return htrain_s1, htest_s1, htrain_b1, htest_b1
    
    
###########################################################
# Write model weights and architecture in txt file
###########################################################

def WriteModel(model, model_json, output):
    arch = json.loads(model_json)
    with open(output, 'w') as fout:
        fout.write('layers ' + str(len(model.layers)) + '\n')
        layers = []
        for ind, l in enumerate(arch["config"]):
            fout.write('layer ' + str(ind) + ' ' + l['class_name'] + '\n')
            layers += [l['class_name']]
            if l['class_name'] == 'Convolution2D':

                W = model.layers[ind].get_weights()[0]
                fout.write(str(W.shape[0]) + ' ' + str(W.shape[1]) + ' ' + str(W.shape[2]) + ' ' + str(W.shape[3]) + ' ' + l['config']['border_mode'] + '\n')

                for i in range(W.shape[0]):
                    for j in range(W.shape[1]):
                        for k in range(W.shape[2]):
                            fout.write(str(W[i,j,k]) + '\n')
                fout.write(str(model.layers[ind].get_weights()[1]) + '\n')

            if l['class_name'] == 'Activation':
                fout.write(l['config']['activation'] + '\n')
            if l['class_name'] == 'MaxPooling2D':
                fout.write(str(l['config']['pool_size'][0]) + ' ' + str(l['config']['pool_size'][1]) + '\n')
            if l['class_name'] == 'Dense':
                W = model.layers[ind].get_weights()[0]
                fout.write(str(W.shape[0]) + ' ' + str(W.shape[1]) + '\n')

                for w in W:
                    fout.write(str(w) + '\n')
                fout.write(str(model.layers[ind].get_weights()[1]) + '\n')

        print 'Writing model in', output
        return
