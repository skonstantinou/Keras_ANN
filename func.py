import ROOT
import plot
import math
import array
import json
###########################################################
# Plot Output
###########################################################

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

    plot.SavePlot(canvas, saveDir, saveName)
    canvas.Close()
    return

###########################################################
# Plot Efficiency
###########################################################

def PlotEfficiency(htest_s, htest_b, saveDir, saveName):
    ROOT.gStyle.SetOptStat(0)
    canvas = ROOT.TCanvas("canvas", "canvas",0,0,800,800)
    canvas.SetLeftMargin(0.12)
    canvas.SetRightMargin(0.12)
    canvas.SetTopMargin(0.06)
    canvas.SetBottomMargin(0.13)
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
        #print Sel_s/All_s, Sel_b/All_b, htest_s.GetBinCenter(i)
    graph_s = plot.GetGraph(xvalue, eff_s, error, error, error, error)
    graph_b = plot.GetGraph(xvalue, eff_b, error, error, error, error)

    graph_s.SetLineColor(ROOT.kBlue)
    graph_s.SetMarkerColor(ROOT.kBlue)
    graph_s.SetMarkerStyle(8)
    graph_s.SetMarkerSize(0.5)
    graph_s.SetLineWidth(3)


    graph_b.SetLineColor(ROOT.kRed)
    graph_b.SetMarkerColor(ROOT.kRed)
    graph_b.SetMarkerStyle(8)
    graph_b.SetMarkerSize(0.5)
    graph_b.SetLineWidth(3)

    graph_s.SetMinimum(0)
    graph_b.SetMinimum(0)
    
    #
    h_signif0=ROOT.TH1F('signif0', '', nbins, 0.0, 1.)
    h_signif1=ROOT.TH1F('signif1', '', nbins, 0.0, 1.)

    #sign = []
    sigmaSel_s = ROOT.Double(0.0)
    sigmaSel_b = ROOT.Double(0.0)

    for i in range(0, nbins+1):
        # Get selected events                                                                                                                                                                       
        sSel = htest_s.IntegralAndError(i, nbins+1, sigmaSel_s, "")
        bSel = htest_b.IntegralAndError(i, nbins+1, sigmaSel_b, "")
        # Calculate Significance                                                                                                                                                                    
        _sign0 = sSel/math.sqrt(sSel+bSel)
        _sign1 = 2*(math.sqrt(sSel+bSel) - math.sqrt(bSel))
        #sign.append(_sign0)
        h_signif0.Fill(htest_s.GetBinCenter(i), _sign0)        
        h_signif1.Fill(htest_s.GetBinCenter(i), _sign1)        

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
    
    ### Put in func
    h_signifScaled0.GetXaxis().SetLabelSize(0.045)
    h_signifScaled0.GetXaxis().SetTitleSize(0.05)
    h_signifScaled0.GetXaxis().SetTitleOffset(1.)
    h_signifScaled0.GetXaxis().SetTitleFont(42)
    
    h_signifScaled0.GetYaxis().SetLabelSize(0.045)
    h_signifScaled0.GetYaxis().SetTitleSize(0.05)
    h_signifScaled0.GetYaxis().SetLabelFont(42)
    h_signifScaled0.GetYaxis().SetLabelOffset(0.007)
    h_signifScaled0.GetYaxis().SetTitleOffset(1.2)
    h_signifScaled0.GetYaxis().SetTitleFont(42)
    
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
    rightAxis.SetTitleOffset(1.25)
    rightAxis.SetLabelOffset(0.005)
    rightAxis.SetLabelSize(0.04)
    rightAxis.SetTitleSize(0.045)
    rightAxis.SetTitle( "Significance" )
    rightAxis.Draw()
        

    plot.SavePlot(canvas, saveDir, saveName)
    canvas.Close()
    return


###########################################################
# Plot Overtraining test
###########################################################

def PlotOvertrainingTest(Y_train_S, Y_test_S, Y_train_B, Y_test_B, saveDir, saveName):
    ROOT.gStyle.SetOptStat(0)
    canvas = ROOT.TCanvas("canvas", "canvas",0,0,800,800)
    canvas.SetLeftMargin(0.12)
    #canvas.SetRightMargin(0.12)
    canvas.SetTopMargin(0.06)
    canvas.SetBottomMargin(0.13)
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
        h.GetXaxis().SetLabelSize(0.045)
        h.GetXaxis().SetTitleSize(0.05)
        h.GetXaxis().SetTitleOffset(1.)
        h.GetXaxis().SetTitleFont(42)

        h.GetYaxis().SetTitle("Entries")
        h.GetYaxis().SetLabelSize(0.045)
        h.GetYaxis().SetTitleSize(0.05)
        h.GetYaxis().SetLabelFont(42)
        h.GetYaxis().SetLabelOffset(0.007)
        h.GetYaxis().SetTitleOffset(1.2)
        h.GetYaxis().SetTitleFont(42)

        h.Rebin(10)
        # Legend
        legText, legStyle = plot.GetLegendStyle(h.GetName())
        leg.AddEntry(h, legText, legStyle)
        
        plot.ApplyStyle(h)
        h.Draw(plot.DrawStyle(h.GetName())+" SAME")

    graph = plot.CreateGraph([0.5, 0.5], [0, ymax*2])
    graph.Draw("same")
    leg.Draw()

    plot.SavePlot(canvas, saveDir, saveName)
    canvas.Close()
    return htrain_s1, htest_s1, htrain_b1, htest_b1


###########################################################
# Write model weights and architecture in txt file
###########################################################

def WriteModel(model, model_json, output):
    arch = json.loads(model_json)
    with open(output, 'w') as fout:
        fout.write('layers ' + str(len(model.layers)) + '\n')
        print "TYPE", type(arch)
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
