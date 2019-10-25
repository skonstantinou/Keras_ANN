import ROOT
import array
import os

def CreateLegend(xmin=0.55, ymin=0.75, xmax=0.85, ymax=0.85):
    leg = ROOT.TLegend(xmin, ymin, xmax, ymax)
    leg.SetFillStyle(0)
    leg.SetBorderSize(0)
    leg.SetTextSize(0.040)
    return leg

def TFtext(tftext, x=0.3, y=0.8):
    tex = ROOT.TLatex(x, y, tftext)
    tex.SetNDC()
    tex.SetTextAlign(31)
    tex.SetTextFont(42)
    tex.SetTextSize(0.040)
    tex.SetLineWidth(2)
    return tex

def CreateGraph(gx, gy):
    graph=ROOT.TGraph(2, array.array("d",gx), array.array("d",gy)) 
    graph.SetFillColor(1)
    graph.SetLineColor(ROOT.kBlack)
    graph.SetLineStyle(3)
    graph.SetLineWidth(2)
    return graph

def GetGraph(x, y, xerrl, xerrh, yerrl, yerrh):
    graph = ROOT.TGraphAsymmErrors(len(x), array.array("d",x), array.array("d",y),
                                   array.array("d",xerrl), array.array("d",xerrh),
                                   array.array("d",yerrl), array.array("d",yerrh))
    return graph

def GetRatioStyle(h_ratio, ytitle, xtitle, ymax=2, ymin=0):
    h_ratio.SetMaximum(ymax)
    h_ratio.SetMinimum(ymin)
    h_ratio.GetYaxis().SetTitleOffset(0.5)
    h_ratio.SetTitle("")
    h_ratio.GetYaxis().SetTitle(ytitle)
    h_ratio.GetXaxis().SetTitle(xtitle)
    h_ratio.GetYaxis().SetLabelSize(0.09)
    h_ratio.GetXaxis().SetLabelSize(0.09)
    h_ratio.GetYaxis().SetTitleSize(0.095)
    h_ratio.GetXaxis().SetTitleSize(0.095)
    h_ratio.GetXaxis().SetTickLength(0.08)
    h_ratio.GetYaxis().SetTitleOffset(0.5)
    return h_ratio

def getDirName(dirName):
    dirName = dirName.replace(".", "p")
    dirName = "/afs/cern.ch/user/s/skonstan/public/html/"+dirName
    return dirName

def SavePlot(canvas, saveDir, saveName):
    
    # Create output directory if it does not exist
    if not os.path.exists(saveDir):
        os.mkdir(saveDir)
        print "Directory " , saveDir ,  " has been created "
    else:
        print "Output saved under", saveDir

    savePath = "%s/%s" % (saveDir, saveName)
    saveURL  = savePath.replace("/afs/cern.ch/user/s/","https://cmsdoc.cern.ch/~")
    saveURL  = saveURL.replace("/public/html/","/")
    canvas.SaveAs(savePath)
    savePath = savePath.replace("pdf","root")
    canvas.SaveAs(savePath)
    print "=== ", saveURL
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
            legText = "signal (test)"
    return legText, legStyle

def ApplyStyle(histo):
    if "_s" in histo.GetName():
        color = ROOT.kBlue
    else:
        color = ROOT.kRed

    histo.SetMarkerColor(color)
    histo.SetLineColor(color)
    histo.SetLineWidth(2)
    histo.SetMarkerStyle(8)
    histo.SetMarkerSize(0.8)
    return

def DrawStyle(histoName):
    if "train" in histoName:
        _style = "P"
    else:
        _style = "HIST"
    return _style