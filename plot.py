import ROOT
import array
import os
import getpass

def CreateCanvas():
    canvas = ROOT.TCanvas("canvas", "canvas",0,0,800,800)
    canvas.SetLeftMargin(0.12)
    canvas.SetRightMargin(0.12)
    canvas.SetTopMargin(0.06)
    canvas.SetBottomMargin(0.13)
    return canvas

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

def getDirName(dirName, baseDir=None):
    print "getDirName() => FIXME! Do not assume full save path"
    usrName = getpass.getuser() 
    usrInit = usrName[0]
    dirName = dirName.replace(".", "p")
    dirName = "/afs/cern.ch/user/%s/%s/public/html/%s" % (usrInit, usrName, dirName)
    return dirName

def SavePlot(canvas, saveDir, saveName, saveFormats=["pdf", "root"]):
    
    # Create output directory if it does not exist
    if not os.path.exists(saveDir):
        os.mkdir(saveDir)
        print "Directory " , saveDir ,  " has been created "
    else:
        if 0:
            print "Output saved under", saveDir
            
    savePath = "%s/%s" % (saveDir, saveName)
    usrInit =  getpass.getuser()[0]
    saveURL  = savePath.replace("/afs/cern.ch/user/%s/" %(usrInit),"https://cmsdoc.cern.ch/~")
    saveURL  = saveURL.replace("/public/html/","/")
    
    for ext in saveFormats:
        fileName = "%s.%s" % (savePath, ext)
        canvas.SaveAs( fileName )
    print "=== ", saveURL
    return

def ApplyStyle(h, color):
    h.SetLineColor(color)
    h.SetMarkerColor(color)
    h.SetMarkerStyle(8)
    h.SetMarkerSize(0.5)
    h.SetLineWidth(3)
    h.SetMinimum(0)
    
    h.GetXaxis().SetTitle("Output")
    h.GetXaxis().SetLabelSize(0.045)
    h.GetXaxis().SetTitleSize(0.05)
    h.GetXaxis().SetTitleOffset(1.)
    h.GetXaxis().SetTitleFont(42)
    
    h.GetYaxis().SetTitle("Efficiency")
    h.GetYaxis().SetLabelSize(0.045)
    h.GetYaxis().SetTitleSize(0.05)
    h.GetYaxis().SetLabelFont(42)
    h.GetYaxis().SetLabelOffset(0.007)
    h.GetYaxis().SetTitleOffset(1.2)
    h.GetYaxis().SetTitleFont(42)
    
    h.SetTitle("")
    return

