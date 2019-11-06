#================================================================================================   
# Import modules
#================================================================================================   
import sys
import os
import ROOT

#================================================================================================   
# Class definition
#================================================================================================   
class JsonWriter:

    def __init__(self, saveDir="", verbose=False):
        self.graphs = {}
        self.parameters = {}
        self.verbose = verbose
        self.saveDir = saveDir
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

    def addParameter(self,name,value):
        self.parameters[name] = value
        
    def addGraph(self, name, graph):
        self.graphs[name] = graph

    def timeStamp(self):
        import datetime
        time = datetime.datetime.now().ctime()
        return time
    
    def write(self, fileName, fileMode="w"):
        
        filePath = os.path.join(self.saveDir, fileName)
        self.Verbose("Opening file %s in %s mode" % (filePath, fileMode), True)
        fOUT = open(filePath, fileMode)

        self.Verbose("Writing timestamp to file %s" % (filePath), True)
        fOUT.write("{\n")
        time = self.timeStamp()
        fOUT.write("  \"timestamp\": \"Generated on " + time + " by jsonWriter.py\",\n")

        self.Verbose("Writing all parameters (=%d) to file %s" % (len(self.parameters.keys()), filePath), True)
        # For-loop: All parameter keys-values
        for key in self.parameters.keys():
            fOUT.write("  \""+key+"\": \"%s\",\n" % self.parameters[key])

        self.Verbose("Writing all graphs (=%d) to file %s" % (len(self.graphs.keys()), filePath), True)
        nkeys = 0
        # For-loop: All graphs
        for key in self.graphs.keys():
            fOUT.write("  \""+key+"\": [\n")
            # For-loop: All entries
            for i in range(self.graphs[key].GetN()):                
                x = self.graphs[key].GetX()
                y = self.graphs[key].GetY()
                if 0:
                    print "\t x = %s, y = %s" % (x[i], y[i])
                comma = ","
                if i == self.graphs[key].GetN() - 1:
                    comma = ""
                fOUT.write("      { \"x\": %s, \"y\": %s }%s\n"%(x[i],y[i],comma))
            nkeys+=1
            if nkeys < len(self.graphs.keys()):
                fOUT.write("  ],\n")
            else:
                fOUT.write("  ]\n")

        # Write and close the file
        fOUT.write("}\n")
        fOUT.close()
        self.Verbose("Created file %s" % (filePath), True)
        return
