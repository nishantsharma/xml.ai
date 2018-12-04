import os, glob

import torch, torchtext
from torchtext.vocab import Vocab

import xml.etree.ElementTree as ET

class Hier2HierExample(torchtext.data.Example):
    def __init__(self, inXml, outStr):
        self.src = inXml
        self.tgt = outStr

class Hier2HierDataset(torchtext.data.Dataset):
    def __init__(self, baseFolder='train/folder_1/', fields=None, loader=None, transform=None):
        self.baseFolder = baseFolder

        # Build input file pairs.
        self.filePairs = []
        inputFiles = glob.glob(self.baseFolder + "dataIn_*.xml")
        for inputFileName in inputFiles:
            index = int(inputFileName[len(self.baseFolder + "dataIn_"):-4])
            outputFileName = self.baseFolder + "dataIn_" + str(index) + ".xml"
            if not os.path.exists(outputFileName):
                continue
            self.filePairs.append((inputFileName, outputFileName))
        
        if loader is None:
            loader = ET.parse
        self.loader = loader

        if transform is None:
            transform = lambda x:x
        self.transform = transform
        
        examples = []
        for inFile, outFile in self.filePairs:
            inXml = self.transform(self.loader(inFile))
            with open(outFile, "r") as fp:
                outStr = fp.read()
            examples.append(Hier2HierExample(inXml, outStr))

        super().__init__(examples, fields=fields)
