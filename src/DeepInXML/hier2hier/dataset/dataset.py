import os, glob
from attrdict import AttrDict

import torch, torchtext
from torchtext.vocab import Vocab

import xml.etree.ElementTree as ET
from .randomXml import generateXml
from hier2hier.util import invertPermutation, methodProfiler, lastCallProfile

class Hier2HierExample(torchtext.data.Example):
    def __init__(self, inXml, outStr):
        self.src = inXml
        self.tgt = outStr

class GeneratedXmlDataset(torchtext.data.Dataset):
    @staticmethod
    def defaultOutputTransform(tree):
        allNodes = list(tree.getroot().iter())
        return ET.tostring(allNodes[int(len(allNodes)/2)])

    def __init__(self, inputSpec, fields=None, outputTransform=None):
        if outputTransform is None:
            outputTransform = self.defaultOutputTransform

        if isinstance(inputSpec, tuple):
            sampleCount, generatorArgs = inputSpec
            examples = []
            for n in range(sampleCount):
                inXml = generateXml(generatorArgs)
                outStr = outputTransform(inXml)
                examples.append(Hier2HierExample(inXml, outStr))
        elif isinstance(inputSpec, list):
            examples = inputSpec
    
        super().__init__(examples, fields=fields)
        

class Hier2HierDataset(torchtext.data.Dataset):
    def __init__(self, baseFolder='train/folder_1/', fields=None, loader=None, transform=None, selectPercent=None):
        # Build input file pairs.
        self.filePairs = []
        inputFiles = list(glob.glob(baseFolder + "dataIn_*.xml"))
        inputFiles.sort()
        if selectPercent is not None:
            inputFiles = inputFiles[0:int(0.01 * selectPercent * len(inputFiles))]
        
        for inputFileName in inputFiles:
            index = int(inputFileName[len(baseFolder + "dataIn_"):-4])
            outputFileName = baseFolder + "dataOut_" + str(index) + ".xml"
            if not os.path.exists(outputFileName):
                continue
            self.filePairs.append((inputFileName, outputFileName))
        
        if loader is None:
            loader = ET.parse
        #self.loader = loader

        if transform is None:
            transform = lambda x:x
        #self.transform = transform
        
        examples = []
        for inFile, outFile in self.filePairs:
            inXml = transform(loader(inFile))
            with open(outFile, "r") as fp:
                outStr = fp.read()
            examples.append(Hier2HierExample(inXml, outStr))

        super().__init__(examples, fields=fields)

class AttrTuple(object):
    pass

class Hier2HierIterator(torchtext.data.BucketIterator):
    def __init__(self, *argc, preprocess_batch=None, **kargv):
        if preprocess_batch is None:
            preprocess_batch = lambda x:(x, None)
        self.preprocess_batch = preprocess_batch

        self.savedBatches = None
        super().__init__(*argc, **kargv)

    @methodProfiler
    def __iter__(self, unitTesting=False):
        if self.savedBatches is None:
            self.savedBatches = []
            for batch in super().__iter__():
                savedBatchData = AttrTuple()
                processedBatch = self.preprocess_batch(batch)
                savedBatchData.sampleCount = len(processedBatch.torchBatch.src)
                savedBatchData.encodedNodesByNdfo = processedBatch.encodedNodesByNdfo
                savedBatchData.parentSelectorByNdfo = processedBatch.parentSelectorByNdfo
                savedBatchData.childSelectorByNdfoList = processedBatch.childSelectorByNdfoList
                savedBatchData.decreasingFanoutsFactorByNdfo = processedBatch.decreasingFanoutsFactorByNdfo
                savedBatchData.encodedAttrLabelsByAvdl = processedBatch.encodedAttrLabelsByAvdl
                savedBatchData.encodedAttrSymbolsByAvdlp = processedBatch.encodedAttrSymbolsByAvdlp
                savedBatchData.avdl2Ndac = torch.LongTensor(processedBatch.avdl2Ndac)
                savedBatchData.ndac2Ndfo = torch.LongTensor(processedBatch.ndac2Ndfo)
                savedBatchData.avdl2Ndfo = torch.LongTensor(processedBatch.avdl2Ndfo)
                savedBatchData.avdlAttrSelectorsListByNdac = processedBatch.avdlAttrSelectorsListByNdac
                savedBatchData.decreasingAttrCountsFactorByNdac = processedBatch.decreasingAttrCountsFactorByNdac
                savedBatchData.encodedTextByTtDLP = processedBatch.encodedTextByTtDLP
                savedBatchData.encodedTailByTlDLP = processedBatch.encodedTailByTlDLP
                savedBatchData.ndttl2Ndac = torch.LongTensor(processedBatch.ndttl2Ndac)
                savedBatchData.ndtll2Ndttl = torch.LongTensor(processedBatch.ndtll2Ndttl)
                savedBatchData.ndfo2Ndtll = torch.LongTensor(processedBatch.ndfo2Ndtll)
                savedBatchData.targetOutputsByTdol = processedBatch.targetOutputsByTdol
                savedBatchData.targetOutputLengthsByTdol = processedBatch.targetOutputLengthsByTdol
                savedBatchData.targetOutputsByTdolList = processedBatch.targetOutputsByTdolList
                savedBatchData.tdol2Toi = processedBatch.tdol2Toi
                savedBatchData.toi2Tdol = processedBatch.toi2Tdol
                savedBatchData.gndtol2Tdol = processedBatch.gndtol2Tdol
                savedBatchData.gndtol2Gni = torch.LongTensor(processedBatch.gndtol2Gni)
                savedBatchData.posNbrhoodGraphByGndtol = processedBatch.posNbrhoodGraphByGndtol
                savedBatchData.fullSpotlight = processedBatch.fullSpotlight
                savedBatchData.targetOutputsByToi = processedBatch.targetOutputsByToi
                savedBatchData.targetOutputLengthsByToi = processedBatch.targetOutputLengthsByToi

                # Test attrs
                if unitTesting:
                    savedBatchData.ndfo2Node = processedBatch.ndfo2Node
                    savedBatchData.attrsByAdfo = processedBatch.attrsByAdfo
                    savedBatchData.avdl2Adfo = processedBatch.avdl2Adfo
                    savedBatchData.encodedTextByNdtlp2 = processedBatch.encodedTextByNdtlp2
                    savedBatchData.ndfo2Ndtl2 = processedBatch.ndfo2Ndtl2
                    savedBatchData.toi2Tdol = processedBatch.toi2Tdol
                    savedBatchData.gni2Tdol = processedBatch.gni2Tdol
                    savedBatchData.inputs = processedBatch.torchBatch.src
                    savedBatchData.outputs = processedBatch.torchBatch.tgt

                    savedBatchData.ndfo2Toi = processedBatch.ndfo2Toi
                    savedBatchData.ndac2Toi = processedBatch.ndac2Toi
                    savedBatchData.avdl2Toi = processedBatch.avdl2Toi
                    savedBatchData.avdlp2Toi = processedBatch.avdlp2Toi
                    savedBatchData.ndtl2Toi2 = processedBatch.ndtl2Toi2
                    savedBatchData.ndtlp2Toi2 = processedBatch.ndtlp2Toi2
                    savedBatchData.gni2Toi = processedBatch.gni2Toi
                    savedBatchData.gndtol2Toi = processedBatch.gndtol2Toi
                self.savedBatches.append(savedBatchData)

        for processedBatch in self.savedBatches:
            yield processedBatch
