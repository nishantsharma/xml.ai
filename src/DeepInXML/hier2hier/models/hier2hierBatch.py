"""
    To efficiently process a batch of XML trees in GPU, we need to permute
    xmlTrees, their nodes, their attributes and text fields in various orders.
    
    Hier2hierBatch implements all those permutations and mappings during batch
    pre-processing.
"""
import copy, inspect
from os.path import basename
from attrdict import AttrDict
import torch
import torch.nn.utils.rnn as rnn

from hier2hier.util import invertPermutation, blockProfiler, methodProfiler, lastCallProfile, longTensor, AppMode
from hier2hier.dataset import Hier2HierIterator

class cached_property_profiler(object):
    def __init__(self, func):
        self.func = func

    def __get__(self, obj, cls):
        if obj is None:
            return self
        name = self.func.__name__
        
        if not name in obj.__dict__:
            with blockProfiler(name):
                obj.__dict__[name] = self.func(obj)

        return obj.__dict__[name]

class Hier2hierBatch(object):
    def __init__(self, torchBatch, device):
        """
            Important Orderings:
                For Trees:
                    TOI: Trees in Original Index order.
                    TDNC: Trees in Decreasing Node Count order.
                    TDOL: Tree in Decreasing Output Length order
                For Nodes:
                    NDFO: Batch Nodes in Decreasing FanOut order.
                    NDAC: Batch Nodes in Decreasing Attribute Count order.
                    NDTL[0/1]: Batch Nodes in Decreasing Text/Tail Length order.
                    NDTtL/NDTlL: Same as NDTL[0/1].
                    NDTLP[0/1]: Resultant ordering of positions within text/tail of nodes
                            upon packing these text/tail already in NDTL order.
                    NDTtLP/NDTlLP: Same as NDTLP[0/1].
                For Attributes:
                    AVDL: Batch Attriutes Values in Decreasing Length order.
                    AVDLP: Resultant ordering of positions within attributes, upon packing
                        attribute values already in AVDL order.
                    ADFO: Batch Attributes in DFO order by nodes.
                    AVDL: Batch Attribute Values in decreasing length order.
                For Text:
                    TtDLP: By decreasing text length.
                For Tail:
                    TlDLP: By decreasing tail length.
                For Graph nodes:
                    GNI: Graph nodes in default order.(NDFO + AVDL + TtDLP + TlDLP)
                    GNDTOL: Graph nodes in TDOL order of their trees.
        """
        self.device = device 
        self.attribs = {}
        self.torchBatch = torchBatch

    @cached_property_profiler
    def sampleCount(self):
        return len(self.torchBatch.src)

    @cached_property_profiler
    def inputs(self):
        return self.torchBatch.src

    @cached_property_profiler
    def outputs(self):
        tgt, tgtLengths = self.torchBatch.tgt
        tgt = torch.tensor(tgt, device=self.device)
        tgtLengths = longTensor(tgtLengths, device=self.device)
        return tgt, tgtLengths

    @cached_property_profiler
    def node2Parent(self):
        # Build node2Parent dictionary.
        retval = { }
        for toi, xmlTree in enumerate(self.torchBatch.src):
            treeRoot = xmlTree.getroot()
            retval[treeRoot] = treeRoot
            for node in xmlTree.iter():
                for childNode in node:
                    retval[childNode] = node
        return retval

    @cached_property_profiler
    def node2Toi(self):
        # Build node2Toi(original tree index) dictionary.
        retval =  { }
        for toi, xmlTree in enumerate(self.torchBatch.src):
            for node in xmlTree.iter():
                retval[node] = toi
        return retval

    @cached_property_profiler
    def tdnc2Toi(self):
        # Compute TDNC permutation for trees.
        retval = list(range(len(self.torchBatch.src)))
        retval.sort(lambda index: (-len(self.torchBatch.src[index]), index))
        return retval

    @cached_property_profiler
    def toi2Tdnc(self):
        # Invert TDNC permutation for trees.
        return invertPermutation(self.tdnc2Toi)

    @cached_property_profiler
    def ndfo2Node(self):
        # Permute nodes into NDFO order.
        retval = sum([ list(xmlTree.iter()) for xmlTree in self.torchBatch.src ], [])
        retval.sort(key=lambda node:-len(node)) # Fanout = len(node)
        return retval

    @cached_property_profiler
    def decreasingFanoutsFactorByNdfo(self):
        return torch.tensor([float(len(node)) for node in self.ndfo2Node if len(node)], device=self.device)

    @cached_property_profiler
    def node2Ndfo(self):
        # Compute NDFO indices for all nodes.
        return { node:index for index, node in enumerate(self.ndfo2Node) }

    @cached_property_profiler
    def encodedNodesByNdfo(self):
        tagVocab = self.torchBatch.dataset.fields["src"].vocabs.tags
        encodedNodesData = [tagVocab.stoi[node.tag] for node in self.ndfo2Node]
        return longTensor(encodedNodesData, device=self.device)

    @cached_property_profiler
    def ndfo2Toi(self):
        return [self.node2Toi[node] for node in self.ndfo2Node]

    @cached_property_profiler
    def ndac2Toi(self):
        return [self.ndfo2Toi[ndfo] for ndfo in self.ndac2Ndfo]

    @cached_property_profiler
    def parentSelectorByNdfo(self):
        return longTensor(
            [
                self.node2Ndfo[self.node2Parent[node]] # NDFO index of the parent.
                for node in self.ndfo2Node # Node at NDFO postion in the selector list.
            ],
            device=self.device,
        )

    @cached_property_profiler
    def childSelectorByNdfoList(self):
        # Build NDFO child node selector lists.
        # The below loop only works because all nodes are in decreasing fanout order.
        maxNodeFanout = len(self.ndfo2Node[0])
        retval = [[] for _ in range(maxNodeFanout)]
        for node in self.ndfo2Node:
            for childNumber, childNode in enumerate(node):
                ndfoChildNodeIndex = self.node2Ndfo[childNode]
                retval[childNumber].append(ndfoChildNodeIndex)
        retval.reverse()
        retval = [longTensor(item, device=self.device) for item in retval]
        return retval

    @cached_property_profiler
    def avdl2Toi(self):
        # AVDL to TOI
        return [ self.__adfo2Toi[adfo] for adfo in self.avdl2Adfo ]

    @cached_property_profiler
    def avdlp2Toi(self):
        # AVDLP to TOI
        return [ self.avdl2Toi[avdl] for avdl in self.avdlp2Avdl ]

    @cached_property_profiler
    def attrsByAdfo(self):
        retval = []
        for node in self.ndfo2Node:
            sortedAttribLabels = sorted(node.attrib.keys())
            for attribLabel in sortedAttribLabels:
                retval.append((attribLabel, node.attrib[attribLabel]))
        return retval

    @cached_property_profiler
    def __node2AdfoList(self):
        retval = {}
        adfo = 0
        for node in self.ndfo2Node:
            retval[node] = []
            for _ in node.attrib:
                retval[node].append(adfo)
                adfo += 1
        return retval

    @cached_property_profiler
    def __adfo2Toi(self):
        retval = []
        for node in self.ndfo2Node:
            for _ in node.attrib:
                retval.append(self.node2Toi[node])

        return retval

    @cached_property_profiler
    def avdl2Adfo(self):
        retval = list(range(len(self.attrsByAdfo)))
        retval.sort(key=lambda adfo: (-len(self.attrsByAdfo[adfo][1]), adfo))
        return retval

    @cached_property_profiler
    def __adfo2Avdl(self):
        return invertPermutation(self.avdl2Adfo)

    @cached_property_profiler
    def encodedAttrLabelsByAvdl(self):
        attrsVocab = self.torchBatch.dataset.fields["src"].vocabs.attrs

        return longTensor([
                attrsVocab.stoi[self.attrsByAdfo[adfo][0]]
                for adfo in self.avdl2Adfo
            ],
            device=self.device,
        )
        
    @cached_property_profiler
    def encodedAttrSymbolsByAvdlp(self):
        attrValuesVocab = self.torchBatch.dataset.fields["src"].vocabs.attrValues
        attrValues = [
            longTensor(
                [
                    attrValuesVocab.stoi[ch]
                    for ch in self.attrsByAdfo[adfo][1]
                ],
                device=self.device,
            )
            for adfo in self.avdl2Adfo
            if self.attrsByAdfo[adfo][1]
        ]

        if attrValues:
            return rnn.pack_sequence(attrValues)
        else:
            return longTensor([], device=self.device,)

    @cached_property_profiler
    def node2AvdlList(self):
        return {
            node:[ self.__adfo2Avdl[adfo] for adfo in adfoAttrIndices ]
            for node, adfoAttrIndices in self.__node2AdfoList.items()
        }

    @cached_property_profiler
    def avdl2Ndfo(self):
        node2AvdlList = self.node2AvdlList
        attrCount = len(self.__adfo2Toi)
        retval = [None for _ in range(attrCount)]
        for node, avdlList in node2AvdlList.items():
            for avdl in avdlList:
                retval[avdl] = self.node2Ndfo[node]
        return longTensor(retval, device=self.device)

    @cached_property_profiler
    def avdl2Ndac(self):
        return [self.ndfo2Ndac[int(ndfo)] for ndfo in self.avdl2Ndfo]

    @cached_property_profiler
    def ndac2Ndfo(self):
        retval = list(range(len(self.ndfo2Node)))
        retval.sort(key = lambda ndfoIndex:(-len(self.ndfo2Node[ndfoIndex].attrib), ndfoIndex))
        return retval

    @cached_property_profiler
    def ndfo2Ndac(self):
        return invertPermutation(self.ndac2Ndfo)

    @cached_property_profiler
    def ndac2AvdlList(self):
        return [
            self.node2AvdlList[self.ndfo2Node[ndfoIndex]]
            for ndfoIndex in self.ndac2Ndfo
        ]

    @cached_property_profiler
    def decreasingAttrCountsFactorByNdac(self):
        return torch.tensor([
                float(len(avdlList))
                for avdlList in self.ndac2AvdlList
                if len(avdlList)
            ],
            device=self.device,
        )

    @cached_property_profiler
    def attrValuesByAvdlp(self):
        if self.avdl2Adfo:
            return rnn.pack_sequence([self.attrsByAdfo[adfoIndex][1] for adfoIndex in self.avdl2Adfo])
        else:
            return None

    @cached_property_profiler
    def avdlAttrSelectorsListByNdac(self):
        maxAttrCount = len(self.ndfo2Node[self.ndac2Ndfo[0]].attrib)
        retval = [[] for _ in range(maxAttrCount)]
        for avdlIndices in self.ndac2AvdlList:
            for attrNumber, avdlIndex in enumerate(avdlIndices):
                retval[attrNumber].append(avdlIndex)

        # Reverse, because we want the items to come in increasing order of length.
        retval.reverse()
        retval = [longTensor(item, device=self.device) for item in retval]

        return retval

    @cached_property_profiler
    def ndac2AttrCounts(self):
        return [len(self.ndfo2Node[ndfo].attrib) for ndfo in self.ndac2Ndfo]

    @cached_property_profiler
    def avdlp2Avdl(self):
        avdl2AttrLength = [len(self.attrsByAdfo[adfoIndex][1]) for adfoIndex in self.avdl2Adfo]
        return self.packed2ObjIndices(avdl2AttrLength)

    @cached_property_profiler
    def ndfo2Text2(self):
        retval = [None, None]

        for i, isTail in enumerate([False, True]):
            def getText(node):
                return node.tail if isTail else node.text

            retval[i] = [getText(node) for node in self.ndfo2Node]

    @cached_property_profiler
    def ndtl2Ndfo2(self):
        retval = [None, None]

        for i, isTail in enumerate([False, True]):
            def getLen(node):
                text = (node.tail if isTail else node.text)
                return 0 if text is None else len(text)
            # Get NDTL
            retval[i] = list(range(len(self.ndfo2Node)))
            retval[i].sort(key=lambda ndfoIndex: (-getLen(self.ndfo2Node[ndfoIndex]), ndfoIndex))

        return retval

    @cached_property_profiler
    def ndfo2Ndtl2(self):
        return [
            invertPermutation(self.ndtl2Ndfo2[0]),
            invertPermutation(self.ndtl2Ndfo2[1])
        ]

    @cached_property_profiler
    def ndfo2Ndttl(self):
        return self.ndfo2Ndtl2[0]

    @cached_property_profiler
    def ndfo2Ndtll(self):
        return self.ndfo2Ndtl2[1]

    @cached_property_profiler
    def ndttl2Ndfo(self):
        return invertPermutation(self.ndfo2Ndttl)

    @cached_property_profiler
    def ndttl2Ndac(self):
        return [self.ndfo2Ndac[ndfo] for ndfo in self.ndttl2Ndfo]

    @cached_property_profiler
    def ndtll2Ndttl(self):
        return longTensor([self.ndfo2Ndttl[ndfo] for ndfo in self.ndtll2Ndfo], device=self.device)

    @cached_property_profiler
    def ndtll2Ndfo(self):
        return invertPermutation(self.ndfo2Ndtll)

    @cached_property_profiler
    def ndtl2Node2(self):
        retval = [None, None]
        for i, isTail in enumerate([False, True]):
            retval[i] = [self.ndfo2Node[ndfo] for ndfo in self.ndtl2Ndfo2[isTail]]

        return retval

    @cached_property_profiler
    def encodedTextByNdtlp2(self):
        retval = [None, None]
        textVocab = self.torchBatch.dataset.fields["src"].vocabs.text
        for i, isTail in enumerate([False, True]):
            def getText(node):
                return node.tail if isTail else node.text

            # Get packed text.
            if self.ndtl2Node2[isTail]:
                result = [ getText(node) for node in self.ndtl2Node2[isTail] ]
                result = [
                    longTensor([textVocab.stoi[ch] for ch in text], device=self.device)
                    for text in result
                    if text not in [None, ""]
                ]
                if result:
                    retval[i] = rnn.pack_sequence(result)
                else:
                    retval[i] = None
        return retval

    @cached_property_profiler
    def ndtlp2Ndtl2(self):
        retval = [None, None]
        for i, isTail in enumerate([False, True]):
            def getText(node):
                return node.tail if isTail else node.text

            if self.ndtl2Node2[isTail]:
                decreasingTextLengths = []
                for node in self.ndtl2Node2[isTail]:
                    text = getText(node)
                    if text is None:
                        break
                    decreasingTextLengths.append(len(text))

                retval[i] = self.packed2ObjIndices(decreasingTextLengths)
        return retval

    @cached_property_profiler
    def ndtxSymbolPos2Ndtlp2(self):
        """
        Maps the tuple (ndtx index of node, position index in node.text/node.tail)
        to index in packed ndtlp2 index inside encodedTextByTtDLP/encodedTailByTlDLP.
        """
        retval = [None, None]
        for isTail in [False, True]:
            def getText(node):
                return node.tail if isTail else node.text

            if self.ndtl2Node2[isTail]:
                decreasingTextLengths = []
                for node in self.ndtl2Node2[isTail]:
                    text = getText(node)
                    if text is None:
                        break
                    decreasingTextLengths.append(len(text))

                retval[isTail] = self.tuple2PackedIndex(decreasingTextLengths)
        return retval            

    @cached_property_profiler
    def encodedTextByTtDLP(self):
        return self.encodedTextByNdtlp2[0]

    @cached_property_profiler
    def encodedTailByTlDLP(self):
        return self.encodedTextByNdtlp2[1]

    @cached_property_profiler
    def ndtl2Toi2(self):
        # NDTL to TOI
        retval = [None, None]
        for i, isTail in enumerate([False, True]):
            retval[i] = [ self.node2Toi[self.ndfo2Node[ndfo]] for ndfo in self.ndtl2Ndfo2[i] ]
            
        return retval

    @cached_property_profiler
    def ndtlp2Toi2(self):
        # NDTLP to TOI
        retval = [None, None]
        for i, isTail in enumerate([False, True]):
            retval[i] = [ self.ndtl2Toi2[i][ndtl] for ndtl in self.ndtlp2Ndtl2[isTail] ]

        return retval

    @cached_property_profiler
    def targetOutputsByToi(self):
        return self.outputs[0]

    @cached_property_profiler
    def targetOutputLengthsByToi(self):
        return self.outputs[1]

    @cached_property_profiler
    def tdol2Toi(self):
        retval = list(range(len(self.targetOutputLengthsByToi)))
        retval.sort(key =lambda toi:(-self.targetOutputLengthsByToi[toi], toi))
        return longTensor(retval, device=self.device)

    @cached_property_profiler
    def toi2Tdol(self):
        return longTensor(invertPermutation(self.tdol2Toi.tolist()), device=self.device)

    @cached_property_profiler
    def targetOutputsByTdol(self):
        return self.targetOutputsByToi[self.tdol2Toi]

    @cached_property_profiler
    def targetOutputLengthsByTdol(self):
        return self.targetOutputLengthsByToi[self.tdol2Toi]

    @cached_property_profiler
    def targetOutputsByTdolList(self):
        dimSqueezePoints = computeDimSqueezePoints(self.targetOutputLengthsByTdol)
        targetOutputsByTdol = self.targetOutputsByTdol
        retval = []
        for (outputIndexLimit, sampleIndexLimit) in  dimSqueezePoints:
            while True:
                curIndexLimit = len(retval)
                retval.append(targetOutputsByTdol[0:sampleIndexLimit, curIndexLimit])
                if curIndexLimit+1 == outputIndexLimit:
                    break
        return retval

    @cached_property_profiler
    def ndfo2Gni(self):
        return list(range(len(self.ndfo2Toi)))

    @cached_property_profiler
    def avdl2Gni(self):
        start=len(self.ndfo2Toi)
        end = start + len(self.avdl2Toi)
        return list(range(start, end))

    @cached_property_profiler
    def ndttp2Gni(self):
        start=len(self.ndfo2Toi) + len(self.avdl2Toi)
        end = start + len(self.ndtlp2Toi2[0])
        return list(range(start, end))

    @cached_property_profiler
    def ndtlp2Gni(self):
        start=len(self.ndfo2Toi) + len(self.avdl2Toi) + len(self.ndtlp2Toi2[0])
        end = start + len(self.ndtlp2Toi2[1])
        return list(range(start, end))

    @cached_property_profiler
    def gni2Toi(self):
        retval = [ None for _ in range(self.graphNodeCount) ]
        # ndfo2Toi, avdl2Toi, avdlp2Toi, ndtlp2Toi2[0], ndtlp2Toi2[1].
        for ndfo, toi in enumerate(self.ndfo2Toi):
            retval[self.ndfo2Gni[ndfo]] = toi

        for avdl, toi in enumerate(self.avdl2Toi):
            retval[self.avdl2Gni[avdl]] = toi

        for isTail in [False, True]:
            ndtxp2Gni = self.ndtlp2Gni if isTail else self.ndttp2Gni
            for ndtxp, toi in enumerate(self.ndtlp2Toi2[isTail]):
                retval[ndtxp2Gni[ndtxp]] = toi

        return retval

    @cached_property_profiler
    def gni2Tdol(self):
        return longTensor([ int(self.toi2Tdol[toi]) for toi in self.gni2Toi ], device=self.device)

    @cached_property_profiler
    def graphNodeCount(self):
        return (
            len(self.node2Ndfo) + len(self.avdl2Ndac) +
            len(self.ndtlp2Ndtl2[0]) + len(self.ndtlp2Ndtl2[1])
        )

    @cached_property_profiler
    def posNbrhoodGraphByGni(self):
        """
        Build neighborhoods for:
        1) tagByNdfo
        2) attrVauesByAvdlPacked
        3) attrLabelsByAvdl
        3) textByNdtlp
        4) tailByNdtlp

        """
        # Graph nodes include
        # 1) All XML nodes.
        # 2) All XML attributes.
        # 3) All text positions inside node.text and node.tail.
        nbrHoodGraph = [ [] for _ in range(self.graphNodeCount)]

        # Create node-to-node links first.
        for ndfo, node in enumerate(self.ndfo2Node):
            for childNode in node.iter():
                childNdfo = self.node2Ndfo[childNode]
                nbrHoodGraph[ndfo].append(childNdfo)
                nbrHoodGraph[childNdfo].append(ndfo)

        # Next create node-to-attr links.
        for ndfo, node in enumerate(self.ndfo2Node):
            for attrAvdlIndex in self.node2AvdlList[node]:
                attrAvdlIndexInGraph = self.avdl2Gni[attrAvdlIndex]
                nbrHoodGraph[ndfo].append(attrAvdlIndexInGraph)
                nbrHoodGraph[attrAvdlIndexInGraph].append(ndfo)
        
        for isTail in [False, True]:
            # Next create node-to-text and node-to-tail links.
            i = int(isTail)
            lastNdtlp = {}
            for ndtlp, ndtl in enumerate(self.ndtlp2Ndtl2[i]):
                ndtxp2Gni = self.ndtlp2Gni if isTail else self.ndttp2Gni
                ndtx2Ndfo = self.ndtl2Ndfo2[i]
                ndfoIndexInGraph = self.ndfo2Gni[ndtx2Ndfo[ndtl]]
                ndtlpIndexInGraph = ndtxp2Gni[ndtlp]

                # Create graph link.
                nbrHoodGraph[ndfoIndexInGraph].append(ndtlpIndexInGraph)
                nbrHoodGraph[ndtlpIndexInGraph].append(ndfoIndexInGraph)

                if ndtl in lastNdtlp:
                    # Create graph link within neighboring positions in text.
                    ourNdtlpNeighbor = lastNdtlp[ndtl]
                    nbrHoodGraph[ourNdtlpNeighbor].append(ndtlpIndexInGraph)
                    nbrHoodGraph[ndtlpIndexInGraph].append(ourNdtlpNeighbor)

                # Set position seen in same text for next time.
                lastNdtlp[ndtl] = ndtlpIndexInGraph

        # Get all tail/text neighbor pairs in pre-order traversal for tail-to-text.
        tailTextNeighborPairs = sum(
            [
                self.__tailTextNeighborPairs(xmlTree)
                for xmlTree in self.torchBatch.src
            ],
            []
        )

        # Finally create tail/text neighbor links.
        for nbr1, nbr2 in tailTextNeighborPairs:
            nbrHoodGraph[nbr1].append(nbr2)
            nbrHoodGraph[nbr2].append(nbr1)

        for i, adjList in enumerate(nbrHoodGraph):
            nbrHoodGraph[i] = sorted(set(adjList))
        return nbrHoodGraph

    @cached_property_profiler
    def gndtol2Gni(self):
        """
        Mapping of GNTDOL indices to GNI indices.
        """
        retval = list(range(self.graphNodeCount))
        return sorted(retval, key=lambda gni:(int(self.gni2Tdol[gni]), gni))

    @cached_property_profiler
    def gndtol2Toi(self):
        """
        Mapping of GNDTOL indices to TOI tree indices.
        """
        return [self.gni2Toi[gni] for gni in self.gndtol2Gni]

    @cached_property_profiler
    def gndtol2Tdol(self):
        return self.gni2Tdol[self.gndtol2Gni]

    @cached_property_profiler
    def gni2Gndtol(self):
        """
        Mapping of GNI indices to GNTDOL indices.
        """
        return invertPermutation(self.gndtol2Gni)

    @cached_property_profiler
    def posNbrhoodGraphByGndtol(self):
        """
        Mapping of GNTDOL indices to GNI indices.
        """
        adjListTensor = [
            longTensor(
                sorted([
                    self.gni2Gndtol[nbrGni]
                    for nbrGni in self.posNbrhoodGraphByGni[self.gndtol2Gni[gndtol]]
                ]),
                device=self.device
            )
            for gndtol in range(self.graphNodeCount)
        ]
        adjLengthsTensor = longTensor([len(adjList) for adjList in adjListTensor], device=self.device)

        adjListTensor = rnn.pad_sequence(adjListTensor, batch_first=True)

        return (adjListTensor, adjLengthsTensor)

    @cached_property_profiler
    def fullSpotlight(self):
        return longTensor(list(range(self.graphNodeCount)), device=self.device)

    @staticmethod
    def tuple2PackedIndex(decreasingObjLengths):
        # Build tuple2PackedIndex.
        packedIndex = 0
        objCount = len(decreasingObjLengths)
        if not objCount:
            return []
        maxObjLength = decreasingObjLengths[0]

        tuple2PackedIndex = {}
        for indexWithinObj in range(maxObjLength):
            for objIndex in range(objCount):
                if indexWithinObj >= decreasingObjLengths[objIndex]:
                    # This tree doesn't have any more nodes.
                    break
                tuple2PackedIndex[(objIndex, indexWithinObj)] = packedIndex
                packedIndex += 1

        return tuple2PackedIndex

    @staticmethod
    def linear2PackedIndex(decreasingObjLengths):
        # Build linear2PackedIndex.
        packedIndex = 0
        objCount = len(decreasingObjLengths)
        if not objCount:
            return []
        packedLength = sum(decreasingObjLengths)
        maxObjLength = decreasingObjLengths[0]
        linear2PackedIndex = [
            [
                None
                for objLength in range(decreasingObjLengths[objIndex])
            ]
            for objIndex in range(objCount)
        ]
        for indexWithinObj in range(maxObjLength):
            for objIndex in range(objCount):
                if indexWithinObj >= decreasingObjLengths[objIndex]:
                    # This tree doesn't have any more nodes.
                    break
                linear2PackedIndex[objIndex][indexWithinObj] = packedIndex
                packedIndex += 1

        # Append all.
        linear2PackedIndex = sum(linear2PackedIndex, [])

        return linear2PackedIndex

    @staticmethod
    def packed2ObjIndices(decreasingObjLengths):
        # Build linear2PackedIndex.
        packedIndex = 0
        objCount = len(decreasingObjLengths)
        if not objCount:
            return []
        packedLength = sum(decreasingObjLengths)
        maxObjLength = decreasingObjLengths[0]
        packed2ObjIndices = [None for _ in range(packedLength)]
        for indexWithinObj in range(maxObjLength):
            for objIndex in range(objCount):
                if indexWithinObj >= decreasingObjLengths[objIndex]:
                    # This tree doesn't have any more nodes.
                    break
                packed2ObjIndices[packedIndex] = objIndex
                packedIndex += 1

        return packed2ObjIndices

    @methodProfiler
    def __tailTextNeighborPairs(self, xmlTree):
        def textNtailTraversal(node, tailTextOrdering):
            # First, we have node.text.
            if node.text not in [None, ""]:
                tailTextOrdering.append((node, False))

            # Then, come all children.
            for childNode in node:
                textNtailTraversal(childNode, tailTextOrdering)

            # Finally, we have node.tail.
            if node.tail not in [None, ""]:
                tailTextOrdering.append((node, True))

        # Build tail text ordering.
        tailTextOrdering = []
        textNtailTraversal(xmlTree.getroot(), tailTextOrdering)

        def getGraphIndex(arg):
            node, isTail = arg
            ndfoIndex = self.node2Ndfo[node]
            ndtlIndex = self.ndfo2Ndtl2[isTail][ndfoIndex]
            ndtx2Gni = self.ndtlp2Gni if isTail else self.ndttp2Gni
            graphIndex = ndtx2Gni[ndtlIndex]
            return graphIndex

        retval = []
        if tailTextOrdering:
            lastBlock = tailTextOrdering[0]
            for curBlock in tailTextOrdering[1:]:
                retval.append((getGraphIndex(lastBlock), getGraphIndex(curBlock)))
                lastBlock = curBlock
        return retval

def splitByToi(allVecs, vecIndex2Toi, sampleCount, prefix=""):
    frame = inspect.stack()[1]
    name = "{0}{1}:{2}".format(prefix, basename(frame.filename), frame.lineno)
    retval = [[] for _ in range(sampleCount)]
    for index, vec in enumerate(allVecs):
        retval[vecIndex2Toi[index]].append(vec)
    retval = [
        [
            vec.view([1] + list(vec.shape)) # Reshape for better catting.
            for vec in vecList
        ]
        for vecList in retval
    ]
    retval = [
        torch.cat(vecList) if vecList else torch.Tensor([], device=self.device)
        for vecList in retval
    ]

    return name, retval

def computeDimSqueezePoints(outputLimitsInOrder):
    """
    Compute the positions in output symbol computation, where we exclude another batch of trees from
    further consideration. We do that because the excluded output trees have their symbol computation already
    completed and need no more computation. Only used during training, when target output lengths are available.

    Input:
        outputLimitsInOrder: Length of target outputs in decreasing order.

    Output:
        dimSqueezePoints: List of tuples (outputIndexLimit, sampleIndexLimit)
            [(oil1, sil1), (oil2, sil2), (oil3, sil3), (oil4, sil4), ]
            For output indices [0, 1, ..., oil1-1] we use sampleIndexLimit as sil1.
            For output indices [oil1, oil1+1, ..., oil2-1] we use sampleIndexLimit as sil2.
            For output indices [oil2, oil2+1, ..., oil3-1] we use sampleIndexLimit as sil3.
            .
            .
            For output indices [oil2ndLast, oil2ndLast+1, ..., oilLast-1] we use sampleIndexLimit as silLast.

    """
    dimSqueezePoints = []
    outputLimitsInOrder = [ int(outputLimit) for outputLimit in outputLimitsInOrder ]
    sampleCount = len(outputLimitsInOrder)
    curOutputLimit = outputLimitsInOrder[-1]

    dimSqueezePoints.append((curOutputLimit, sampleCount))

    for sampleLimit, outputLimit in enumerate(outputLimitsInOrder[::-1]):
        if outputLimit == curOutputLimit:
            continue

        curOutputLimit = outputLimit
        dimSqueezePoints.append((curOutputLimit, sampleCount - sampleLimit))

    return dimSqueezePoints


def batchDataUnitTest(trainer, test_data):
    sampleCount = len(test_data)
    # Create data batch.
    trainer.load(test_data)
    batch_iterator = Hier2HierIterator(
        preprocess_batch=trainer.model.preprocess_batch,
        dataset=test_data, batch_size=len(test_data),
        sort=False, shuffle=True, sort_within_batch=False,
        sort_key=lambda x: len(x.tgt),
        repeat=False)
    batch_generator = batch_iterator.__iter__(mode=AppMode.Test)
    test_data_batch = list(batch_generator)[0]

    # Vocabs.
    tagVocab = test_data.fields["src"].vocabs.tags
    attrsVocab = test_data.fields["src"].vocabs.attrs
    attrValuesVocab = test_data.fields["src"].vocabs.attrValues
    textVocab = test_data.fields["src"].vocabs.text
    outputVocab = test_data.fields["tgt"].vocab

    # Check encodedNodesByNdfo
    allNodes = sum([list(tree.iter()) for tree in test_data_batch.inputs], [])
    allNodesInNdfo = test_data_batch.ndfo2Node
    lastFanout = 100000
    for node in allNodesInNdfo:
        # NDFO nodes are not in decreasing fanout order.
        assert(len(node) <= lastFanout)
        lastFanout = len(node)
    assert(set(allNodesInNdfo) == set(allNodes))
    assert(len(allNodesInNdfo) == test_data_batch.encodedNodesByNdfo.shape[0])
    for i in range(test_data_batch.encodedNodesByNdfo.shape[0]):
        assert(tagVocab.stoi[allNodesInNdfo[i].tag] == int(test_data_batch.encodedNodesByNdfo[i]))

    # Check parentSelectorByNdfo.
    node2ndfo = {node:index for index, node in enumerate(allNodesInNdfo)}
    node2parent = {}
    for ndfo, node in enumerate(allNodesInNdfo):
        for child in node:
            node2parent[child] = node
    for ndfo, node in enumerate(allNodesInNdfo):
        for child in node:
            childNdfo = node2ndfo[child]
            parentNdfo = test_data_batch.parentSelectorByNdfo[childNdfo]
            assert(parentNdfo == ndfo)
    allRoots = [tree.getroot() for tree in test_data_batch.inputs]
    for root in allRoots:
        rootNdfo = node2ndfo[root]
        rootParentNdfo = test_data_batch.parentSelectorByNdfo[rootNdfo]
        assert(rootNdfo == rootParentNdfo)
    assert(sampleCount + len(node2parent) == len(allNodes))
    
    # Check childSelectorByNdfoList and decreasingFanoutsFactorByNdfo.
    lastLen = 0
    totalChildren = sum([len(childList) for childList in test_data_batch.childSelectorByNdfoList])
    assert(totalChildren == len(node2parent))
    for childSelectorByNdfo in test_data_batch.childSelectorByNdfoList:
        curLen = len(childSelectorByNdfo)
        assert(curLen != 0)
        assert(lastLen <= curLen)
        lastLen = curLen
    for ndfo, node in enumerate(allNodesInNdfo):
        curChildList = []
        for childSelectorByNdfo in test_data_batch.childSelectorByNdfoList:
            if ndfo < len(childSelectorByNdfo):
                curChildNdfo = childSelectorByNdfo[ndfo]
                curChild = allNodesInNdfo[curChildNdfo]
                curChildList.append(curChild)
        curChildList.reverse()
        for childIndex, childNode in enumerate(node):
            assert(curChildList[childIndex] == childNode)
        assert(len(curChildList) == len(node))
        if len(node):
            assert(float(len(curChildList)) == float(test_data_batch.decreasingFanoutsFactorByNdfo[ndfo]))
        else:
            assert(ndfo >= test_data_batch.decreasingFanoutsFactorByNdfo.shape[0])

    # Check encodedAttrLabelsByAvdl.
    allAttrs = sum([list(node.attrib.items()) for node in allNodes], [])
    assert(len(allAttrs) == len(test_data_batch.attrsByAdfo))
    allAttrsInAvdl = [test_data_batch.attrsByAdfo[adfo] for adfo in test_data_batch.avdl2Adfo]
    assert(len(allAttrsInAvdl) == len(test_data_batch.encodedAttrLabelsByAvdl))
    for avdl, attrLabelCode in enumerate(test_data_batch.encodedAttrLabelsByAvdl):
        assert(int(attrLabelCode) == attrsVocab.stoi[allAttrsInAvdl[avdl][0]])

    # Check encodedAttrSymbolsByAvdlp.
    attrValues, attrValueLengths = rnn.pad_packed_sequence(
        test_data_batch.encodedAttrSymbolsByAvdlp,
        batch_first=True)
    assert(len(attrValues) <= len(allAttrsInAvdl))
    for i in range(len(attrValues), len(allAttrsInAvdl)):
        assert(allAttrsInAvdl[i][1] == "")
    for avdl, (attr, attrValue) in enumerate(allAttrsInAvdl):
        if avdl < attrValueLengths.shape[0]:
            assert(len(attrValue) == int(attrValueLengths[avdl]))
        for i, attrSymbol in enumerate(attrValue):
            assert(int(attrValues[avdl, i]) == attrValuesVocab.stoi[attrSymbol])

    # Check ndac2Ndfo.
    assert(
        set([test_data_batch.ndfo2Node[ndfo ]for ndfo in test_data_batch.ndac2Ndfo])
        == set(allNodes)
    )
    lastAttrCount = 1000000
    for ndac, ndfo in enumerate(test_data_batch.ndac2Ndfo):
        ndfo = int(ndfo)
        curAttrCount = len(allNodesInNdfo[ndfo].attrib)
        assert(curAttrCount <= lastAttrCount)
        lastAttrCount = curAttrCount

    # Check avdl2ndac
    allNodesInNdac = [test_data_batch.ndfo2Node[ndfo] for ndfo in test_data_batch.ndac2Ndfo]
    for avdl, ndac in enumerate(test_data_batch.avdl2Ndac):
        curAttribs = allNodesInNdac[ndac].attrib
        attrLabel, attrValue = allAttrsInAvdl[avdl]
        assert(curAttribs[attrLabel] == attrValue)

    # Check avdl2ndfo
    for avdl, ndfo in enumerate(test_data_batch.avdl2Ndfo):
        curAttribs = allNodesInNdfo[ndfo].attrib
        attrLabel, attrValue = allAttrsInAvdl[avdl]
        assert(curAttribs[attrLabel] == attrValue)

    # Check avdlAttrSelectorsListByNdac and decreasingAttrCountsFactorByNdac.
    lastLen = 0
    for avdlAttrSelectorByNdac in test_data_batch.avdlAttrSelectorsListByNdac:
        curLen = len(avdlAttrSelectorByNdac)
        assert(curLen != 0)
        assert(lastLen <= curLen)
        lastLen = curLen
    for ndac, ndfo in enumerate(test_data_batch.ndac2Ndfo):
        node = test_data_batch.ndfo2Node[ndfo]
        curAttrList = []
        attrCountFound = 0
        for avdlAttrSelectorByNdac in test_data_batch.avdlAttrSelectorsListByNdac:
            if ndac < len(avdlAttrSelectorByNdac):
                curAttrAvdl = avdlAttrSelectorByNdac[ndac]
                attrLabel, attrValue = allAttrsInAvdl[curAttrAvdl]
                assert(node.attrib[attrLabel] == attrValue)
                attrCountFound += 1
        assert(len(node.attrib) == attrCountFound)
        if attrCountFound:
            assert(float(attrCountFound) == float(test_data_batch.decreasingAttrCountsFactorByNdac[ndac]))
        else:
            assert(ndac >= test_data_batch.decreasingAttrCountsFactorByNdac.shape[0])

    # Check encodedTextByNdtlp2(i.e. encodedTextByTtDLP and encodedTailByTlDLP)
    for isTail in [False, True]:
        def getText(node):
            return node.tail if isTail else node.text
        encodedTextByDTL = test_data_batch.encodedTextByNdtlp2[isTail]
        if encodedTextByDTL is None:
            assert(all([getText(node) in [None, ""] for node in allNodes]))
        else:
            # Length order is already enforced.
            encodedText, textLengths = rnn.pad_packed_sequence(encodedTextByDTL, batch_first=True)

            ndfo2Ndtl = test_data_batch.ndfo2Ndtl2[isTail]

            assert(len(ndfo2Ndtl) == len(allNodes))
            for ndfo, node in enumerate(allNodesInNdfo):
                ndtl = ndfo2Ndtl[ndfo]
                text = getText(node)
                if text in [None, ""]:
                    assert(ndtl >= len(textLengths))
                else:
                    assert(len(text) == textLengths[ndtl])
                    for i, ch in enumerate(text):
                       assert(int(encodedText[ndtl][i]) == textVocab.stoi[ch])

    # ndttl2Ndac, ndtll2ndt  and ndtll2Ndac.
    for ndfo, ndtll in enumerate(test_data_batch.ndfo2Ndtll):
        ndttl = test_data_batch.ndtll2Ndttl[ndtll]
        ndac = test_data_batch.ndttl2Ndac[ndttl]
        assert(ndfo == test_data_batch.ndac2Ndfo[ndac])

    # targetOutputsByTdol and targetOutputLengthsByTdol.
    assert(len(test_data_batch.targetOutputsByTdol) == sampleCount)
    assert(len(test_data_batch.targetOutputLengthsByTdol) == sampleCount)
    targetOutputByToi = test_data_batch.targetOutputsByToi
    targetOutputLengthsByToi = test_data_batch.targetOutputLengthsByToi
    for toi, targetOutput in enumerate(targetOutputByToi):
        length = targetOutputLengthsByToi[toi]
        tdol = int(test_data_batch.toi2Tdol[toi])
        assert(length == test_data_batch.targetOutputLengthsByTdol[tdol])
        for i in range(length):
            assert(targetOutput[i] == test_data_batch.targetOutputsByTdol[tdol][i])

    for i in range(sampleCount-1):
        assert(test_data_batch.targetOutputLengthsByTdol[i] >= test_data_batch.targetOutputLengthsByTdol[i+1])

    # gndtol2Gni and gndtol2Tdol
    lastTdol = -1
    for gndtol, gni in enumerate(test_data_batch.gndtol2Gni):
        assert(test_data_batch.gndtol2Tdol[gndtol] == test_data_batch.gni2Tdol[gni])
        assert(test_data_batch.gndtol2Tdol[gndtol] >= lastTdol)
        lastTdol = test_data_batch.gndtol2Tdol[gndtol]
