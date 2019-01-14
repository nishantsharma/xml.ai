"""
    To efficiently process a batch of XML trees in GPU, we need to permute
    xmlTrees, their nodes, their attributes and text fields in various orders.
    
    Hier2hierBatch implements all those permutations and mappings during batch
    pre-processing.
"""
from cached_property import cached_property
from attrdict import AttrDict
import torch
import torch.nn.utils.rnn as rnn

from hier2hier.util import invertPermutation

class Hier2hierBatch(object):
    def __init__(self, torchBatch):
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
                    DTtL: By decreasing text length.
                For Tail:
                    DTlL: By decreasing tail length.
        """
        self.torchBatch = torchBatch

    @cached_property
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

    @cached_property
    def node2Toi(self):
        # Build node2Toi(original tree index) dictionary.
        retval =  { }
        for toi, xmlTree in enumerate(self.torchBatch.src):
            for node in xmlTree.iter():
                retval[node] = toi
        return retval

    @cached_property
    def tdnc2Toi(self):
        # Compute TDNC permutation for trees.
        retval = list(range(len(self.torchBatch.src)))
        retval.sort(lambda index: -len(self.torchBatch.src[index]))
        return retval

    @cached_property
    def toi2Tdnc(self):
        # Invert TDNC permutation for trees.
        return invertPermutation(self.tdnc2Toi)

    @cached_property
    def ndfo2Node(self):
        # Permute nodes into NDFO order.
        retval = sum([ list(xmlTree.iter()) for xmlTree in self.torchBatch.src ], [])
        retval.sort(key=lambda node:-len(node)) # Fanout = len(node)
        return retval

    @cached_property
    def decreasingFanoutsFactorByNdfo(self):
        return torch.tensor([float(len(node)) for node in self.ndfo2Node if len(node)])

    @cached_property
    def node2Ndfo(self):
        # Compute NDFO indices for all nodes.
        return { node:index for index, node in enumerate(self.ndfo2Node) }

    @cached_property
    def encodedNodesByNdfo(self):
        tagVocab = self.torchBatch.dataset.fields["src"].vocabs.tags
        return torch.LongTensor([tagVocab.stoi[node.tag] for node in self.ndfo2Node])

    @cached_property
    def ndfo2Toi(self):
        return [self.node2Toi[node] for node in self.ndfo2Node]

    @cached_property
    def parentSelectorByNdfo(self):
        return [
            self.node2Ndfo[self.node2Parent[node]] # NDFO index of the parent.
            for node in self.ndfo2Node # Node at NDFO postion in the selector list.
        ]

    @cached_property
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

        return retval

    @cached_property
    def attrsByAdfo(self):
        retval = []
        for node in self.ndfo2Node:
            for attribLabel, attribValue in node.attrib.items():
                retval.append((attribLabel, attribValue))
        return retval

    @cached_property
    def avdl2Toi(self):
        # AVDL to TOI
        return [ self.adfo2Toi[adfo] for adfo in self.avdl2Adfo ]

    @cached_property
    def avdlp2Toi(self):
        # AVDLP to TOI
        return [ self.avdl2Toi[avdl] for avdl in self.avdlp2Avdl ]

    @cached_property
    def node2AdfoList(self):
        retval = {}
        adfo = 0
        for node in self.ndfo2Node:
            retval[node] = []
            for _ in node.attrib:
                retval[node].append(adfo)
                adfo += 1
        return retval

    @cached_property
    def adfo2Toi(self):
        adfo = 0
        retval = []
        for node in self.ndfo2Node:
            for _ in node.attrib:
                retval.append(self.node2Toi[node])
                adfo += 1

        return retval

    @cached_property
    def avdl2Adfo(self):
        retval = list(range(len(self.attrsByAdfo)))
        retval.sort(key=lambda origIndex: -len(self.attrsByAdfo[origIndex][1]))
        return retval

    @cached_property
    def adfo2Avdl(self):
        return invertPermutation(self.avdl2Adfo)

    @cached_property
    def encodedAttrLabelsByAvdl(self):
        attrsVocab = self.torchBatch.dataset.fields["src"].vocabs.attrs

        return torch.LongTensor([
            attrsVocab.stoi[self.attrsByAdfo[adfo][0]]
            for adfo in self.avdl2Adfo
        ])
        
    @cached_property
    def encodedAttrSymbolsByAvdlp(self):
        attrValuesVocab = self.torchBatch.dataset.fields["src"].vocabs.attrValues
        attrValues = [
            torch.LongTensor([
                attrValuesVocab.stoi[ch]
                for ch in self.attrsByAdfo[adfo][1]
            ])
            for adfo in self.avdl2Adfo
        ]

        if attrValues:
            return rnn.pack_sequence(attrValues)
        else:
            return None

    @cached_property
    def node2AvdlList(self):
        return {
            node:[ self.adfo2Avdl[adfoAttrIndex] for adfoAttrIndex in adfoAttrIndices ]
            for node, adfoAttrIndices in self.node2AdfoList.items()
        }

    @cached_property
    def avdl2Ndfo(self):
        node2AvdlList = self.node2AvdlList
        attrCount = len(self.adfo2Toi)
        retval = [None for _ in range(attrCount)]
        for node, avdlList in node2AvdlList.items():
            for avdl in avdlList:
                retval[avdl] = self.node2Ndfo[node]
        return retval

    @cached_property
    def avdl2Ndac(self):
        return [self.ndfo2Ndac[ndfo] for ndfo in self.avdl2Ndfo]

    @cached_property
    def ndac2Ndfo(self):
        retval = list(range(len(self.ndfo2Node)))
        retval.sort(key = lambda ndfoIndex:-len(self.ndfo2Node[ndfoIndex]))
        return retval

    @cached_property
    def ndfo2Ndac(self):
        return invertPermutation(self.ndac2Ndfo)

    @cached_property
    def ndac2AvdlList(self):
        return [
            self.node2AvdlList[self.ndfo2Node[ndfoIndex]]
            for ndfoIndex in self.ndac2Ndfo
        ]

    @cached_property
    def decreasingAttrCountsFactorByNdac(self):
        return torch.tensor([
            float(len(avdlList))
            for avdlList in self.ndac2AvdlList
            if len(avdlList)
        ])

    @cached_property
    def attrValuesByAvdlp(self):
        if self.avdl2Adfo:
            return rnn.pack_sequence([self.attrsByAdfo[adfoIndex][1] for adfoIndex in self.avdl2Adfo])
        else:
            return None

    @cached_property
    def avdlAttrSelectorsListByNdac(self):
        maxAttrCount = len(self.ndfo2Node[self.ndac2Ndfo[0]].attrib)
        retval = [[] for _ in range(maxAttrCount)]
        for avdlIndices in self.ndac2AvdlList:
            for attrNumber, avdlIndex in enumerate(avdlIndices):
                retval[attrNumber].append(avdlIndex)
        retval.reverse()

        return retval

    @cached_property
    def ndac2AttrCounts(self):
        return [len(self.ndfo2Node[ndfo].attrib) for ndfo in self.ndac2Ndfo]

    @cached_property
    def avdlp2Avdl(self):
        return self.__packed2ObjIndices(self.ndac2AttrCounts)

    @cached_property
    def ndfo2Text2(self):
        retval = [None, None]

        for i, isTail in enumerate([False, True]):
            def getText(node):
                return node.tail if isTail else node.text

            retval[i] = [getText(node) for node in self.ndfo2Node]

    @cached_property
    def ndtl2Ndfo2(self):
        retval = [None, None]

        for i, isTail in enumerate([False, True]):
            def getLen(node):
                text = (node.tail if isTail else node.text)
                return 0 if text is None else len(text)
            # Get NDTL
            retval[i] = list(range(len(self.ndfo2Node)))
            retval[i].sort(key=lambda ndfoIndex: -getLen(self.ndfo2Node[ndfoIndex]))    

        return retval

    @cached_property
    def ndfo2Ndtl2(self):
        return [
            invertPermutation(self.ndtl2Ndfo2[0]),
            invertPermutation(self.ndtl2Ndfo2[1])
        ]

    @cached_property
    def ndfo2Ndttl(self):
        return self.ndfo2Ndtl2[0]

    @cached_property
    def ndfo2Ndtll(self):
        return self.ndfo2Ndtl2[1]

    @cached_property
    def ndttl2Ndfo(self):
        return invertPermutation(self.ndfo2Ndttl)

    @cached_property
    def ndttl2Ndac(self):
        return [self.ndfo2Ndac[ndfo] for ndfo in self.ndttl2Ndfo]

    @cached_property
    def ndtll2Ndttl(self):
        return torch.LongTensor([self.ndfo2Ndttl[ndfo] for ndfo in self.ndtll2Ndfo])

    @cached_property
    def ndtll2Ndfo(self):
        return invertPermutation(self.ndfo2Ndtll)

    @cached_property
    def ndttl2Ndac(self):
        return [self.ndfo2Ndac[ndfo] for ndfo in self.ndttl2Ndfo]

    @cached_property
    def ndtl2Node2(self):
        retval = [None, None]
        for i, isTail in enumerate([False, True]):
            retval[i] = [self.ndfo2Node[ndfo] for ndfo in self.ndtl2Ndfo2[isTail]]

        return retval

    @cached_property
    def encodedTextByNdtlp2(self):
        retval = [None, None]
        textVocab = self.torchBatch.dataset.fields["src"].vocabs.text
        for i, isTail in enumerate([False, True]):
            def getText(node):
                return node.tail if isTail else node.text

            # Get packed text.
            if self.ndtl2Node2[isTail]:
                retval[i] = [ getText(node) for node in self.ndtl2Node2[isTail] ]
                retval[i] = [ text for text in retval[i] if text is not None ]
                retval[i] = [
                    torch.LongTensor([textVocab.stoi[ch] for ch in text])
                    for text in retval[i]
                ]
                retval[i] = rnn.pack_sequence(retval[i])
        return retval

    @cached_property
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

                retval[i] = self.__packed2ObjIndices(decreasingTextLengths)
        return retval

    @cached_property
    def encodedTextByDTtL(self):
        return self.encodedTextByNdtlp2[0]

    @cached_property
    def encodedTailByDTlL(self):
        return self.encodedTextByNdtlp2[1]

    @cached_property
    def ndtl2Toi2(self):
        # NDTL to TOI
        retval = [None, None]
        for i, isTail in enumerate([False, True]):
            retval[i] = [ self.node2Toi[self.ndfo2Node[ndfo]] for ndfo in self.ndtl2Ndfo2[i] ]
            
        return retval

    @cached_property
    def ndtlp2Toi2(self):
        # NDTLP to TOI
        retval = [None, None]
        for i, isTail in enumerate([False, True]):
            retval[i] = [ self.ndtl2Toi2[i][ndtl] for ndtl in self.ndtlp2Ndtl2[isTail] ]

        return retval

    @cached_property
    def targetOutputsByToi(self):
        return self.torchBatch.tgt[0]

    @cached_property
    def targetOutputLengthsByToi(self):
        return self.torchBatch.tgt[1]

    @cached_property
    def tdol2Toi(self):
        retval = list(range(len(self.targetOutputLengthsByToi)))
        retval.sort(key =lambda toi:-self.targetOutputLengthsByToi[toi])
        return torch.LongTensor(retval)

    @cached_property
    def toi2Tdol(self):
        return torch.LongTensor(invertPermutation(self.tdol2Toi.tolist()))

    @cached_property
    def targetOutputsByTdol(self):
        return self.targetOutputsByToi[self.tdol2Toi]

    @cached_property
    def targetOutputLengthsByTdol(self):
        return self.targetOutputLengthsByToi[self.tdol2Toi]

    @cached_property
    def graphIndexOffsets(self):
        return AttrDict({
            "ndfo": 0,
            "avdl": len(self.ndfo2Toi),
            "ndttp": len(self.ndfo2Toi) + len(self.avdl2Toi),
            "ndtlp": len(self.ndfo2Toi) + len(self.avdl2Toi) + len(self.ndtlp2Toi2[0]),
            "end": len(self.ndfo2Toi) + len(self.avdl2Toi) + len(self.ndtlp2Toi2[0]) + len(self.ndtlp2Toi2[1]),
        })

    @cached_property
    def gni2Toi(self):
        retval = [ None for _ in range(self.graphIndexOffsets.end) ]
        # ndfo2Toi, avdl2Toi, avdlp2Toi, ndtlp2Toi2[0], ndtlp2Toi2[1].
        for ndfo, toi in enumerate(self.ndfo2Toi):
            retval[self.graphIndexOffsets.ndfo + ndfo] = toi

        for avdl, toi in enumerate(self.avdl2Toi):
            retval[self.graphIndexOffsets.avdl + avdl] = toi

        for isTail in [False, True]:
            offset = self.graphIndexOffsets.ndtlp if isTail else self.graphIndexOffsets.ndttp
            for ndtlp, toi in enumerate(self.ndtlp2Toi2[isTail]):
                retval[offset + ndtlp] = toi

        return retval

    @cached_property
    def gni2Tdol(self):
        return torch.LongTensor([ self.toi2Tdol[toi] for toi in self.gni2Toi ])

    @cached_property
    def attnReadyPosNbrhoodGraph(self):
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
        graphNodeCount = (
            len(self.node2Ndfo) + len(self.avdl2Ndac) +
            len(self.ndtlp2Ndtl2[0]) + len(self.ndtlp2Ndtl2[1])
        )
        nbrHoodGraph = [ [] for _ in range(graphNodeCount)]

        # Create node-to-node links first.
        for ndfo, node in enumerate(self.ndfo2Node):
            for childNode in node.iter():
                childNdfo = self.node2Ndfo[childNode]
                nbrHoodGraph[ndfo].append(childNdfo)
                nbrHoodGraph[childNdfo].append(ndfo)

        # Next create node-to-attr links.
        for ndfo, node in enumerate(self.ndfo2Node):
            for attrAvdlIndex in self.node2AvdlList[node]:
                attrAvdlIndexInGraph = self.graphIndexOffsets.avdl + attrAvdlIndex
                nbrHoodGraph[ndfo].append(attrAvdlIndexInGraph)
                nbrHoodGraph[attrAvdlIndexInGraph].append(ndfo)
        
        for isTail in [False, True]:
            # Next create node-to-text and node-to-tail links.
            i = int(isTail)
            lastNdtlp = {}
            for ndtlp, ndtl in enumerate(self.ndtlp2Ndtl2[i]):
                if isTail:
                    indexInGaphOffset = self.graphIndexOffsets.ndtlp
                else:
                    indexInGaphOffset = self.graphIndexOffsets.ndttp
                ndfoIndexInGraph = self.ndtl2Ndfo2[i][ndtl]
                ndtlpIndexInGraph = ndtlp + indexInGaphOffset

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

        return nbrHoodGraph

    @staticmethod
    def __packed2ObjIndices(decreasingObjLengths):
        # Build linear2PackedIndex.
        packedIndex = 0
        objCount = len(decreasingObjLengths)
        packedLength = sum(decreasingObjLengths)
        maxObjLength = decreasingObjLengths[0]
        # linear2PackedIndex = [
        #     [
        #         None
        #         for objLength in range(decreasingObjLengths[objIndex])
        #     ]
        #     for objIndex in range(objCount)
        # ]
        # tuple2PackedIndex = {}
        packed2ObjIndices = [None for _ in range(packedLength)]
        for indexWithinObj in range(maxObjLength):
            for objIndex in range(objCount):
                if indexWithinObj >= decreasingObjLengths[objIndex]:
                    # This tree doesn't have any more nodes.
                    break
                # tuple2PackedIndex[(objIndex, indexWithinObj)] = packedIndex
                # linear2PackedIndex[objIndex][indexWithinObj] = packedIndex
                packed2ObjIndices[packedIndex] = objIndex
                packedIndex += 1

        # Append all.
        # linear2PackedIndex = sum(linear2PackedIndex, [])

        return packed2ObjIndices

    def __tailTextNeighborPairs(self, xmlTree):
        def textNtailTraversal(node, tailTextOrdering):
            # First, we have node.text.
            if node.text is not None:
                tailTextOrdering.append((node, False))

            # Then, come all children.
            for childNode in node:
                textNtailTraversal(childNode, tailTextOrdering)

            # Finally, we have node.tail.
            if node.tail is not None:
                tailTextOrdering.append((node, True))

        # Build tail text ordering.
        tailTextOrdering = []
        textNtailTraversal(xmlTree.getroot(), tailTextOrdering)

        def getGraphIndex(arg):
            node, isTail = arg
            ndfoIndex = self.node2Ndfo[node]
            ndtlIndex = self.ndfo2Ndtl2[isTail][ndfoIndex]
            if isTail:
                graphIndex = ndtlIndex + self.graphIndexOffsets.ndtlp
            else:
                graphIndex = ndtlIndex + self.graphIndexOffsets.ndttp
            return graphIndex

        retval = []
        lastBlock = tailTextOrdering[0]
        for curBlock in tailTextOrdering[1:]:
            retval.append((getGraphIndex(lastBlock), getGraphIndex(lastBlock)))
        return retval
