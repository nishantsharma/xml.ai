import logging, copy

from orderedattrdict import AttrDict
from collections import Counter

import torchtext

import xml.etree.ElementTree as ET
SYM_SOS = '<sos>'
SYM_EOS = '<eos>'
SYM_PAD = '<pad>'

def buildVocabs(hier_dataset, max_size=None):
    # Collect all symbols.
    symbolCounters = AttrDict({"src": None, "tgt": None})

    resultVocabs = AttrDict({})
    for srcOrTgt in ["src", "tgt"]:
        setattr(
            symbolCounters,
            srcOrTgt,
            AttrDict({
                "tags":Counter(),
                "attrs":Counter(),
                "attrValues":Counter(),
                "text":Counter(),
                "all":Counter(),
            })
        )

        curXmlList = getattr(hier_dataset, srcOrTgt)
        curSymbolCounters = symbolCounters[srcOrTgt]
        for xmlTree in curXmlList:
            for xmlNode in xmlTree.getroot().iter():
                allConcateNated = ""
                # Node tag.
                curSymbolCounters.tags[xmlNode.tag] += 1
                allConcateNated += xmlNode.tag

                # Node text.
                if xmlNode.text is not None:
                    curSymbolCounters.text[SYM_SOS] += 1
                    for textSym in xmlNode.text:
                        curSymbolCounters.text[textSym] += 1
                    allConcateNated += xmlNode.text
                    curSymbolCounters.text[SYM_EOS] += 1
                if xmlNode.tail is not None:
                    curSymbolCounters.text[SYM_SOS] += 1
                    for textSym in xmlNode.tail:
                        curSymbolCounters.text[textSym] += 1
                    allConcateNated += xmlNode.tail
                    curSymbolCounters.text[SYM_EOS] += 1

                for attrName, attrValue in xmlNode.attrib.items():
                    # Attr name.
                    curSymbolCounters.attrs[attrName] += 1

                    # Attr values.
                    curSymbolCounters.attrValues[SYM_SOS] += 1
                    for attrSym in attrValue:
                        curSymbolCounters.attrValues[attrSym] += 1
                    allConcateNated += attrName
                    allConcateNated += attrValue
                    curSymbolCounters.attrValues[SYM_EOS] += 1

                for sym in allConcateNated:
                    curSymbolCounters.all[sym] += 1
        curSymbolCounters.all[SYM_SOS] = (
            curSymbolCounters.text[SYM_SOS]
            + curSymbolCounters.tags[SYM_SOS]
            + curSymbolCounters.attrs[SYM_SOS]
            + curSymbolCounters.attrValues[SYM_SOS]
        )
        curSymbolCounters.all[SYM_EOS] = curSymbolCounters.all[SYM_SOS]

        vocabs = AttrDict({})
        vocabs.tags = torchtext.vocab.Vocab(curSymbolCounters.tags, max_size=max_size)
        vocabs.attrs = torchtext.vocab.Vocab(curSymbolCounters.attrs, max_size=max_size)
        vocabs.attrValues = torchtext.vocab.Vocab(
            curSymbolCounters.attrValues,
            max_size=max_size,
            specials=[SYM_SOS, SYM_EOS, SYM_PAD],
            )
        vocabs.text = torchtext.vocab.Vocab(
            curSymbolCounters.text,
            max_size=max_size,
            specials=[SYM_SOS, SYM_EOS, SYM_PAD],
        )
        vocabs.all = torchtext.vocab.Vocab(
            curSymbolCounters.all,
            max_size=max_size,
            specials=[SYM_SOS, SYM_EOS, SYM_PAD, "<", ">", "/", " "],
        )
        setattr(resultVocabs, srcOrTgt, vocabs)

    tgtToSrcVocabMap = AttrDict({})
    for vocabKey in ["tags", "attrs", "attrValues", "text", "all"]:
        curTgtToSrcVocabMap = []
        for tgtSymbol in resultVocabs.tgt[vocabKey].itos:
            tgtVocabKey = "text" if vocabKey=="all" else vocabKey
            srcSymbolCode = resultVocabs.src[tgtVocabKey].stoi.get(tgtSymbol, -1)
            curTgtToSrcVocabMap.append(srcSymbolCode)
            

    return resultVocabs.src, resultVocabs.tgt, tgtToSrcVocabMap

class SourceField(torchtext.data.RawField):
    """ Wrapper class of torchtext.data.Field that forces batch_first and include_lengths to be True. """

    def __init__(self, **kwargs):
        self.is_target=False
        logger = logging.getLogger(__name__)
        super().__init__(**kwargs)

    def setVocabs(self, srcVocabs):
        self.vocabs = srcVocabs

class TargetField(torchtext.data.RawField):
    """ Wrapper class of torchtext.data.Field that forces batch_first to be True and prepend <sos> and append <eos> to sequences in preprocessing step.

    Attributes:
        sos_id: index of the start of sentence symbol
        eos_id: index of the end of sentence symbol
    """

    def __init__(self, **kwargs):
        self.is_target=True
        logger = logging.getLogger(__name__)
        kwargs["preprocessing"] = self.preprocess
        kwargs["postprocessing"] = self.postprocess
        super().__init__(**kwargs)

    def preprocess(self, *argv, **kargv):
        import pdb;pdb.set_trace()

    def postprocess(self, tgtBatch):
        values = []
        lengths = []
        for tgtXml in tgtBatch:
            tgtStr = ET.tostring(tgtXml.getroot(), encoding = "unicode")
            tgtVec = [self.sos_id] + [self.vocabs.all.stoi[sym] for sym in tgtStr ] + [self.eos_id]
            values.append(tgtVec)
            lengths.append(len(tgtVec))

        maxLength = max(lengths)
        for i, value in enumerate(values):
            values[i] = value + [self.pad_id for _ in range(maxLength - len(value))]

        return values, lengths

    def setVocabs(self, tgtVocabs):
        self.vocabs = tgtVocabs
        self.sos_id = self.vocabs.all.stoi[SYM_SOS]
        self.eos_id = self.vocabs.all.stoi[SYM_EOS]
        self.pad_id = self.vocabs.all.stoi[SYM_PAD]

        return
