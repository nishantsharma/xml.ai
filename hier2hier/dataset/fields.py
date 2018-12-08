import logging

from orderedattrdict import AttrDict
from collections import Counter

import torchtext

SYM_SOS = '<sos>'
SYM_EOS = '<eos>'

class SourceField(torchtext.data.RawField):
    """ Wrapper class of torchtext.data.Field that forces batch_first and include_lengths to be True. """

    def __init__(self, **kwargs):
        logger = logging.getLogger(__name__)
        super().__init__(**kwargs, postprocessing=self.postprocess, preprocessing=self.preprocess)

    def preprocess(self,  example):
        import pdb;pdb.set_trace()

    def postprocess(self,  batch):
        return batch
        
    def build_vocabs(self, hier_dataset, max_size=None):
        # Collect all symbols.
        inputTags, inputAttribs, inputAttribValues, inputText = Counter(), Counter(), Counter(), Counter()
        for inXmlObj in hier_dataset.src:
            for xmlNode in inXmlObj.getroot().iter():
                # Node tag.
                inputTags[xmlNode.tag] += 1

                # Node text.
                inputText[SYM_SOS] += 1
                for textSym in xmlNode.text:
                    inputText[textSym] += 1
                inputText[SYM_EOS] += 1

                for attrbName, attribValue in xmlNode.attrib.items():
                    # Attrib name.
                    inputAttribs[attribName] += 1

                    # Attrib values.
                    inputAttribValues[SYM_SOS] += 1
                    for attribSym in attribValue:
                        inputAttribValues[attribSym] += 1
                    inputAttribValues[SYM_EOS] += 1

        # Build all vocabs.
        self.vocabs = AttrDict()
        self.vocabs.tags = torchtext.vocab.Vocab(inputTags, max_size)
        self.vocabs.attribs = torchtext.vocab.Vocab(inputAttribs, max_size)
        self.vocabs.attribValues = torchtext.vocab.Vocab(inputAttribValues, max_size)
        self.vocabs.text = torchtext.vocab.Vocab(inputText, max_size) 

class TargetField(torchtext.data.Field):
    """ Wrapper class of torchtext.data.Field that forces batch_first to be True and prepend <sos> and append <eos> to sequences in preprocessing step.

    Attributes:
        sos_id: index of the start of sentence symbol
        eos_id: index of the end of sentence symbol
    """


    def __init__(self, **kwargs):
        logger = logging.getLogger(__name__)

        if kwargs.get('sequential') is False:
            logger.warning("Target field in pytorch-hier2hier mut be set to True.  Changed.")
        kwargs["sequential"] = True

        if kwargs.get('batch_first') == False:
            logger.warning("Option batch_first has to be set to use pytorch-hier2hier.  Changed to True.")
        kwargs['batch_first'] = True

        if kwargs.get('preprocessing') is None:
            kwargs['preprocessing'] = lambda seq: [self.SYM_SOS] + seq + [self.SYM_EOS]
        else:
            func = kwargs['preprocessing']
            kwargs['preprocessing'] = lambda seq: [self.SYM_SOS] + func(seq) + [self.SYM_EOS]

        if kwargs.get('include_lengths') is False:
             logger.warning("Option include_lengths has to be set to use pytorch-seq2seq.  Changed to True.")
        kwargs['include_lengths'] = True

        kwargs["init_token"] = SYM_SOS
        kwargs["eos_token"] = SYM_EOS
        super(TargetField, self).__init__(**kwargs)

    def build_vocab(self, *args, **kwargs):
        super(TargetField, self).build_vocab(*args, **kwargs)
        self.sos_id = self.vocab.stoi[SYM_SOS]
        self.eos_id = self.vocab.stoi[SYM_EOS]
