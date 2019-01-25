from __future__ import unicode_literals, print_function, division
from collections import OrderedDict
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from hier2hier.util import blockProfiler, methodProfiler, lastCallProfile, nullTensorBoardHook

from .moduleBase import ModuleBase
from .hier2hierBatch import Hier2hierBatch, splitByToi
from .hierarchyPropagator import HierarchyPropagator
from .attrInfoPropagator import AttrInfoPropagator
from .encoderRNN import EncoderRNN
from .outputDecoder import OutputDecoder

curSchemaVersion = 0

class Hier2hier(ModuleBase):
    n = 0
    def __init__(self,
            modelArgs,
            debug,
            inputVocabs,
            outputVocab,
            sos_id,
            eos_id,
            device=None):
        super().__init__(device)
        self.debug = debug
        self.propagated_info_len = modelArgs.propagated_info_len

        #####################################################
        ############### Build sub-components. ###############
        #####################################################
        self.nodeTagsEmbedding = nn.Embedding(
            len(inputVocabs.tags),
            modelArgs.propagated_info_len,
        )

        # First hierarchy propagator.
        self.hierarchyPropagator1 = HierarchyPropagator(
            modelArgs.propagated_info_len,
            modelArgs.node_info_propagator_stack_depth,
            disable_batch_norm=modelArgs.disable_batch_norm,
            input_dropout_p=modelArgs.input_dropout_p,
            dropout_p=modelArgs.dropout_p,
            device=device,
        )

        # Attribute labels are also embedded into vectors.
        # These attribute label vectors act as initial state, when encoding attribute values.
        self.attrLabelEmbedding = nn.Embedding(
            len(inputVocabs.attrs),
            modelArgs.attrib_value_vec_len,
        )

        # Module to encode variable length attribute values into a fixed length vector.
        # NOTE:
        #   Positions within attribute value text are(currently) not used as a attention
        #   ready decoding position. This is because, it is belived that information
        #   content of each attribute text should be O(1) bits and comparable to
        #   a single symbol in the input text.
        self.attrValueEncoder = EncoderRNN(
            len(inputVocabs.attrValues),
            modelArgs.attrib_value_vec_len,
            input_dropout_p=modelArgs.input_dropout_p,
            dropout_p=modelArgs.dropout_p,
            n_layers=1,
            bidirectional=False,
            rnn_cell="gru",
            doEmbed=True,
            device=device,
        )

        # Module for propagating attributes and generating attention ready vectors, one for each attribute.
        self.attrInfoPropagator = AttrInfoPropagator(
            modelArgs.attrib_value_vec_len,
            modelArgs.propagated_info_len,
            device=device
        )

        # Info propagator for node.text.
        self.textInfoPropagator = EncoderRNN(
            len(inputVocabs.text),
            modelArgs.propagated_info_len,
            input_dropout_p=modelArgs.input_dropout_p,
            dropout_p=modelArgs.dropout_p,
            n_layers=1,
            bidirectional=False,
            rnn_cell="gru",
            doEmbed=True,
            update_embedding=False, # Permanently one-hot-encoded.
            device=device,
        )

        # Info propagator for node.tail.
        self.tailInfoPropagator = EncoderRNN(
            len(inputVocabs.text),
            modelArgs.propagated_info_len,
            input_dropout_p=modelArgs.input_dropout_p,
            dropout_p=modelArgs.dropout_p,
            n_layers=1,
            bidirectional=False,
            rnn_cell="gru",
            doEmbed=True,
            update_embedding=False, # Permanently one-hot-encoded.
            device=device,
        )

        # Second hierarchy propagator.
        self.hierarchyPropagator2 = HierarchyPropagator(
            modelArgs.propagated_info_len,
            modelArgs.node_info_propagator_stack_depth,
            disable_batch_norm=modelArgs.disable_batch_norm,
            input_dropout_p=modelArgs.input_dropout_p,
            dropout_p=modelArgs.dropout_p,
            device=device,
        )

        # Output Decoder.
        self.outputDecoder = OutputDecoder(
            outputVocab,
            modelArgs.propagated_info_len,
            modelArgs.attentionSubspaceVecLen,
            modelArgs.output_decoder_state_width,
            modelArgs.output_decoder_stack_depth,
            modelArgs.max_output_len,
            sos_id,
            eos_id,
            enableSpotlight=modelArgs.enableSpotlight,
            spotlightThreshold=modelArgs.spotlightThreshold,
            teacher_forcing_ratio=modelArgs.teacher_forcing_ratio,
            input_dropout_p=modelArgs.input_dropout_p,
            dropout_p=modelArgs.dropout_p,
            device=device,
            runtests=self.debug.runtests,
        )
        #####################################################
        #####################################################
        #####################################################

        if device is not None:
            super().cuda(device)
        else:
            super().cpu()

    def reset_parameters(self, device):
        self.hierarchyPropagator2.reset_parameters(device)
        self.tailInfoPropagator.reset_parameters(device)
        self.textInfoPropagator.reset_parameters(device)
        self.attrInfoPropagator.reset_parameters(device)
        self.attrValueEncoder.reset_parameters(device)
        self.attrLabelEmbedding.reset_parameters()
        self.hierarchyPropagator1.reset_parameters(device)
        self.nodeTagsEmbedding.reset_parameters()
        self.outputDecoder.reset_parameters(device)

    def preprocess_batch(self, batch):
        return Hier2hierBatch(batch)

    @methodProfiler
    def propagateThroughHierarchyInit(
        self,
        encodedNodesByNdfo,
        parentSelectorByNdfo,
        childSelectorByNdfoList,
        decreasingFanoutsFactorByNdfo,
        tensorBoardHook,
        debugPack=None,
    ):
        propagatedNodeInfoByNdfo = self.nodeTagsEmbedding(encodedNodesByNdfo)
        if debugPack is not None:
            (dataStagesToDebug, hier2hierBatch) = debugPack
            # Use ndfo2Toi partition.
            dataStagesToDebug.append(splitByToi(
                propagatedNodeInfoByNdfo,
                hier2hierBatch.ndfo2Toi,
                hier2hierBatch.sampleCount
            ))

        # Propagate through nodes.
        propagatedNodeInfoByNdfo = self.hierarchyPropagator1(
                    propagatedNodeInfoByNdfo,
                    parentSelectorByNdfo,
                    childSelectorByNdfoList,
                    decreasingFanoutsFactorByNdfo,
                    tensorBoardHook,
                    debugPack,
                )

        return propagatedNodeInfoByNdfo

    @methodProfiler
    def propagateThroughAttributes(
        self,
        propagatedNodeInfoByNdfo,
        encodedAttrLabelsByAvdl,
        encodedAttrSymbolsByAvdlp,
        avdl2Ndac,
        avdl2Ndfo,
        ndac2Ndfo,
        avdlAttrSelectorsListByNdac,
        decreasingAttrCountsFactorByNdac,
    ):
        # Propagated info in Avdl order(permutation of the attribute world).
        propagatedNodeInfoByAvdl = torch.index_select(
            propagatedNodeInfoByNdfo,
            0,
            avdl2Ndfo
        )

        # Encode attribute labels.
        embeddedAttrLabelsByAvdl = self.attrLabelEmbedding(encodedAttrLabelsByAvdl)

        # Encode attributes.
        (
            ignoredAttnReadyAttrPositionInfoByAvdlp,
            encodedAttrsByAvdl,
        ) = self.attrValueEncoder(
                    encodedAttrSymbolsByAvdlp,
                    embeddedAttrLabelsByAvdl.view(
                        [1] + list(embeddedAttrLabelsByAvdl.shape)
                    ),
                )

        # Propagate through attributes.
        propagatedNodeInfoByNdac = propagatedNodeInfoByNdfo[ndac2Ndfo]
        if not avdlAttrSelectorsListByNdac:
            propagatedNodeInfoByNdac = propagatedNodeInfoByNdfo[ndac2Ndfo]
            attnReadyAttrInfoByAvdl = torch.tensor([])
        else:
            propagatedNodeInfoByNdac, attnReadyAttrInfoByAvdl = self.attrInfoPropagator(
                encodedAttrsByAvdl,
                propagatedNodeInfoByNdac,
                avdlAttrSelectorsListByNdac,
                decreasingAttrCountsFactorByNdac,
                avdl2Ndac,
            )

        return propagatedNodeInfoByNdac, attnReadyAttrInfoByAvdl, ignoredAttnReadyAttrPositionInfoByAvdlp

    @methodProfiler
    def propagateThroughTextAndTail(
        self,
        propagatedNodeInfoByNdac,
        encodedTextByTtDLP,
        encodedTailByTlDLP,
        ndttl2ndac,
        ndtll2ndttl,
        ndfo2ndtll,
    ):
        # Propagate through node.text
        propagatedNodeInfoByNDTtL = propagatedNodeInfoByNdac[ndttl2ndac]
        if encodedTextByTtDLP is not None:
            attnReadyTextInfoByTtDLP, propagatedNodeInfoByNDTtL = self.textInfoPropagator(
                        encodedTextByTtDLP,
                        propagatedNodeInfoByNDTtL.view(
                            [1] + list(propagatedNodeInfoByNDTtL.shape)
                        ),
                    )
        else:
            attnReadyTextInfoByTtDLP = torch.zeros([0, self.propagated_info_len])

        # Propagate through node.tail
        propagatedNodeInfoByNDTlL = propagatedNodeInfoByNDTtL[ndtll2ndttl]
        if encodedTailByTlDLP is not None:
            attnReadyTextInfoByTlDLP, propagatedNodeInfoByNDTlL = self.tailInfoPropagator(
                        encodedTailByTlDLP,
                        propagatedNodeInfoByNDTlL.view(
                            [1] + list(propagatedNodeInfoByNDTlL.shape)
                        ),
                    )
        else:
            attnReadyTextInfoByTlDLP = torch.zeros([0, self.propagated_info_len])

        # Permute back into NDFO.
        propagatedNodeInfoByNdfo = propagatedNodeInfoByNDTlL[ndfo2ndtll]

        return attnReadyTextInfoByTtDLP, attnReadyTextInfoByTlDLP, propagatedNodeInfoByNdfo

    @methodProfiler
    def propagateThroughHierarchyFinal(
        self,
        propagatedNodeInfoByNdfo,
        parentSelectorByNdfo,
        childSelectorByNdfoList,
        decreasingFanoutsFactorByNdfo,
        tensorBoardHook,
        debugPack=None,
    ):
        if debugPack is not None:
            (dataStagesToDebug, hier2hierBatch) = debugPack
        # Propagate through hierarchy one final time. This gives attention ready nodes
        # to be influenced by text and attr data, one last time.
        propagatedNodeInfoByNdfo = self.hierarchyPropagator2(
                    propagatedNodeInfoByNdfo,
                    parentSelectorByNdfo,
                    childSelectorByNdfoList,
                    decreasingFanoutsFactorByNdfo,
                    tensorBoardHook,
                    debugPack,
                )

        return propagatedNodeInfoByNdfo

    @methodProfiler
    def forward(self,
            hier2hierBatch,
            tensorBoardHook=nullTensorBoardHook,
            beam_count=None,
            collectOutput=None,
            debugDataStages=False,
            clip_output_len=None,
    ):
        if debugDataStages:
            dataStagesToDebug = []
            debugPack = (dataStagesToDebug, hier2hierBatch)
        else:
            dataStagesToDebug = None
            debugPack = None

        # Handle node hierarchy(first pass).
        propagatedNodeInfoByNdfo = self.propagateThroughHierarchyInit(
            hier2hierBatch.encodedNodesByNdfo,
            hier2hierBatch.parentSelectorByNdfo,
            hier2hierBatch.childSelectorByNdfoList,
            hier2hierBatch.decreasingFanoutsFactorByNdfo,
            tensorBoardHook,
            debugPack,
        )
        if debugDataStages:
            # Use ndfo2Toi partition.
            dataStagesToDebug.append(splitByToi(
                propagatedNodeInfoByNdfo,
                hier2hierBatch.ndfo2Toi,
                hier2hierBatch.sampleCount
            ))

        # Handle node.attributes.
        (
            propagatedNodeInfoByNdac,
            attnReadyAttrInfoByAvdl,
            ignoredAttnReadyAttrPositionInfoByAvdlp
        ) = self.propagateThroughAttributes(
            propagatedNodeInfoByNdfo,
            hier2hierBatch.encodedAttrLabelsByAvdl,
            hier2hierBatch.encodedAttrSymbolsByAvdlp,
            hier2hierBatch.avdl2Ndac,
            hier2hierBatch.avdl2Ndfo,
            hier2hierBatch.ndac2Ndfo,
            hier2hierBatch.avdlAttrSelectorsListByNdac,
            hier2hierBatch.decreasingAttrCountsFactorByNdac,
        )
        if debugDataStages:
            # Use ndfo2Toi(ndac2Ndfo(x)), avdl2Toi and avdlp2Toi.
            dataStagesToDebug.append(splitByToi(
                propagatedNodeInfoByNdac,
                hier2hierBatch.ndac2Toi,
                hier2hierBatch.sampleCount,
            ))
            dataStagesToDebug.append(splitByToi(
                attnReadyAttrInfoByAvdl,
                hier2hierBatch.avdl2Toi,
                hier2hierBatch.sampleCount,
            ))
            dataStagesToDebug.append(splitByToi(
                ignoredAttnReadyAttrPositionInfoByAvdlp.data,
                hier2hierBatch.avdlp2Toi,
                hier2hierBatch.sampleCount,
            ))

        # Handle node.text and node.tail.
        (
            attnReadyTextInfoByTtDLP,
            attnReadyTailInfoByTlDLP,
            propagatedNodeInfoByNdfo,
        ) = self.propagateThroughTextAndTail(
            propagatedNodeInfoByNdac,
            hier2hierBatch.encodedTextByTtDLP,
            hier2hierBatch.encodedTailByTlDLP,
            hier2hierBatch.ndttl2Ndac,
            hier2hierBatch.ndtll2Ndttl,
            hier2hierBatch.ndfo2Ndtll,
        )
        if debugDataStages:
            # Use ndfo2Toi and ndtxl2Toi.
            # Use ndfo2Toi(ndac2Ndfo(x)), avdl2Toi and avdlp2Toi.
            dataStagesToDebug.append(splitByToi(
                attnReadyTextInfoByTtDLP.data,
                hier2hierBatch.ndtlp2Toi2[False],
                hier2hierBatch.sampleCount,
            ))
            dataStagesToDebug.append(splitByToi(
                attnReadyTailInfoByTlDLP.data,
                hier2hierBatch.ndtlp2Toi2[True],
                hier2hierBatch.sampleCount,
            ))
            dataStagesToDebug.append(splitByToi(
                propagatedNodeInfoByNdfo,
                hier2hierBatch.ndfo2Toi,
                hier2hierBatch.sampleCount,
            ))

        # Handle node hierarchy(final pass).
        attnReadyNodeInfoByNdfo = self.propagateThroughHierarchyFinal(
            propagatedNodeInfoByNdfo,
            hier2hierBatch.parentSelectorByNdfo,
            hier2hierBatch.childSelectorByNdfoList,
            hier2hierBatch.decreasingFanoutsFactorByNdfo,
            tensorBoardHook,
            debugPack,
        )
        if debugDataStages:
            # Use ndfo2Toi.
            dataStagesToDebug.append(splitByToi(
                attnReadyNodeInfoByNdfo,
                hier2hierBatch.ndfo2Toi,
                hier2hierBatch.sampleCount
            ))

        # Concatenate all attention ready infos.
        posEncodedVecsByGni = torch.cat([
            attnReadyNodeInfoByNdfo,
            attnReadyAttrInfoByAvdl,
            attnReadyTextInfoByTtDLP.data,
            attnReadyTailInfoByTlDLP.data,
        ])
        posEncodedVecsByGndtol = posEncodedVecsByGni[hier2hierBatch.gndtol2Gni]

        if debugDataStages:
            # Use gni2Toi.
            dataStagesToDebug.append(splitByToi(
                posEncodedVecsByGndtol,
                hier2hierBatch.gndtol2Toi,
                hier2hierBatch.sampleCount
            ))

        # Decode attnReadyEncodedPositions to obtain the desired output.
        (
            outputSymbolsByTdolList,
            outputSymbols,
            teacherForcedSelections,
        ) = self.outputDecoder(
            hier2hierBatch.sampleCount,
            hier2hierBatch.posNbrhoodGraphByGndtol,
            hier2hierBatch.fullSpotlight,
            posEncodedVecsByGndtol,
            hier2hierBatch.targetOutputsByTdol,
            hier2hierBatch.targetOutputLengthsByTdol,
            hier2hierBatch.gndtol2Tdol,
            hier2hierBatch.tdol2Toi,
            hier2hierBatch.toi2Tdol,
            tensorBoardHook,
            collectOutput=collectOutput,
            beam_count=beam_count,
            clip_output_len=clip_output_len,
            debugPack=debugPack)
        if debugDataStages:
            for symIndex, outputSymbolsByTdol in enumerate(outputSymbolsByTdolList):
                dataStagesToDebug.append(splitByToi(
                    outputSymbolsByTdol,
                    hier2hierBatch.tdol2Toi,
                    hier2hierBatch.sampleCount,
                    "{0}@".format(symIndex),
                ))

        if debugDataStages:
            return dataStagesToDebug
        else:
            return outputSymbolsByTdolList, outputSymbols
