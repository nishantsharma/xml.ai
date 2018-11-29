'''
Generates training data for XML transforms learner.
'''
from __future__ import print_function

from orderedattrdict import AttrDict
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from utils import saveModel
import xml.etree.ElementTree as ET

import numpy as np

def extractTrainableInput(origTree, trainableDiffTree, panrentSignal, trainableInputSamples):
    origNodeName = origTree.tag
    origChildren = origTree.getchildren()
    origAttrs = origTree.getchildren()

    newNodeName = trainableDiffTree.tag
    newChildren = trainableDiffTree.getchildren()
    newAttrs = trainableDiffTree.getchildren()

    # Extract code for attributes.
    origAttrsInputCode = attributesCodec.encodeInput(origNodeName, origChildren, origAttrs)
    origAttrsCode = attributesCodec.encode(origAttrsInputCode)

    # Extract code for children.
    origChildrenCodeList = []
    for index, origChild in enumerate(origChildren):
        origChildrenCodeList.append(
                extractTrainableInput(origChild, newChildren[index], origAttrsCode, trainableInputSamples))
    origChildrenInputCode = childrenCodec.encodeInput(origChildrenCodeList)
    origChildrenCode = childrenCodec.encode(origChildrenInputCode)

    # Code for complete node is concatenation of codes of attribuets and children.
    curNodeCode = (origAttrsCode, origChildrenCode)

    # Build decoder inputs.
    childrenDecoderInputs = childrenCodec.decoderInputsForTraining(curNodeCode, newChildren)
    attrsDecoderInputs = attributesCodec.decoderInputsForTraining(curNodeCode, newAttributes)
    decoderInputs = (childrenDecoderInputs, attrsDecoderInputs)

    # List training data.
    trainableInputSamples.append([origAttrsInputCode, origChildrenInputCode, decoderInputs])

    # Use code for the current node, to be used by parent.
    return curNodeCode

def get_inputs(trainArgs):
    origFilesFolder = trainArgs.sourceFolder
    updatedFilesFolder = trainArgs.updatedFilesFolder
    trainingDiffsFolder = trainArgs.trainingDiffsFolder

    trainableInputSamples = []
    for filename in os.path.listdir(origFilesFolder):
        if not filename.lower().endsith(".xml"):
            continue

        origXmlTree = ET.parse(origFilesFolder + filename)
        trainableDiffTree = ET.parse(trainingDiffsFolder + filename)

        extractTrainableInput(
                origXmlTree,
                trainableDiffTree,
                None,
                trainableInputSamples)

    return trainableInputSamples

