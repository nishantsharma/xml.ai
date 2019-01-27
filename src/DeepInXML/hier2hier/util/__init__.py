import glob, os, argparse, attrdict, math
from attrdict import AttrDict
from enum import IntEnum

import torch

from .profiler import (lastCallProfile, appendProfilingData, methodProfiler,
                        blockProfiler, summarizeLabelNodes, appendProfilingLabel)
from .tensorboard import TensorBoardHook

nullTensorBoardHook = TensorBoardHook(0)

class AppMode(IntEnum):
    Generate=0
    Train=1
    Evaluate=2
    Test=3

def longTensor(data, device):
    return torch.tensor(torch.LongTensor(data), device=device)

def invertPermutation(perm, asList=None):
    if isinstance(perm, dict):
        keys = perm.keys()
        values = perm.values()
    else:
        keys = range(len(perm))
        values = perm

    if set(values) == set(range(len(perm))):
        # Build a list.
        inverse = [ None for _ in range(len(perm)) ]
        for k, v in zip(keys, values):
            inverse[v] = k
        return inverse
    else:
        return sorted(list(zip(values, keys)))
    
def onehotencode(n, i):
    return [1.0 if j==i else 0.0 for j in range(n)]

def str2bool(v):
    """
    Converts string to bool.
    
    This is used as type in argparse arguments to parse command line arguments.
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def str2bool3(v):
    """
    Converts string to bool or None.
    
    This is used as type in argparse arguments to parse command line arguments.
    """
    if v is None:
        return v
    else:
        return str2bool(v)

debugNans = False
def checkNans(t):
    if not debugNans:
        return
    if isinstance(t, torch.Tensor):
        nanCount = torch.isnan(t.view(torch.numel(t))).sum() + torch.isinf(t.view(torch.numel(t))).sum()
    elif isinstance(t, list):
        return any([checkNans(t_) for t_ in t])
    elif t is not None and (math.isnan(t) or math.isinf(t)):
        nanCount = 1
    else:
        nanCount = 0
    
    if int(nanCount):
        import pdb;pdb.set_trace()
        return True
    else:
        return False

def levelDown(parsedArgs, label, keys):
        """
        Command line arguments parsed using argparse are not hierarchical. This
        method helps them make hierarchical.

        The input parsedArgs is a config tree. We move each key in keys=["k1", "k2", "k3",
        ...] currently present under parsedArgs to a level below under parsedArgs[label].
        """
        lowerLevelDict = AttrDict()
        for key in keys:
            if hasattr(parsedArgs, key):
                lowerLevelDict[key] = getattr(parsedArgs, key)
                delattr(parsedArgs, key)

        return lowerLevelDict

def batched_index_select(input, dim, index):
    views = [input.shape[0]] + [1 if i != dim else -1 for i in range(1, len(input.shape))]
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(input, dim, index)

def computeAccuracy(expectedOutputs, expectedLengths, generatedOutput, device=None):
    matchCount, maskCount = 0, 0
    for i in range(expectedOutputs.shape[0]):
        n=int(min(expectedLengths[i], len(generatedOutput[i])))
        assert(expectedOutputs.shape[1] >= expectedLengths[i])
        expected = expectedOutputs[i][0:n]
        generated = generatedOutput[i][0:n]
        if isinstance(generated, list):
            generated = torch.tensor(generated, device=device)

        if expected.shape[0] < generated.shape[0]:
            generated = generated[0:expected.shape[0]]
        elif expected.shape[0] > generated.shape[0]:
            expected = expected[0:generated.shape[0]]

        sampleMatcheCount = int(sum((expected - generated == 0).int()))
        matchCount += sampleMatcheCount
        if sampleMatcheCount == 0:
            break

    return matchCount/float(sum(expectedLengths.int()))
