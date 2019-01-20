from __future__ import print_function, division

import torch
import torchtext

import hier2hier
from hier2hier.loss import NLLLoss
from hier2hier.util import computeAccuracy
from hier2hier.dataset import Hier2HierIterator

class Evaluator(object):
    """ Class to evaluate models with given datasets.

    Args:
        model (hier2hier.models): model to evaluate
        data (hier2hier.dataset.dataset.Dataset): dataset to evaluate against
        loss (hier2hier.loss, optional): loss for evaluator (default: hier2hier.loss.NLLLoss)
        batch_size (int, optional): batch size for evaluator (default: 64)
    """

    def __init__(self, model, device, data, loss=NLLLoss(), batch_size=64):
        self.loss = loss
        self.batch_size = batch_size

        self.model = model
        self.batch_iterator = Hier2HierIterator(
            preprocess_batch=model.preprocess_batch,
            dataset=data, batch_size=self.batch_size,
            sort=False, shuffle=True,
            device=device, train=False)

    def evaluate(self, calcAccuracy=False):
        """ Evaluate a model on given dataset and return performance.

        Args:
        Returns:
            loss (float): loss of the given model on the given dataset
        """
        self.model.eval()

        loss = self.loss
        loss.reset()
        match = 0
        total = 0

        count = 0
        accuracy_total = 0
        beamAccuracy_total = 0

        with torch.no_grad():
            batch_generator = self.batch_iterator.__iter__()
            for hier2hierBatch in batch_generator:
                targetOutputsByToi = hier2hierBatch.targetOutputsByToi
                targetOutputLengthsByToi = hier2hierBatch.targetOutputLengthsByToi
                targetOutputsByTdolList = hier2hierBatch.targetOutputsByTdolList

                decodedSymbolsByTdolList, decodedSymbolsByToi = self.model(
                    hier2hierBatch,
                    collectOutput=calcAccuracy,
                    )

                if calcAccuracy:
                    accuracy = computeAccuracy(
                                            targetOutputsByToi,
                                            targetOutputLengthsByToi,
                                            decodedSymbolsByToi,
                                            device=self.device)

                    _, beamDecodedSymbolsByToi = self.model(
                                            hier2hierBatch,
                                            beam_count=6,
                                            collectOutput=calcAccuracy,
                                            )

                    beamAccuracy = computeAccuracy(
                                            targetOutputsByToi,
                                            targetOutputLengthsByToi,
                                            beamDecodedSymbolsByToi,
                                            device=self.device)

                    count += 1
                    accuracy_total += accuracy
                    beamAccuracy_total += beamAccuracy

                # Evaluation
                assert(len(targetOutputsByTdolList) == len(decodedSymbolsByTdolList))
                for i, decodedSymbolsByTdol in enumerate(decodedSymbolsByTdolList):
                    loss.eval_batch(decodedSymbolsByTdol, targetOutputsByTdolList[i])
                    
        if calcAccuracy:
            accuracy_avg = accuracy_total/count
            beamAccuracy_avg = beamAccuracy_total/count
        else:
            accuracy_avg = None
            beamAccuracy_avg = None

        return loss.get_loss() , accuracy_avg, beamAccuracy_avg
