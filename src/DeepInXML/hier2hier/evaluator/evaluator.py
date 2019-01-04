from __future__ import print_function, division

import torch
import torchtext

import hier2hier
from hier2hier.loss import NLLLoss
from hier2hier.util import computeAccuracy

class Evaluator(object):
    """ Class to evaluate models with given datasets.

    Args:
        loss (hier2hier.loss, optional): loss for evaluator (default: hier2hier.loss.NLLLoss)
        batch_size (int, optional): batch size for evaluator (default: 64)
    """

    def __init__(self, loss=NLLLoss(), batch_size=64):
        self.loss = loss
        self.batch_size = batch_size

    def evaluate(self, model, device, data, calcAccuracy=False):
        """ Evaluate a model on given dataset and return performance.

        Args:
            model (hier2hier.models): model to evaluate
            data (hier2hier.dataset.dataset.Dataset): dataset to evaluate against

        Returns:
            loss (float): loss of the given model on the given dataset
        """
        model.eval()

        loss = self.loss
        loss.reset()
        match = 0
        total = 0

        batch_iterator = torchtext.data.BucketIterator(
            dataset=data, batch_size=self.batch_size,
            sort=False, # sort_key=lambda x: len(x.src),
            device=device, train=False)
        tgt_vocab = data.fields[hier2hier.tgt_field_name].vocab
        pad = tgt_vocab.stoi[data.fields[hier2hier.tgt_field_name].pad_token]

        count = 0
        accuracy_total = 0
        beamAccuracy_total = 0

        with torch.no_grad():
            for batch in batch_iterator:
                input_variables  = getattr(batch, hier2hier.src_field_name)
                target_variables, target_lengths = getattr(batch, hier2hier.tgt_field_name)

                decodedSymbolProbs, decodedSymbols = model(
                    input_variables,
                    target_variables,
                    target_lengths,
                    collectOutput=calcAccuracy,
                    )

                if calcAccuracy:
                    accuracy = computeAccuracy(
                                            target_variables,
                                            target_lengths,
                                            decodedSymbols,
                                            device=device)

                    _, beamDecodedSymbols = model(
                                            input_variables,
                                            beam_count=6,
                                            collectOutput=calcAccuracy,
                                            )

                    beamAccuracy = computeAccuracy(
                                            target_variables,
                                            target_lengths,
                                            beamDecodedSymbols,
                                            device=device)

                    count += 1
                    accuracy_total += accuracy
                    beamAccuracy_total += beamAccuracy

                # Evaluation
                for step, step_output in enumerate(decodedSymbolProbs):
                    loss.eval_batch(step_output, target_variables[step])
                    
        if calcAccuracy:
            accuracy_avg = accuracy_total/count
            beamAccuracy_avg = beamAccuracy_total/count
        else:
            accuracy_avg = None
            beamAccuracy_avg = None

        return loss.get_loss() , accuracy_avg, beamAccuracy_avg
