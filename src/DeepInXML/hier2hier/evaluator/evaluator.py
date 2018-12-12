from __future__ import print_function, division

import torch
import torchtext

import hier2hier
from hier2hier.loss import NLLLoss

class Evaluator(object):
    """ Class to evaluate models with given datasets.

    Args:
        loss (hier2hier.loss, optional): loss for evaluator (default: hier2hier.loss.NLLLoss)
        batch_size (int, optional): batch size for evaluator (default: 64)
    """

    def __init__(self, loss=NLLLoss(), batch_size=64):
        self.loss = loss
        self.batch_size = batch_size

    def evaluate(self, model, data):
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

        device = None if torch.cuda.is_available() else -1
        batch_iterator = torchtext.data.BucketIterator(
            dataset=data, batch_size=self.batch_size,
            sort=False, # sort_key=lambda x: len(x.src),
            device=device, train=False)
        tgt_vocab = data.fields[hier2hier.tgt_field_name].vocab
        pad = tgt_vocab.stoi[data.fields[hier2hier.tgt_field_name].pad_token]

        with torch.no_grad():
            for batch in batch_iterator:
                input_variables  = getattr(batch, hier2hier.src_field_name)
                target_variables, target_lengths = getattr(batch, hier2hier.tgt_field_name)

                decoder_outputs, decoder_hidden = model(input_variables, target_variables, target_lengths)

                # Evaluation
                # seqlist = other['sequence']
                for step, step_output in enumerate(decoder_outputs):
                    loss.eval_batch(step_output, target_variables[step])

                    #non_padding = target.ne(pad)
                    #correct = seqlist[step].view(-1).eq(target).masked_select(non_padding).sum().item()
                    #match += correct
                    #total += non_padding.sum().item()

        #if total == 0:
        #    accuracy = float('nan')
        #else:
        #    accuracy = match / total

        return loss.get_loss() #, accuracy
