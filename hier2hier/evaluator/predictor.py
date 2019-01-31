import torch
from torch.autograd import Variable


class Predictor(object):

    def __init__(self, model, src_vocabs, tgt_vocab):
        """
        Predictor class to evaluate for a given model.
        Args:
            model (hier2hier.models): trained model. This can be loaded from a checkpoint
                using `hier2hier.util.checkpoint.load`
            src_vocab (hier2hier.dataset.vocabulary.Vocabulary): source sequence vocabulary
            tgt_vocab (hier2hier.dataset.vocabulary.Vocabulary): target sequence vocabulary
        """
        if torch.cuda.is_available():
            self.model = model.cuda()
        else:
            self.model = model.cpu()
        self.model.eval()
        self.src_vocabs = src_vocabs
        self.tgt_vocab = tgt_vocab

    def get_decoder_features(self, src_seq):
        src_id_seq = torch.LongTensor([self.src_vocab.stoi[tok] for tok in src_seq]).view(1, -1)
        if torch.cuda.is_available():
            src_id_seq = src_id_seq.cuda()

        with torch.no_grad():
            softmax_list, _, other = self.model(src_id_seq, [len(src_seq)])

        return other

    def predict(self, src_trees):
        """ Make prediction given `src_seq` as input.

        Args:
            src_seq (list): list of tokens in source language

        Returns:
            tgt_seq (list): list of tokens in target language as predicted
            by the pre-trained model
        """
        _, outputSymbolsList = self.model(src_trees)

        tgt_outputs = [
            [self.tgt_vocab.itos[tok] for tok in outputSymbols]
            for outputSymbols in outputSymbolsList
        ]
        return tgt_outputs

    def predict_n(self, src_seq, n=1):
        """ Make 'n' predictions given `src_seq` as input.

        Args:
            src_seq (list): list of tokens in source language
            n (int): number of predicted seqs to return. If None,
                     it will return just one seq.

        Returns:
            tgt_seq (list): list of tokens in target language as predicted
                            by the pre-trained model
        """
        other = self.get_decoder_features(src_seq)

        result = []
        for x in range(0, int(n)):
            length = other['topk_length'][0][x]
            tgt_id_seq = [other['topk_sequence'][di][0, x, 0].data[0] for di in range(length)]
            tgt_seq = [self.tgt_vocab.itos[tok] for tok in tgt_id_seq]
            result.append(tgt_seq)

        return result

#if __name__ == "__main__":

