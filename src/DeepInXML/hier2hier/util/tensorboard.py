from tensorboardX import SummaryWriter 

class TensorBoardHook(SummaryWriter):
    def __init__(self, periodicity, *argv, **kargv):
        super().__init__(*argv, **kargv)
        self._batch = -1
        self._epoch = -1
        self._step = -1
        self.periodicity = periodicity

    def batchNext(self):
        self._batch += 1
        self._step += 1

    def epochNext(self):
        self._batch = -1
        self._epoch += 1

    def stepReset(self, step=None, epoch=None, steps_per_epoch=None):
        if step is not None:
            self._step = step
        if epoch is not None:
            self._epoch = epoch
        if steps_per_epoch is not None:
            self._batch = step - (epoch-1)*steps_per_epoch

    def add_scalar(self, label, value):
        if self.periodicity is 0:
            return
        super().add_scalar(label, value, self._epoch)

    def add_histogram(self, label, value):
        if self.periodicity is 0 or self._epoch % self.periodicity != 0:
            return
        super().add_histogram(label, value.cpu().detach().numpy(), self._epoch)
