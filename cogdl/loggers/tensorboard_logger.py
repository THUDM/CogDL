from tensorboardX import SummaryWriter

from . import Logger


class TBLogger(Logger):
    def __init__(self, log_path):
        super(TBLogger, self).__init__(log_path)
        self.last_step = 0

    def start(self):
        self.writer = SummaryWriter(logdir=self.log_path)

    def note(self, metrics, step=None):
        if not hasattr(self, "writer"):
            self.start()
        if step is None:
            step = self.last_step
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)
        self.last_step = step

    def finish(self):
        self.writer.close()
