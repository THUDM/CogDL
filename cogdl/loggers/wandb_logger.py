import warnings
from . import Logger

try:
    import wandb
except Exception:
    warnings.warn("Please install wandb first")


class WandbLogger(Logger):
    def __init__(self, log_path, project=None):
        super(WandbLogger, self).__init__(log_path)
        self.last_step = 0
        self.project = project

    def start(self):
        self.run = wandb.init(reinit=True, dir=self.log_path, project=self.project)

    def note(self, metrics, step=None):
        if not hasattr(self, "run"):
            self.start()
        if step is None:
            step = self.last_step
        self.run.log(metrics, step=step)
        self.last_step = step

    def finish(self):
        self.run.finish()
