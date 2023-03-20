from cogdl.backend import BACKEND

if BACKEND == "jittor":
    from .jittor import Trainer

elif BACKEND == "torch":
    from .torch import Trainer
else:
    raise ("Unsupported backend:", BACKEND)
