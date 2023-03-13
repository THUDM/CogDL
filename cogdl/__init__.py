__version__ = "0.5.3"

from cogdl.backend import BACKEND

if BACKEND == "jittor":
    from .experiments_jt import experiment
elif BACKEND == "torch":
    from .experiments import experiment
else:
    raise ("Unsupported backend:", BACKEND)
# from .pipelines import pipeline
