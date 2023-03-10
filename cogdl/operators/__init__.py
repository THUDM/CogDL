
from cogdl.backend import BACKEND

if BACKEND == 'jittor':
    from .jittor import *
elif BACKEND == 'torch':
    from .torch import *
else:
    raise ("Unsupported backend:", BACKEND)