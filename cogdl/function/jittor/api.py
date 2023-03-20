import jittor as jt
import numpy as np


def is_tensor(obj):  # r isinstance():
    return isinstance(obj, jt.Var)


def ones(*shape, dtype="float32", device=None):
    if isinstance(shape, jt.Var):
        shape = shape.item()
    if isinstance(shape[0], tuple):
        return jt.ones(shape[0], dtype=dtype)
    else:
        return jt.ones(shape, dtype=dtype)


def zeros(*shape, dtype="float32", device=None):
    if isinstance(shape, jt.Var):
        shape = shape.item()
    if isinstance(shape[0], tuple):
        return jt.zeros(shape[0], dtype=dtype)
    else:
        return jt.zeros(shape, dtype=dtype)


def zeros_like(input, dtype=None, device=None):
    return jt.zeros_like(input, dtype=dtype)


def ones_like(input):
    if isinstance(input, jt.Var):
        input = input.item()
    return jt.ones_like(input)


def arange(start=0, end=None, step=1, dtype=None, device=None):
    if end is None:
        end, start = start, 0
    return jt.arange(start, end, step, dtype)


def cat(input, dim=0):
    return jt.concat(input, dim)


def pow(x, y):
    return jt.pow(x, y)


def isinf(x):
    return x == float("inf")


def stack(x, dim=0):
    return jt.stack(x, dim=dim)


def where(input):  # jittor.bool
    return jt.where(input)


def max(input, dim=None):
    # return value
    if dim is None:
        return jt.max(input)
    else:
        return jt.max(input, dim=dim)


def argmax(input, dim=0):
    # NOTE: return indices
    return jt.argmax(input, dim=dim)[0]


def tensor(*input, dtype=None):
    return jt.array(input[0], dtype=dtype)

def FloatTensor(*input, size=None):
    if size is not None:
        return jt.rand(size)
    elif len(input)>1 or len(input)==1 and isinstance(input[-1], int):  #(3,2),((3))
        return jt.rand(input) 
    else:
        return jt.array(input[0], dtype="float")


def LongTensor(*input, size=None):
    if size is not None:
        return jt.randint(0,100,shape=size)
    elif len(input)>1 or len(input)==1 and isinstance(input[-1], int):  #(3,2)
        return jt.randint(0,100,shape=input)
    else:
        return jt.array(input[0], dtype="long")


def BoolTensor(input):
    return jt.array(input, dtype="bool")

def full(shape, val, dtype="float32", device=None):
    return jt.full(shape, val, dtype)


def rand(*size, dtype="float32", requires_grad=True, device=None):
    return jt.rand(*size, dtype="float32", requires_grad=True)


def randn(*size, dtype="float32", requires_grad=True, device=None):
    return jt.randn(*size, dtype="float32", requires_grad=True)


def randint(low, high, size, device=None):
    return jt.randint(low, high, shape=size)


def from_numpy(input):
    if isinstance(input, np.matrix):
        input = input.getA()
    return jt.array(input)


def as_tensor(data, dtype=None, device=None):
    return jt.array(data, dtype=dtype)


def unique(input, dim=None, return_inverse=False, return_counts=False):
    # retyrn output，counts
    output, inverse, counts = jt.unique(input, dim=None, return_inverse=True, return_counts=True)
    if return_counts and not return_inverse:
        return output, counts
    elif return_inverse and not return_counts:
        return output, inverse
    elif not return_inverse and not return_counts:
        return output
    else:
        return output, inverse, counts


def cpu(input):
    return input


def to(input, to_device):
    return input


def device(input):
    if jt.flags.use_cuda == 0:
        return "cpu"
    else:
        return "cuda"


def dtype_dict(dtype):
    type = {
        "float16": "float16",
        "float": "float32",
        "float32": "float32",
        "float64": "float64",
        "int8": "int8",
        "int16": "int16",
        "int32": "int32",
        "int64": "int64",
        "long": "int64",
        "bool": "bool",
        "tensor": jt.Var,
    }
    return type[dtype]


def dim(input):
    return len(input.shape)


def load(path):
    return jt.load(path)


def save(obj, path):
    return jt.save(obj, path)


def sort(data, dim=-1, descending=False):
    # return value, index
    index, value = jt.argsort(data, dim=dim, descending=descending)
    return value, index


def argsort(data, dim=-1, descending=False):
    # return index
    index, value = jt.argsort(data, dim=dim, descending=descending)
    return index


def sum(input, dim, keepdim=False):
    return jt.sum(input, dim=dim, keepdims=keepdim)


def isnan(input):
    return jt.isnan(input)


def scatter_add_(x, dim, index, src, reduce="add"):
    return x.scatter_(dim=dim, index=index, src=src, reduce=reduce)


def bincount(input, weights=None, minlength=0):
    if isinstance(weights, jt.Var):
        weights = weights.numpy()
    return jt.array(np.bincount(input.numpy(), weights=weights, minlength=minlength))


def set_random_seed(seed):
    jt.misc.set_global_seed(seed)


def cuda_is_available():
    if jt.flags.use_cuda == 0:
        return False
    else:
        return True


def index_select(x, dim, index):
    return x.getitem(((slice(None),) * dim) + (index,))


def astype(input, type):
    return input.astype(type)


def type_as(x, y):
    return x.type_as(y)


def logical_not(input):
    return jt.logical_not(input)


def squeeze(input, dim=None):
    # jt.squeeze,Add dim is NOne
    if dim is None:
        shape = list(input.shape)
        for i in shape:
            if i == 1:
                shape.remove(1)
        return input.reshape(shape)
    return jt.squeeze(input, dim=dim)


def unsqueeze(input, dim):
    return jt.unsqueeze(input, dim)


def eq(input, other):
    return jt.equal(input, other)


def repeat_interleave(input, repeat, dim=None):
    # TODO Add dim, repeadt is int
    output = jt.array([item for n, s in zip(repeat.numpy(), input.numpy()) for item in [s] * n])
    return output


def mean(x, dim=None, keepdims=False):
    if dim is None:
        return jt.mean(x)
    else:
        return jt.mean(x, dim=dim, keepdims=keepdims)


def matmul(x, y):
    return jt.matmul(x, y)


def cuda_empty_cache():
    jt.gc()


def bernoulli(x):
    return jt.bernoulli(x)


def contiguous(x):
    return x.clone()


def div(a, b):
    return jt.divide(a, b)


def fill_(a, b):
    a[a.bool()] = b
    return a


def xavier_uniform_(x):
    jt.nn.init.xavier_uniform_(x)


def sigmoid(x):
    return jt.sigmoid(x)


def exp(x):
    return jt.exp(x)


def abs(x):
    return jt.abs(x)


def log(x):
    return jt.log(x)


def normalize(input, p, dim):
    return jt.normalize(input=input, p=p, dim=dim)


def sparse_FloatTensor(indices, values, shape):
    return jt.sparse.sparse_array(indices, values, shape)


def sparse_mm(x, y):
    return jt.sparse.mm(x, y)  # TODO


def diag(x, diagonal=0):
    return jt.diag(x, diagonal)
