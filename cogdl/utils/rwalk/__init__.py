"""

"""
import numpy as np
import numpy.ctypeslib as npct
from ctypes import c_float, c_int
from os.path import dirname

array_1d_int = npct.ndpointer(dtype=np.int32, ndim=1, flags="CONTIGUOUS")

librwalk = npct.load_library("librwalk", dirname(__file__))

print("rwalk: Loading library from: {}".format(dirname(__file__)))
# librwalk.random_walk.restype = None
# librwalk.random_walk.argtypes = [array_1d_int, array_1d_int, c_int, c_int, c_int, c_int, c_int, array_1d_int]

librwalk.random_walk.restype = None
librwalk.random_walk.argtypes = [
    array_1d_int,
    array_1d_int,
    array_1d_int,
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
    c_float,
    array_1d_int,
]


def random_walk(nodes, ptr, neighs, num_walks=1, num_steps=1, nthread=-1, seed=111413, restarp_prop=0.0):
    assert ptr.flags["C_CONTIGUOUS"]
    assert neighs.flags["C_CONTIGUOUS"]
    assert ptr.dtype == np.int32
    assert neighs.dtype == np.int32
    assert nodes.dtype == np.int32
    n = nodes.size
    walks = -np.ones((n * num_walks, (num_steps + 1)), dtype=np.int32, order="C")
    assert walks.flags["C_CONTIGUOUS"]

    librwalk.random_walk(
        nodes,
        ptr,
        neighs,
        n,
        num_walks,
        num_steps,
        seed,
        nthread,
        restarp_prop,
        np.reshape(walks, (walks.size,), order="C"),
    )

    return walks


def read_edgelist(fname, comments="#", delimiter=None):
    edges = np.genfromtxt(fname, comments=comments, delimiter=delimiter, defaultfmt="%d", dtype=np.int32)
    assert len(edges.shape) == 2
    assert edges.shape[1] == 2

    # Sort so smaller index comes first
    edges.sort(axis=1)
    n = np.amax(edges) + 1

    # Duplicate
    duplicated_edges = np.vstack((edges, edges[:, ::-1]))

    # Sort duplicated edges by first index
    _tmp = np.zeros((duplicated_edges.shape[0],), dtype=np.int64)
    _tmp += duplicated_edges[:, 0]
    _tmp *= np.iinfo(np.int32).max
    _tmp += duplicated_edges[:, 1]

    ind_sort = np.argsort(_tmp)
    sorted_edges = duplicated_edges[ind_sort, :]

    # Calculate degree and create ptr, neighs
    vals, counts = np.unique(sorted_edges[:, 0], return_counts=True)
    degs = np.zeros((n,), dtype=np.int32)
    degs[vals] = counts
    ptr = np.zeros((n + 1,), dtype=np.int32)
    ptr[1:] = np.cumsum(degs)
    neighs = np.copy(sorted_edges[:, 1])

    # Check ptr, neighs
    ptr.flags.writeable = False
    assert ptr.flags.owndata is True
    assert ptr.flags.aligned is True
    assert ptr.flags.c_contiguous is True

    neighs.flags.writeable = False
    assert neighs.flags.owndata is True
    assert neighs.flags.aligned is True
    assert neighs.flags.c_contiguous is True

    return ptr, neighs
