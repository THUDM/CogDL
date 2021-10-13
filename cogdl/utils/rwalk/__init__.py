"""

"""
import numpy as np
import numpy.ctypeslib as npct
from ctypes import c_float, c_int
from os.path import dirname

array_1d_int = npct.ndpointer(dtype=np.int32, ndim=1, flags="CONTIGUOUS")

librwalk = npct.load_library("librwalk", dirname(__file__))

# print("rwalk: Loading library from: {}".format(dirname(__file__)))
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


def random_walk(nodes, ptr, neighs, num_walks=1, num_steps=1, nthread=-1, seed=111413, restart_prob=0.0):
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
        restart_prob,
        np.reshape(walks, (walks.size,), order="C"),
    )

    return walks
