import os
os.environ['CogDLBACKEND']="jittor"
import jittor
jittor.flags.use_cuda = 1
import cogdl
from cogdl import experiment

experiment(dataset="cora", model="grand",epochs=500)
experiment(dataset="cora", model="gcn",epochs=500)
experiment(dataset="cora", model="gcnii",epochs=500)
experiment(dataset="cora", model="graphsage",epochs=500)
experiment(dataset="cora", model="dgi",epochs=500)
experiment(dataset="cora", model="mvgrl",epochs=500)
experiment(dataset="cora", model="gat",epochs=500)
experiment(dataset="cora", model="drgat",epochs=500)
experiment(dataset="cora", model="grace",epochs=500)