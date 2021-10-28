import numpy as np
from cogdl import pipeline

data = np.array([[0, 0], [0, 1], [0, 2], [1, 1], [1, 3], [1, 4], [2, 4], [2, 5], [2, 6]])
rec = pipeline("recommendation", model="lightgcn", data=data, epochs=10, evaluate_interval=1000, cpu=True)
print(rec([0]))

rec = pipeline("recommendation", model="lightgcn", dataset="ali", epochs=1, n_negs=1, evaluate_interval=1000)
print(rec([0]))
