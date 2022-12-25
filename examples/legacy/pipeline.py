from cogdl import pipeline

# print the statistics of datasets
stats = pipeline("dataset-stats")
stats(["cora", "citeseer"])

# visualize k-hop neighbors of seed in the dataset
visual = pipeline("dataset-visual")
visual("cora", seed=0, depth=3)

# load OAGBert model and perform inference
oagbert = pipeline("oagbert")
outputs = oagbert(["CogDL is developed by KEG, Tsinghua.", "OAGBert is developed by KEG, Tsinghua."])
