from cogdl.configurations import result_table

dataset = ["cora", "citeseer", "pubmed", "cora_geom", "citeseer_geom", "pubmed_geom", "chameleon", "squirrel", "film", "cornell", "texas", "wisconsin", "ogbn-arxiv"]
model = ["mlp", "gcn", "gcnii", "ppnp"]

result_table(dataset=dataset, model=model)