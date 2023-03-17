import torch
from cogdl.data import Graph
from cogdl.datasets import NodeDataset, GraphDataset
from cogdl.experiments import experiment


class MyNodeClassificationDataset(NodeDataset):
    def __init__(self, path="data.pt"):
        super(MyNodeClassificationDataset, self).__init__(path)

    def process(self):
        num_nodes = 100
        num_edges = 300
        feat_dim = 30

        # load or generate your dataset
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        x = torch.randn(num_nodes, feat_dim)
        y = torch.randint(0, 2, (num_nodes,))

        # set train/val/test mask in node_classification task
        train_mask = torch.zeros(num_nodes).bool()
        train_mask[0 : int(0.3 * num_nodes)] = True
        val_mask = torch.zeros(num_nodes).bool()
        val_mask[int(0.3 * num_nodes) : int(0.7 * num_nodes)] = True
        test_mask = torch.zeros(num_nodes).bool()
        test_mask[int(0.7 * num_nodes) :] = True
        data = Graph(x=x, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
        torch.save(data, "mydata.pt")
        return data


class MyGraphClassificationDataset(GraphDataset):
    def __init__(self, path="data_graph.pt"):
        super(MyGraphClassificationDataset, self).__init__(path)

    def process(self):
        graphs = []
        for i in range(200):
            edges = torch.randint(0, 1000, (2, 30))
            label = torch.randint(0, 7, (1,))
            graphs.append(Graph(edge_index=edges, y=label))
        torch.save(graphs, self.path)
        return graphs


def test_customized_dataset():
    dataset = MyNodeClassificationDataset()
    assert isinstance(dataset[0], Graph)
    assert dataset[0].x.shape[0] == 100


def test_customized_graph_dataset():
    dataset = MyGraphClassificationDataset()
    result = experiment(model="gin", dataset=dataset, degree_node_features=True, epochs=10, cpu=True)
    result = list(result.values())[0][0]
    assert result["test_acc"] >= 0


def test_customized_graph_dataset_graph2vec():
    dataset = MyGraphClassificationDataset()
    result = experiment(model="graph2vec", dataset=dataset, degree_node_features=False, epochs=10, cpu=True)
    result = list(result.values())[0][0]
    assert result["acc"] >= 0


if __name__ == "__main__":
    test_customized_dataset()
    test_customized_graph_dataset()
    test_customized_graph_dataset_graph2vec()
