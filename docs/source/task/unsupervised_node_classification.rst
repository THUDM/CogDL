Unsupervised Node Classification
================================

In this tutorial, we will introduce a important task, unsupervised node classification. In this task, we usually apply L2 normalized logisitic regression to train a classifier and use *F1-score* or *Accuracy* to measure the performance.

Unsupervied node classificatioin includes `network embedding` methods(DeepWalk, LINE, ProNE adn etc.) and `GNN self-supervied`
methods(DGI, GraphSAGE and etc.). In this section, we mainly introduce the part for `network embeddings` and the other
will be presented in next section `trainer`.

**Unsupervised Graph Embedding Methods**

+----------+----------+---------+---------------+-----------------+---------+
| Method   | Weighted | shallow | Matrix        | Reproducibility | GPU     |
|          |          | network | Factorization |                 | support |
+==========+==========+=========+===============+=================+=========+
|DeepWalk  |          |   `√`   |               |        `√`      |         |
+----------+----------+---------+---------------+-----------------+---------+
|LINE      |  `√`     |   `√`   |               |        `√`      |         |
+----------+----------+---------+---------------+-----------------+---------+
|Node2Vec  |      `√` |   `√`   |               |        `√`      |         |
+----------+----------+---------+---------------+-----------------+---------+
|NetMF     |      `√` |         |      `√`      |       `√`       |         |
+----------+----------+---------+---------------+-----------------+---------+
|NetSMF    |     `√`  |         |      `√`      |       `√`       |         |
+----------+----------+---------+---------------+-----------------+---------+
|HOPE      |      `√` |         |      `√`      |       `√`       |         |
+----------+----------+---------+---------------+-----------------+---------+
|GraRep    |      `√` |         |      `√`      |                 |         |
+----------+----------+---------+---------------+-----------------+---------+
|SDNE      |      `√` |   `√`   |               |        `√`      |  `√`    |
+----------+----------+---------+---------------+-----------------+---------+
|DNGR      |      `√` |   `√`   |               |                 |  `√`    |
+----------+----------+---------+---------------+-----------------+---------+
|ProNE     |      `√` |         |      `√`      |       `√`       |         |
+----------+----------+---------+---------------+-----------------+---------+

**Unsupervised Graph Neural Network Representation Learning Methods**

+------------+----------+-----------+-----------------+
| Method     | Sampling | Inductive | Reproducibility |
+============+==========+===========+=================+
| DGI        |          |           |    `√`          |
+------------+----------+-----------+-----------------+
| MVGRL      |  `√`     |   `√`     |    `√`          |
+------------+----------+-----------+-----------------+
| GRACE      |          |   `√`     |                 |
+------------+----------+-----------+-----------------+
| GraphSAGE  |  `√`     |   `√`     |    `√`          |
+------------+----------+-----------+-----------------+


First we define the `UnsupervisedNodeClassification`  class, which has two parameters `hidden-size`  and `num-shuffle` . `hidden-size`  represents the dimension of node representation, while `num-shuffle`  means the shuffle times in classifier.

.. code-block:: python

    @register_task("unsupervised_node_classification")
    class UnsupervisedNodeClassification(BaseTask):
        """Node classification task."""

        @staticmethod
        def add_args(parser):
            """Add task-specific arguments to the parser."""
            # fmt: off
            parser.add_argument("--hidden-size", type=int, default=128)
            parser.add_argument("--num-shuffle", type=int, default=5)
            # fmt: on

        def __init__(self, args):
            super(UnsupervisedNodeClassification, self).__init__(args)


Then we can build dataset according to input graph's type, and get `self.label_matrix`.

.. code-block:: python

        dataset = build_dataset(args)
        self.data = dataset[0]
        if issubclass(dataset.__class__.__bases__[0], InMemoryDataset):
            self.num_nodes = self.data.y.shape[0]
            self.num_classes = dataset.num_classes
            self.label_matrix = np.zeros((self.num_nodes, self.num_classes), dtype=int)
            self.label_matrix[range(self.num_nodes), self.data.y] = 1
            self.data.edge_attr = self.data.edge_attr.t()
        else:
            self.label_matrix = self.data.y
            self.num_nodes, self.num_classes = self.data.y.shape

After that, we can build model and run `model.train(G)` to obtain node representation.

.. code-block:: python

        self.model = build_model(args)
        self.is_weighted = self.data.edge_attr is not None


        def train(self):
            G = nx.Graph()
            if self.is_weighted:
                edges, weight = (
                    self.data.edge_index.t().tolist(),
                    self.data.edge_attr.tolist(),
                )
                G.add_weighted_edges_from(
                    [(edges[i][0], edges[i][1], weight[0][i]) for i in range(len(edges))]
                )
            else:
                G.add_edges_from(self.data.edge_index.t().tolist())
            embeddings = self.model.train(G)


The spectral propagation in ProNE/ProNE++ can improve the quality of representation learned from other methods,
so we can use `enhance_emb` to enhance performance. ProNE++ automatically searches for the best graph filter to help improve
the embedding.

.. code-block:: python

            if self.enhance is True:
                embeddings = self.enhance_emb(G, embeddings)

When the embeddings are obtained, we can save them at `self.save_dir`.


At last, we evaluate embedding via run `num_shuffle` times classification under different training ratio with `features_matrix` and `label_matrix`.

.. code-block:: python

    def _evaluate(self, features_matrix, label_matrix, num_shuffle):
        # shuffle, to create train/test groups
        shuffles = []
        for _ in range(num_shuffle):
            shuffles.append(skshuffle(features_matrix, label_matrix))

        # score each train/test group
        all_results = defaultdict(list)
        training_percents = [0.1, 0.3, 0.5, 0.7, 0.9]
        for train_percent in training_percents:
            for shuf in shuffles:

In each shuffle, split data into two parts(training and testing) and use `LogisticRegression` to evaluate.

.. code-block:: python

    # ... shuffle to generate train/test set X_train/X_test, y_train/y_test

    clf = TopKRanker(LogisticRegression())
    clf.fit(X_train, y_train)

    # find out how many labels should be predicted
    top_k_list = list(map(int, y_test.sum(axis=1).T.tolist()[0]))
    preds = clf.predict(X_test, top_k_list)
    result = f1_score(y_test, preds, average="micro")
    all_results[train_percent].append(result)

Node in graph may have multiple labels, so we conduct multilbel classification built from TopKRanker.

.. code-block:: python

    from sklearn.multiclass import OneVsRestClassifier

    class TopKRanker(OneVsRestClassifier):
        def predict(self, X, top_k_list):
            assert X.shape[0] == len(top_k_list)
            probs = np.asarray(super(TopKRanker, self).predict_proba(X))
            all_labels = sp.lil_matrix(probs.shape)

            for i, k in enumerate(top_k_list):
                probs_ = probs[i, :]
                labels = self.classes_[probs_.argsort()[-k:]].tolist()
                for label in labels:
                    all_labels[i, label] = 1
            return all_labels


Finally, we get the results of Micro-F1 score under different training ratio for different models on datasets.

``Cogdl`` supports evaluating the trained embeddings ignoring the training process. With `--load-emb-path` set to the path of your result,
``Cogdl`` will skip the training and directly evalute the embeddings.

The overall implementation of `UnsupervisedNodeClassification` is at (https://github.com/THUDM/cogdl/blob/master/cogdl/tasks/unsupervised_node_classification.py).

To run UnsupervisedNodeClassification, we can use following instruction:

.. code-block:: python

    python scripts/train.py --task unsupervised_node_classification --dataset ppi wikipedia --model deepwalk prone -seed 0 1


Then We get experimental results like this:

=========================  ==============  ==============  ==============  ==============  ==============
Variant                    Micro-F1 0.1    Micro-F1 0.3    Micro-F1 0.5    Micro-F1 0.7    Micro-F1 0.9
=========================  ==============  ==============  ==============  ==============  ==============
('ppi', 'deepwalk')        0.1547±0.0002   0.1846±0.0002   0.2033±0.0015   0.2161±0.0009   0.2243±0.0018
('ppi', 'prone')           0.1777±0.0016   0.2214±0.0020   0.2397±0.0015   0.2486±0.0022   0.2607±0.0096
('wikipedia', 'deepwalk')  0.4255±0.0027   0.4712±0.0005   0.4916±0.0011   0.5011±0.0017   0.5166±0.0043
('wikipedia', 'prone')     0.4834±0.0009   0.5320±0.0020   0.5504±0.0045   0.5586±0.0022   0.5686±0.0072
=========================  ==============  ==============  ==============  ==============  ==============
