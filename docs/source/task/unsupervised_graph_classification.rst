Unsupervised Graph Classification
==================================
In this section, we will introduce the implementation "Unsupervised graph classification task".

**Unsupervised Graph Classificaton Methods**

+-----------+-----------+--------+-----------------+
| Method    | Node      | Kernel | Reproducibility |
|           | Feature   |        |                 |
+===========+===========+========+=================+
| InfoGraph |   `√`     |        |  `√`            |
+-----------+-----------+--------+-----------------+
| DGK       |           |  `√`   |  `√`            |
+-----------+-----------+--------+-----------------+
| Graph2Vec |           |  `√`   |  `√`            |
+-----------+-----------+--------+-----------------+
| HGP_SL    |   `√`     |        |  `√`            |
+-----------+-----------+--------+-----------------+


**Task Design**

1. Set up "UnsupervisedGraphClassification" class, which has two specific parameters.

   * `num-shuffle` : Shuffle times in classifier
   * `degree-feature`: Use one-hot node degree as node feature, for datasets such as lmdb-binary and lmdb-multi, which don't have node features.
   * `lr`: learning

.. code-block:: python

   @register_task("unsupervised_graph_classification")
   class UnsupervisedGraphClassification(BaseTask):
       r"""Unsupervised graph classification"""
       @staticmethod
       def add_args(parser):
           """Add task-specific arguments to the parser."""
           # fmt: off
           parser.add_argument("--num-shuffle", type=int, default=10)
           parser.add_argument("--degree-feature", dest="degree_feature", action="store_true")
           parser.add_argument("--lr", type=float, default=0.001)
           # fmt: on
      def __init__(self, args):
        # ...

2. Build dataset and convert it to a list of `Data` defined in Cogdl.

.. code-block:: python

   dataset = build_dataset(args)
   self.label = np.array([data.y for data in dataset])
   self.data = [
   	Data(x=data.x, y=data.y, edge_index=data.edge_index, edge_attr=data.edge_attr,
   		pos=data.pos).apply(lambda x:x.to(self.device))
   		for data in dataset
   ]

3. Then we build model and can run `train` to train the model and obtain graph representation. In this part, the training process of shallow models and deep models are implemented separately.

.. code-block:: python

   self.model = build_model(args)
   self.model = self.model.to(self.device)

   def train(self):
        if self.use_nn:
           # deep neural network models
   		epoch_iter = tqdm(range(self.epoch))
           for epoch in epoch_iter:
               loss_n = 0
               for batch in self.data_loader:
                   batch = batch.to(self.device)
                   predict, loss = self.model(batch.x, batch.edge_index, batch.batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_n += loss.item()
        # ...
       else:
          # shallow models
           prediction, loss = self.model(self.data)
           label = self.label



4. When graph representation is obtained, we evaluate the embedding with `SVM` via running `num_shuffle` times under different training ratio. You can also call `save_emb` to save the embedding.

.. code-block:: python

   return self._evaluate(prediction, label)
   def _evaluate(self, embedding, labels):
       # ...
       for training_percent in training_percents:
            for shuf in shuffles:
               # ...
               clf = SVC()
               clf.fit(X_train, y_train)
               preds = clf.predict(X_test)
               # ...




The overall implementation of UnsupervisedGraphClassification is at (https://github.com/THUDM/cogdl/blob/master/cogdl/tasks/unsupervised_graph_classification.py).

**Create a model**

To create a model for task unsupervised graph classification, the following functions have to be implemented.

1. `add_args(parser)`: add necessary hyper-parameters used in model.

.. code-block:: python

   @staticmethod
   def add_args(parser):
     parser.add_argument("--hidden-size", type=int, default=128)
     parser.add_argument("--nn", type=bool, default=False)
     parser.add_argument("--lr", type=float, default=0.001)
     # ...

2. `build_model_from_args(cls, args)`: this function is called in 'task' to build model.

3. `forward`: For shallow models, this function runs as training process of model and will be called only once; For deep neural network models,  this function is actually the forward propagation process and will be called many times.

.. code-block:: python

   # shallow model
   def forward(self, graphs):
        # ...
       self.model = Doc2Vec(
           self.doc_collections,
   		...
       )
       vectors = np.array([self.model["g_"+str(i)] for i in range(len(graphs))])
       return vectors, None

**Run**

To run UnsupervisedGraphClassification, we can use the following command:

.. code-block:: python
    
    python scripts/train.py --task unsupervised_graph_classification --dataset proteins --model dgk graph2vec

Then we get experimental results like this:

=========================== =================
Variant                      Acc
=========================== =================
('proteins', 'dgk')          0.7259±0.0118
('proteins', 'graph2vec')    0.7330±0.0043
('proteins', 'infograph')    0.7393±0.0070
=========================== =================
