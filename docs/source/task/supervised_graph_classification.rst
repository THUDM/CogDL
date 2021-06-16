Supervised Graph Classification
================================

In this section, we will introduce the implementation "Graph classification task".

** Supervised Graph Classification Methods **

+-----------+-----------+--------+-----------------+
| Method    | Node      | Kernel | Reproducibility |
|           | Feature   |        |                 |
+===========+===========+========+=================+
| GIN       |   `√`     |        |  `√`            |
+-----------+-----------+--------+-----------------+
| DiffPool  |   `√`     |        |  `√`            |
+-----------+-----------+--------+-----------------+
| SortPool  |   `√`     |        |  `√`            |
+-----------+-----------+--------+-----------------+
| PATCH_SAN |   `√`     | `√`    |  `√`            |
+-----------+-----------+--------+-----------------+
| DGCNN     |   `√`     |        |  `√`            |
+-----------+-----------+--------+-----------------+
| SAGPool   |   `√`     |        |  `√`            |
+-----------+-----------+--------+-----------------+


**Task Design**

1. Set up "SupervisedGraphClassification" class, which has two specific parameters.

   * `degree-feature`: Use one-hot node degree as node feature, for datasets such as lmdb-binary and lmdb-multi, which don't have node features.
   * `gamma`: Multiplicative factor of learning rate decay.
   * `lr`: Learning rate.

2. Build dataset convert it to a list of `Data` defined in Cogdl. Specially, we reformat the data according to the input format of specific models. `generate_data` is implemented to convert dataset.

.. code-block:: python

   dataset = build_dataset(args)
   self.data = self.generate_data(dataset, args)

   def generate_data(self, dataset, args):
        if "ModelNet" in str(type(dataset).__name__):
            train_set, test_set = dataset.get_all()
            args.num_features = 3
            return {"train": train_set, "test": test_set}
       else:
           datalist = []
           if isinstance(dataset[0], Data):
               return dataset
           for idata in dataset:
               data = Data()
               for key in idata.keys:
                   data[key] = idata[key]
                   datalist.append(data)

           if args.degree_feature:
               datalist = node_degree_as_feature(datalist)
               args.num_features = datalist[0].num_features
           return datalist



3. Then we build model and can run `train` to train the model.

.. code-block:: python

   def train(self):
       for epoch in epoch_iter:
            self._train_step()
            val_acc, val_loss = self._test_step(split="valid")
            # ...
   	    return dict(Acc=test_acc)

   def _train_step(self):
       self.model.train()
       loss_n = 0
       for batch in self.train_loader:
           batch = batch.to(self.device)
           self.optimizer.zero_grad()
           output, loss = self.model(batch)
           loss_n += loss.item()
           loss.backward()
           self.optimizer.step()

   def _test_step(self, split):
       """split in ['train', 'test', 'valid']"""
       # ...
       return acc, loss

The overall implementation of GraphClassification is at (https://github.com/THUDM/cogdl/blob/master/cogdl/tasks/graph_classification.py).

**Create a model**

To create a model for task graph classification, the following functions have to be implemented.

1. `add_args(parser)`: add necessary hyper-parameters used in model.

.. code-block:: python

   @staticmethod
   def add_args(parser):
        parser.add_argument("--hidden-size", type=int, default=128)
        parser.add_argument("--num-layers", type=int, default=2)
        parser.add_argument("--lr", type=float, default=0.001)
        # ...

2. `build_model_from_args(cls, args)`: this function is called in 'task' to build model.

3. `split_dataset(cls, dataset, args)`: split train/validation/test data and return correspondent dataloader according to requirement of model.

.. code-block:: python

   def split_dataset(cls, dataset, args):
       random.shuffle(dataset)
       train_size = int(len(dataset) * args.train_ratio)
       test_size = int(len(dataset) * args.test_ratio)
       bs = args.batch_size
       train_loader = DataLoader(dataset[:train_size], batch_size=bs)
       test_loader = DataLoader(dataset[-test_size:], batch_size=bs)
       if args.train_ratio + args.test_ratio < 1:
            valid_loader = DataLoader(dataset[train_size:-test_size], batch_size=bs)
       else:
            valid_loader = test_loader
       return train_loader, valid_loader, test_loader

4. `forward`: forward propagation, and the return should be (predication, loss) or (prediction, None), respectively for training and test. Input parameters of `forward` is class `Batch`, which

.. code-block:: python

   def forward(self, batch):
    h = batch.x
    layer_rep = [h]
    for i in range(self.num_layers-1):
        h = self.gin_layers[i](h, batch.edge_index)
        h = self.batch_norm[i](h)
        h = F.relu(h)
        layer_rep.append(h)

    final_score = 0
    for i in range(self.num_layers):
    pooled = scatter_add(layer_rep[i], batch.batch, dim=0)
    final_score += self.dropout(self.linear_prediction[i](pooled))
    final_score = F.softmax(final_score, dim=-1)
    if batch.y is not None:
        loss = self.loss(final_score, batch.y)
        return final_score, loss
    return final_score, None


**Run**

To run GraphClassification, we can use the following command:

.. code-block:: python

    python scripts/train.py --task graph_classification --dataset proteins --model gin diffpool sortpool dgcnn --seed 0 1

Then We get experimental results like this:

============================ ===============
Variants                      Acc
============================ ===============
('proteins', 'gin')          0.7286±0.0598
('proteins', 'diffpool')     0.7530±0.0589
('proteins', 'sortpool')     0.7411±0.0269
('proteins', 'dgcnn')        0.6677±0.0355
('proteins', 'patchy_san')   0.7550±0.0812
============================ ===============
