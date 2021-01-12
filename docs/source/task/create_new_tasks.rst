Create new tasks
================

You can build a new task in the CogDL. The BaseTask class are: 

.. code-block:: python

    class BaseTask(object):
        @staticmethod
        def add_args(parser):
            """Add task-specific arguments to the parser."""
            pass

        def __init__(self, args):
            pass

        def train(self, num_epoch):
            raise NotImplementedError

You can create a subclass to implement 'train' method like CommunityDetection, which get representation of each node and apply clustering algorithm (K-means) to evaluate.

.. code-block:: python

    @register_task("community_detection")
    class CommunityDetection(BaseTask):
        """Community Detection task."""

        @staticmethod
        def add_args(parser):
            """Add task-specific arguments to the parser."""
            parser.add_argument("--hidden-size", type=int, default=128)
            parser.add_argument("--num-shuffle", type=int, default=5)

        def __init__(self, args):
            super(CommunityDetection, self).__init__(args)
            dataset = build_dataset(args)
            self.data = dataset[0]
  
            self.num_nodes, self.num_classes = self.data.y.shape
            self.label = np.argmax(self.data.y, axis=1)
            self.model = build_model(args)
            self.hidden_size = args.hidden_size
            self.num_shuffle = args.num_shuffle

        def train(self):
            G = nx.Graph()
            G.add_edges_from(self.data.edge_index.t().tolist())
            embeddings = self.model.train(G)

            clusters = [30, 50, 70]
            all_results = defaultdict(list)
            for num_cluster in clusters:
                for _ in range(self.num_shuffle):
                    model = KMeans(n_clusters=num_cluster).fit(embeddings)
                    nmi_score = normalized_mutual_info_score(self.label, model.labels_)
                    all_results[num_cluster].append(nmi_score)
                
            return dict(
                (
                    f"normalized_mutual_info_score {num_cluster}",
                    sum(all_results[num_cluster]) / len(all_results[num_cluster]),
                )
                for num_cluster in sorted(all_results.keys())
            )


After creating your own task, you could run the task on different models and dataset.
You can use 'build_model', 'build_dataset', 'build_task' method to build them with coresponding hyper-parameters.

.. code-block:: python

    from cogdl.tasks import build_task
    from cogdl.datasets import build_dataset
    from cogdl.models import build_model
    from cogdl.utils import build_args_from_dict

    def run_deepwalk_ppi():
        default_dict = {'hidden_size': 64, 'num_shuffle': 1, 'cpu': True}
        args = build_args_from_dict(default_dict)
        
        # model, dataset and task parameters
        args.model = 'spectral'
        args.dataset = 'ppi'
        args.task = 'community_detection'
        
        # build model, dataset and task
        dataset = build_dataset(args)
        model = build_model(args)
        task = build_task(args)
        
        # train model and get evaluate results
        ret = task.train()
        print(ret)
