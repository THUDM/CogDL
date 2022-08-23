import torch
import torch.nn as nn
from .. import ModelWrapper
from tqdm import tqdm
import torch.nn.functional as F


class TripleModelWrapper(ModelWrapper):
    @classmethod
    def add_args(self, parser):
        # fmt: off
        parser.add_argument("--negative_adversarial_sampling", default=False)
        parser.add_argument("--negative_sample_size", type=int , default=128)
        parser.add_argument("--uni_weight", action="store_true", help="Otherwise use subsampling weighting like in word2vec")
        parser.add_argument("--regularization", default=1e-9, type=float)
        parser.add_argument('--lr', default=0.001, type=float)
        parser.add_argument("--adversarial_temperature", default=1.0, type=float)
        parser.add_argument("--save-emb-path", default="./checkpoints")
        parser.add_argument("--eval-step", type=int, default=501)
        parser.add_argument("--do_test", default=True)
        parser.add_argument("--do_valid", default=True)

        self.parser = parser
        return self.parser
        # fmt: on

    def __init__(self, model, optimizer_cfg):
        super(TripleModelWrapper, self).__init__()

        self.model = model
        self.optimizer_cfg = optimizer_cfg
        self.args = self.parser.parse_args()

    def train_step(self, subgraph):
        """
        A single train step. Apply back-propation and return the loss
        """
        train_iterator = subgraph
        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)

        positive_sample = positive_sample.to(self.device)
        negative_sample = negative_sample.to(self.device)
        subsampling_weight = subsampling_weight.to(self.device)

        negative_score = self.model((positive_sample, negative_sample), mode=mode)

        if self.args.negative_adversarial_sampling:
            # In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (F.softmax(negative_score * self.args.adversarial_temperature, dim=1).detach() * F.logsigmoid(-negative_score)).sum(dim=1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim=1)

        positive_score = self.model(positive_sample)

        positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

        if self.args.uni_weight:
            positive_sample_loss = -positive_score.mean()
            negative_sample_loss = -negative_score.mean()
        else:
            positive_sample_loss = -(subsampling_weight * positive_score).sum() / subsampling_weight.sum()
            negative_sample_loss = -(subsampling_weight * negative_score).sum() / subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss) / 2
        if self.args.regularization != 0.0:
            # Use L3 regularization for ComplEx and DistMult
            regularization = self.args.regularization * (
                self.model.entity_embedding.norm(p=3) ** 3 + self.model.relation_embedding.norm(p=3).norm(p=3) ** 3
            )
            loss = loss + regularization

        return loss

    def test_step(self, subgraph):
        print("Test Dataset:")
        metrics = self.eval_step(subgraph)
        return dict(mrr=metrics["MRR"], mr=metrics["MR"], hits1=metrics["HITS@1"], hits3=metrics["HITS@3"], hits10=metrics["HITS@10"])

    def val_step(self, subgraph):
        print("Val Dataset:")
        metrics = self.eval_step(subgraph)
        return dict(mrr=metrics["MRR"], mr=metrics["MR"], hits1=metrics["HITS@1"], hits3=metrics["HITS@3"], hits10=metrics["HITS@10"])

    def eval_step(self, subgraph):        
        test_dataloader_head, test_dataloader_tail = subgraph
        logs = []
        test_dataset_list = [test_dataloader_head, test_dataloader_tail]
        for test_dataset in test_dataset_list:
            pbar = tqdm(test_dataset)
            for positive_sample, negative_sample, filter_bias, mode in pbar:
                pbar.set_description("Evaluating the model: Use mode({})".format(mode))
                positive_sample = positive_sample.to(self.device)
                negative_sample = negative_sample.to(self.device)
                filter_bias = filter_bias.to(self.device)

                batch_size = positive_sample.size(0)

                score = self.model((positive_sample, negative_sample), mode)
                score += filter_bias

                # Explicitly sort all the entities to ensure that there is no test exposure bias
                argsort = torch.argsort(score, dim=1, descending=True)

                if mode == "head-batch":
                    positive_arg = positive_sample[:, 0]
                elif mode == "tail-batch":
                    positive_arg = positive_sample[:, 2]
                else:
                    raise ValueError("mode %s not supported" % mode)

                for i in range(batch_size):
                    # Notice that argsort is not ranking
                    ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                    assert ranking.size(0) == 1

                    # ranking + 1 is the true ranking used in evaluation metrics
                    ranking = 1 + ranking.item()
                    logs.append(
                        {
                            "MRR": 1.0 / ranking,
                            "MR": float(ranking),
                            "HITS@1": 1.0 if ranking <= 1 else 0.0,
                            "HITS@3": 1.0 if ranking <= 3 else 0.0,
                            "HITS@10": 1.0 if ranking <= 10 else 0.0,
                        }
                    )

        metrics = {}
        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs]) / len(logs)
        print("The Dataset metrics:", metrics)
        return metrics

    def setup_optimizer(self):
        lr, weight_decay = self.optimizer_cfg["lr"], self.optimizer_cfg["weight_decay"]
        return torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

    def set_early_stopping(self):
        return "mrr", ">"
