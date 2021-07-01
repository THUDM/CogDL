import heapq
import multiprocessing as mp
import random

import numpy as np
import torch
from cogdl.datasets import build_dataset
from cogdl.models import build_model
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from . import BaseTask, register_task


def recall(rank, ground_truth, N):
    return len(set(rank[:N]) & set(ground_truth)) / float(len(set(ground_truth)))


def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k]
    return np.mean(r)


def average_precision(r, cut):
    """Score is average precision (area under PR curve)
    Relevance is binary (nonzero is relevant).
    Returns:
        Average precision
    """
    r = np.asarray(r)
    out = [precision_at_k(r, k + 1) for k in range(cut) if r[k]]
    if not out:
        return 0.0
    return np.sum(out) / float(min(cut, np.sum(r)))


def mean_average_precision(rs):
    """Score is mean average precision
    Relevance is binary (nonzero is relevant).
    Returns:
        Mean average precision
    """
    return np.mean([average_precision(r) for r in rs])


def dcg_at_k(r, k, method=1):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError("method must be 0 or 1.")
    return 0.0


def ndcg_at_k(r, k, ground_truth, method=1):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Normalized discounted cumulative gain

        Low but correct defination
    """
    GT = set(ground_truth)
    if len(GT) > k:
        sent_list = [1.0] * k
    else:
        sent_list = [1.0] * len(GT) + [0.0] * (k - len(GT))
    dcg_max = dcg_at_k(sent_list, k, method)
    if not dcg_max:
        return 0.0
    return dcg_at_k(r, k, method) / dcg_max


def recall_at_k(r, k, all_pos_num):
    # if all_pos_num == 0:
    #     return 0
    r = np.asfarray(r)[:k]
    return np.sum(r) / all_pos_num


def hit_at_k(r, k):
    r = np.array(r)[:k]
    if np.sum(r) > 0:
        return 1.0
    else:
        return 0.0


def F1(pre, rec):
    if pre + rec > 0:
        return (2.0 * pre * rec) / (pre + rec)
    else:
        return 0.0


def AUC(ground_truth, prediction):
    try:
        res = roc_auc_score(y_true=ground_truth, y_score=prediction)
    except Exception:
        res = 0.0
    return res


def ranklist_by_heapq(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = 0.0
    return r, auc


def get_auc(item_score, user_pos_test):
    item_score = sorted(item_score.items(), key=lambda kv: kv[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]
    posterior = [x[1] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = AUC(ground_truth=r, prediction=posterior)
    return auc


def ranklist_by_sorted(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = get_auc(item_score, user_pos_test)
    return r, auc


def get_performance(user_pos_test, r, auc, Ks):
    precision, recall, ndcg, hit_ratio = [], [], [], []

    for K in Ks:
        precision.append(precision_at_k(r, K))
        recall.append(recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(ndcg_at_k(r, K, user_pos_test))
        hit_ratio.append(hit_at_k(r, K))

    return {
        "recall": np.array(recall),
        "precision": np.array(precision),
        "ndcg": np.array(ndcg),
        "hit_ratio": np.array(hit_ratio),
        "auc": auc,
    }


def get_feed_dict(train_entity_pairs, train_pos_set, start, end, n_items, n_negs=1, device="cpu"):
    def sampling(user_item, train_set, n):
        neg_items = []
        for user, _ in user_item.cpu().numpy():
            user = int(user)
            negitems = []
            for i in range(n):  # sample n times
                while True:
                    negitem = random.choice(range(n_items))
                    if negitem not in train_set[user]:
                        break
                negitems.append(negitem)
            neg_items.append(negitems)
        return neg_items

    feed_dict = {}
    entity_pairs = train_entity_pairs[start:end]
    feed_dict["users"] = entity_pairs[:, 0]
    feed_dict["pos_items"] = entity_pairs[:, 1]
    feed_dict["neg_items"] = torch.LongTensor(sampling(entity_pairs, train_pos_set, n_negs * 1)).to(device)
    return feed_dict


def early_stopping(log_value, best_value, stopping_step, expected_order="acc", flag_step=100):
    # early stopping strategy:
    assert expected_order in ["acc", "dec"]

    if (expected_order == "acc" and log_value >= best_value) or (expected_order == "dec" and log_value <= best_value):
        stopping_step = 0
        best_value = log_value
    else:
        stopping_step += 1

    if stopping_step >= flag_step:
        print("Early stopping is trigger at step: {} log:{}".format(flag_step, log_value))
        should_stop = True
    else:
        should_stop = False
    return best_value, stopping_step, should_stop


def test_one_user(x):
    rating = x[0]
    training_items = x[1]
    user_pos_test = x[2]
    Ks = x[3]
    n_items = x[4]

    all_items = set(range(0, n_items))
    test_items = list(all_items - set(training_items))

    r, auc = ranklist_by_heapq(user_pos_test, test_items, rating, Ks)

    return get_performance(user_pos_test, r, auc, Ks)


@register_task("recommendation")
class Recommendation(BaseTask):
    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument("--evaluate-interval", type=int, default=5)
        parser.add_argument("--max-epoch", type=int, default=3000)
        parser.add_argument("--patience", type=int, default=10)
        parser.add_argument('--batch_size', type=int, default=2048, help='batch size')
        parser.add_argument("--lr", type=float, default=0.001)
        parser.add_argument("--weight-decay", type=float, default=0)
        parser.add_argument("--num-workers", type=int, default=4)
        parser.add_argument('--Ks', default=[20], type=int, nargs='+', metavar='N',
                            help='Output sizes of every layer')
        # fmt: on

    def __init__(self, args, dataset=None, model=None):
        super(Recommendation, self).__init__(args)

        self.device = "cpu" if not torch.cuda.is_available() or args.cpu else args.device_id[0]
        dataset = build_dataset(args) if dataset is None else dataset
        self.data = dataset[0]
        self.data.apply(lambda x: x.to(self.device))

        args.n_users = self.data.n_params["n_users"]
        args.n_items = self.data.n_params["n_items"]
        args.adj_mat = self.data.norm_mat
        model = build_model(args) if model is None else model

        self.model = model.to(self.device)
        self.model.set_device(self.device)

        self.max_epoch = args.max_epoch
        self.patience = args.patience
        self.n_negs = args.n_negs
        self.batch_size = args.batch_size
        self.evaluate_interval = args.evaluate_interval
        self.Ks = args.Ks
        self.num_workers = args.num_workers

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def train(self, unittest=False):
        stopping_step = 0
        best_value = 0
        should_stop = False
        best_ret = None

        print("start training ...")
        for epoch in range(self.max_epoch):
            loss = self._train_step()

            if (epoch + 1) % self.evaluate_interval == 0:
                self.model.eval()
                test_ret = self._test_step(split="test", unittest=unittest)
                test_ret = [
                    epoch,
                    loss,
                    test_ret["recall"],
                    test_ret["ndcg"],
                    test_ret["precision"],
                    test_ret["hit_ratio"],
                ]
                print(test_ret)

                if self.data.user_dict["valid_user_set"] is None:
                    valid_ret = test_ret
                else:
                    valid_ret = self._test_step(split="valid", unittest=unittest)
                    valid_ret = [
                        epoch,
                        loss,
                        valid_ret["recall"],
                        valid_ret["ndcg"],
                        valid_ret["precision"],
                        valid_ret["hit_ratio"],
                    ]
                    print(valid_ret)

                if valid_ret[2] >= best_value:
                    stopping_step = 0
                    best_value = valid_ret[2]
                    best_ret = test_ret
                    if self.save_path is not None:
                        torch.save(self.model.state_dict(), self.save_path)
                else:
                    stopping_step += 1

                if stopping_step >= self.patience:
                    print("Early stopping is trigger at step: {} log:{}".format(epoch, valid_ret[2]))
                    should_stop = True
                else:
                    should_stop = False

                if should_stop:
                    break
            else:
                # logging.info('training loss at epoch %d: %f' % (epoch, loss.item()))
                print("raining loss at epoch %d: %.4f" % (epoch, loss))

        print("Stopping at %d, recall@20:%.4f" % (epoch, best_value))

        if best_ret is not None:
            Recall, NDCG = best_ret[2], best_ret[3]
        else:
            Recall = NDCG = 0.0
        return dict(Recall=Recall, NDCG=NDCG)

    def _train_step(self):
        # shuffle training data
        train_cf_ = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in self.data.train_cf], np.int32))
        index = np.arange(len(train_cf_))
        np.random.shuffle(index)
        train_cf_ = train_cf_[index].to(self.device)

        """training"""
        self.model.train()
        loss, s = 0, 0
        for s in tqdm(range(0, len(self.data.train_cf), self.batch_size)):
            batch = get_feed_dict(
                train_cf_,
                self.data.user_dict["train_user_set"],
                s,
                s + self.batch_size,
                self.data.n_params["n_items"],
                self.n_negs,
                self.device,
            )

            batch_loss, _, _ = self.model(batch)

            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()

            loss += batch_loss.item()

        return loss

    def _test_step(self, split="val", unittest=False):
        """testing"""

        result = {
            "precision": np.zeros(len(self.Ks)),
            "recall": np.zeros(len(self.Ks)),
            "ndcg": np.zeros(len(self.Ks)),
            "hit_ratio": np.zeros(len(self.Ks)),
            "auc": 0.0,
        }

        n_items = self.data.n_params["n_items"]
        if unittest:
            n_items = n_items // 100

        user_dict = self.data.user_dict
        train_user_set = user_dict["train_user_set"]
        if split == "test":
            test_user_set = user_dict["test_user_set"]
        else:
            test_user_set = user_dict["valid_user_set"]
            if test_user_set is None:
                test_user_set = user_dict["test_user_set"]

        pool = mp.Pool(self.num_workers)

        u_batch_size = self.batch_size
        i_batch_size = self.batch_size

        test_users = list(test_user_set.keys())
        n_test_users = len(test_users) if not unittest else len(test_users) // 1000
        n_user_batchs = n_test_users // u_batch_size + 1

        count = 0

        user_gcn_emb, item_gcn_emb = self.model.generate()

        for u_batch_id in range(n_user_batchs):
            start = u_batch_id * u_batch_size
            end = (u_batch_id + 1) * u_batch_size

            user_list_batch = test_users[start:end]
            user_batch = torch.LongTensor(np.array(user_list_batch)).to(self.device)
            u_g_embeddings = user_gcn_emb[user_batch]

            # batch-item test
            n_item_batchs = n_items // i_batch_size + 1
            rate_batch = np.zeros(shape=(len(user_batch), n_items))

            i_count = 0
            for i_batch_id in range(n_item_batchs):
                i_start = i_batch_id * i_batch_size
                i_end = min((i_batch_id + 1) * i_batch_size, n_items)

                item_batch = torch.LongTensor(np.array(range(i_start, i_end))).view(i_end - i_start).to(self.device)
                i_g_embddings = item_gcn_emb[item_batch]

                i_rate_batch = self.model.rating(u_g_embeddings, i_g_embddings).detach().cpu()

                rate_batch[:, i_start:i_end] = i_rate_batch
                i_count += i_rate_batch.shape[1]

            assert i_count == n_items

            user_batch_rating_uid = []  # zip(rate_batch, user_list_batch, [self.Ks] * len(rate_batch))
            for rate, user in zip(rate_batch, user_list_batch):
                user_batch_rating_uid.append(
                    [
                        rate,
                        train_user_set[user] if user in train_user_set else [],
                        test_user_set[user],
                        self.Ks,
                        n_items,
                    ]
                )
            batch_result = pool.map(test_one_user, user_batch_rating_uid)
            count += len(batch_result)

            for re in batch_result:
                result["precision"] += re["precision"] / n_test_users
                result["recall"] += re["recall"] / n_test_users
                result["ndcg"] += re["ndcg"] / n_test_users
                result["hit_ratio"] += re["hit_ratio"] / n_test_users
                result["auc"] += re["auc"] / n_test_users

        pool.close()
        return result
