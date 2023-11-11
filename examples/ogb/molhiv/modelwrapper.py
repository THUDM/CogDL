import torch

from cogdl.wrappers.model_wrapper import ModelWrapper

from torch.optim.lr_scheduler import StepLR, LambdaLR

def PolynomialDecayLR(step_count, warmup_updates, tot_updates, begin_lr, end_lr, power):
    # print ('step_count, warmup_updates, tot_updates, begin_lr, end_lr, power', step_count, warmup_updates, tot_updates, begin_lr, end_lr, power)
    step_count += 1
    if step_count <= warmup_updates:
        warmup_factor = step_count / float(warmup_updates)
        lr = warmup_factor * begin_lr
    elif step_count >= tot_updates:
        lr = end_lr
    else:
        pct_remaining = 1 - (step_count - warmup_updates) / (tot_updates - warmup_updates)
        lr = (begin_lr - end_lr) * pct_remaining ** power + end_lr
    return lr


class GraphClassificationModelWrapper(ModelWrapper):
    def __init__(self, model, optimizer_cfg, metric_name, scheduler_type, scheduler_round, num_iterations, warmup_epochs, epochs, lr, end_lr):
        super(GraphClassificationModelWrapper, self).__init__()
        self.model = model
        self.optimizer_cfg = optimizer_cfg
        self.metric_name = metric_name
        
        self.scheduler_type = scheduler_type
        self.scheduler_round = scheduler_round
        self.num_iterations = num_iterations
        self.warmup_epochs, self.epochs = warmup_epochs, epochs
        self.begin_lr, self.end_lr = lr, end_lr

    def train_step(self, batch):
        pred = self.model(batch)
        target = batch.y.view(pred.shape[0], -1)
        loss = self.default_loss_fn(pred, target)

        return loss

    def val_step(self, batch):
        pred = self.model(batch)
        target = batch.y.view(pred.shape[0], -1)
        val_loss = self.default_loss_fn(pred, target)
        metric = self.evaluate(pred, target, metric="auto")

        self.note("val_loss", val_loss)
        self.note("val_metric", metric)

    def test_step(self, batch):
        pred = self.model(batch)
        target = batch.y.view(pred.shape[0], -1)
        test_loss = self.default_loss_fn(pred, target)
        metric = self.evaluate(pred, target, metric="auto")

        self.note("test_loss", test_loss)
        self.note("test_metric", metric)

    def setup_optimizer(self):
        cfg = self.optimizer_cfg

        if self.scheduler_type == 'StepLR':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
            scheduler = StepLR(optimizer, step_size=30, gamma=0.25)
            return optimizer, scheduler
        
        if self.scheduler_type == 'PolynomialDecayLR':
            if self.scheduler_round == 'iteration':
                
                optimizer = torch.optim.Adam(self.model.parameters(), lr=1.0, weight_decay=cfg["weight_decay"])
                warmup_updates = self.warmup_epochs * self.num_iterations
                tot_updates = self.epochs * self.num_iterations
                begin_lr, end_lr, power = self.begin_lr, self.end_lr, 1.0
                scheduler = LambdaLR(
                    optimizer, 
                    lr_lambda=lambda x: PolynomialDecayLR(
                        x, warmup_updates, tot_updates, begin_lr, end_lr, power
                    )
                )
                scheduler.scheduler_round = 'iteration'
                return optimizer, scheduler
            
        optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        return optimizer

    
    def set_early_stopping(self):
        if self.metric_name == 'mae':
            return "val_metric", "<" 
        return "val_metric", ">"
