from .. import BaseModel, register_model
from cogdl.layers.strategies_layers import (
    ContextPredictTrainer,
    MaskTrainer,
    InfoMaxTrainer,
    SupervisedTrainer,
)


@register_model("stpgnn")
class stpgnn(BaseModel):
    """
    Implementation of models in paper `"Strategies for Pre-training Graph Neural Networks"`. <https://arxiv.org/abs/1905.12265>
    """

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument("--batch-size", type=int, default=256)
        parser.add_argument("--num-layers", type=int, default=5)
        parser.add_argument("--hidden-size", type=int, default=300)
        parser.add_argument("--JK", type=str, default="last")
        parser.add_argument("--output-model-file", type=str, default="./saved")
        parser.add_argument("--num-workers", type=int, default=4)
        parser.add_argument("--pretrain-task", type=str, default="infomax")
        parser.add_argument("--finetune", action="store_true")
        parser.add_argument("--dropout", type=float, default=0.5)
        # fmt: on
        ContextPredictTrainer.add_args(parser)
        MaskTrainer.add_args(parser)
        SupervisedTrainer.add_args(parser)

    @classmethod
    def build_model_from_args(cls, args):
        return cls(args)

    def __init__(self, args):
        super(stpgnn, self).__init__()
        if args.pretrain_task == "infomax":
            self.trainer = InfoMaxTrainer(args)
        elif args.pretrain_task == "context":
            self.trainer = ContextPredictTrainer(args)
        elif args.pretrain_task == "mask":
            self.trainer = MaskTrainer(args)
        elif args.pretrain_task == "supervised":
            self.trainer = SupervisedTrainer(args)
        else:
            raise NotImplementedError
