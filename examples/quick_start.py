from cogdl.tasks import build_task
from cogdl.options import get_default_args

args = get_default_args(task="node_classification", dataset="cora", model="gcn")
task = build_task(args)
ret = task.train()
