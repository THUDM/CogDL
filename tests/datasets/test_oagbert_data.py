from cogdl.data import dataset
from cogdl.datasets import build_dataset
from cogdl.utils import build_args_from_dict

def test_oagbert_dataset():
    args = build_args_from_dict({'dataset': 'aff30'})
    data = build_dataset(args)
    assert isinstance(data.get_data(), dict)
    assert isinstance(data.get_candidates(), list)

if __name__ == '__main__':
    test_oagbert_dataset()