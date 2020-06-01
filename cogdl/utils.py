class ArgClass(object):
    def __init__(self):
        pass

def build_args_from_dict(dic):
    args = ArgClass()
    for key, value in dic.items():
        args.__setattr__(key, value)
    return args

if __name__ == "__main__":
    args = build_args_from_dict({'a': 1, 'b': 2})
    print(args.a, args.b)
