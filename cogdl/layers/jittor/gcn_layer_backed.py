
class BackedMixin:
    def __call__(self, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        if return_tensors == "pt":
            return self.torch_call()
        elif return_tensors == "jt":
            return self.jittor_call()
        elif return_tensors == "np":
            return self.numpy_call()
        else:
            raise ValueError(f"Framework '{return_tensors}' not recognized!")

    def torch_call():
        pass

    def jittor_call():
        pass