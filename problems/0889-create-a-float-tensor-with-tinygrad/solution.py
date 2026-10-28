from tinygrad import Tensor, dtypes

def to_float_tensor(values):
    # TODO: return a tinygrad Tensor with dtype dtypes.float32 built from `values`
    return Tensor(values, dtype = dtypes.float32)
