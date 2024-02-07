import torch
import tensorflow as tf
from torch import Tensor as t

def torch_model(tf_model):
    def get_output(torch_inp):
        torch_inp = torch_inp.permute(0,2,3,1)
        np_inp = torch_inp.numpy()
        tf_inp = tf.convert_to_tensor(np_inp)
        return t(tf_model.predict(tf_inp, verbose = 0))
    return get_output
