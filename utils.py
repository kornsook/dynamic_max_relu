import torch
import tensorflow as tf

def torch_model(tf_model):
    def get_output(torch_inp):
        np_inp = torch_inp.numpy()
        tf_inp = tf.convert_to_tensor(np_inp)
        return tf_model.predict(tf_inp)
    return get_output
