import torch
import tensorflow as tf

def torch_model(tf_model):
    def get_output(torch_inp):
        torch_inp = torch_inp.transpose(0,2,3,1)
        np_inp = torch_inp.numpy()
        tf_inp = tf.convert_to_tensor(np_inp)
        return torch.FloatTensor(tf_model.predict(tf_inp).numpy())
    return get_output
