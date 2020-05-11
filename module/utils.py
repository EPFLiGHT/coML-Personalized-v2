import tensorflow as tf
from .custom_types import *

def f1():
    return ClientMessage(0,0)

def make_ds(X, y, batch_size):
    return tf.data.Dataset.from_tensor_slices({'features': X, 'label':y})\
                 .batch(batch_size)