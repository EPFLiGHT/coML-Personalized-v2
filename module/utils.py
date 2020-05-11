from enum import Enum
from typing import Sequence, Tuple
import random

import numpy as np
import tensorflow as tf

from .custom_types import *


def load_2017annie_predict_EVD(age_lims: Sequence[int] = [20, 40]):
    X = np.loadtxt('../data/private/predict_EVD/X.csv', skiprows=1, delimiter=',')
    y = np.loadtxt('../data/private/predict_EVD/y.csv', skiprows=1, delimiter=',')
    age = np.loadtxt('../data/private/predict_EVD/age.csv', skiprows=1, delimiter=',')
    assert (X.shape[0] == y.shape[0]) and (y.shape[0] == age.shape[0])
    return X, y, age, age_lims

def make_ds(X, y, batch_size):
    return tf.data.Dataset.from_tensor_slices({'features': X, 'label':y})\
                 .batch(batch_size)

# This function is not stateless, due to the use of random.shuffle. It is used only once, before the training loop.
def split_by_age(
        X:   np.ndarray,
        y:   np.ndarray,
        age: np.ndarray,
        age_lims: Sequence[int]
) -> Sequence[Tuple[np.ndarray, np.ndarray]]:
    """ Split the datum points (X,y) into three sets based on the age of the patients.
    The patients are sorted into age groups [1-20y, 21-40y, >40y].
    The datum points within each set are shuffled.
    """
    if not (X.shape[0] == y.shape[0] and y.shape[0] == age.shape[0]):
        raise ValueError("Shape mismatch between X, y, and age. Their lengths "
                         "along axis 0 are {}, {}, and {}, respectively.".format(
                             X.shape[0], y.shape[0], age.shape[0]
                         ))
    
    age_lims = [-0.01] + age_lims + [150]
    
    
    idx_na = np.asarray((age <= age_lims[0]) | (age_lims[-1] < age)).nonzero()[0]
    num_na = idx_na.shape[0]
    random.shuffle(idx_na)
    
    ids = []
    for i, (lower, upper) in enumerate(zip(age_lims[:-1], age_lims[1:])):
        idx = np.asarray((lower < age) & (age <= upper)).nonzero()[0]
        random.shuffle(idx)
        ids.append(idx)
    
    num_valid = age.shape[0] - num_na
    for i, idx in enumerate(ids[:-1]):
        num_na_for_client = int(len(idx) / num_valid * num_na)
        ids[i] = np.concatenate((idx, idx_na[:num_na_for_client]))
        idx_na = idx_na[num_na_for_client:]
    ids[-1] = np.concatenate((ids[-1], idx_na))
    
    return [(X[idx], y[idx]) for idx in ids]

class SetLoaders(Enum):
    ANNIE_EVD = load_2017annie_predict_EVD
    
class Splitters(Enum):
    AGE_STRICT = split_by_age

def load(set_loader, splitter, seed, loader_kwargs={}, splitter_kwargs={}
        ) -> Sequence[Tuple[np.ndarray, np.ndarray]]:
    """Loads a data set and splits it accross clients.
    
    set_loader: one of the SetLoaders
    splitter:   one of the Splitters
    """
#     for k,v in locals().items():
#         print(k, ': ', v)
    random.seed(seed)
    return splitter(*set_loader(**loader_kwargs), **splitter_kwargs)