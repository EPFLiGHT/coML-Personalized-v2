from enum import Enum
from typing import Sequence, Tuple
import random
import os.path

import numpy as np
import tensorflow as tf

from .custom_types import *

def load_X_y_age(load_dir):
    outs = []
    for file in ['X', 'y', 'age']:
        outs.append(
            np.loadtxt(os.path.join(load_dir, file+'.csv'),
                       skiprows=1,
                       delimiter=','
                      )
        )
    assert (outs[0].shape[0] == outs[1].shape[0]) and (outs[1].shape[0] == outs[2].shape[0])
    return tuple(outs)

def load_2017annie_predict_EVD(age_lims: Sequence[int] = [20, 40]):
    load_dir = '../data/private/predict_EVD'
    X, y, age = load_X_y_age(load_dir)
    return X, y, age, age_lims

def load_Titanic_predict_survived(age_lims: Sequence[int] = [20, 35]):
    load_dir = '../data/non-private/predict_titanic_survived'
    X, y, age = load_X_y_age(load_dir)
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
    The patients are sorted into age groups delimited by age_lims.
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
        ids.append(idx)
    
    num_valid = age.shape[0] - num_na
    for i, idx in enumerate(ids[:-1]):
        num_na_for_client = int(len(idx) / num_valid * num_na)
        ids[i] = np.concatenate((idx, idx_na[:num_na_for_client]))
        idx_na = idx_na[num_na_for_client:]
    ids[-1] = np.concatenate((ids[-1], idx_na))
    for idx in ids:
        random.shuffle(idx)
    return [(X[idx], y[idx]) for idx in ids]

def split_some_by_age(
        X:   np.ndarray,
        y:   np.ndarray,
        age: np.ndarray,
        age_lims: Sequence[int]
) -> Sequence[Tuple[np.ndarray, np.ndarray]]:
    """ Split the datum points (X,y) into three sets based on the age of the patients.
    Like `split_by_age`, except the first two sets are mixed. In effect, yes, this
    implies that the first value in age_lims is not used.
    The datum points within each set are shuffled.
    """
    clients_by_age = split_by_age(X, y, age, age_lims[1:])
    X0, y0 = clients_by_age[0]
    cut = X0.shape[0] // 2
    return [(X0[:cut], y0[:cut]), (X0[cut:], y0[cut:])] + clients_by_age[1:]

class SetLoaders(Enum):
    ANNIE_EVD = load_2017annie_predict_EVD
    TITANIC_SURVIVED = load_Titanic_predict_survived
    
class Splitters(Enum):
    AGE_STRICT = split_by_age
    AGE_SOME   = split_some_by_age

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