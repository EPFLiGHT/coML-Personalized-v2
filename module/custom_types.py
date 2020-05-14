from typing import Any, Generator, Tuple, Mapping, Sequence, Optional, Callable, Union, NamedTuple
from collections import namedtuple

import numpy as np
import haiku as hk
import jax.numpy as jnp
import tensorflow as tf

Batch = Mapping[str, jnp.ndarray]
DatasetSequence = Sequence[tf.data.Dataset]
LossFunction = Callable[[hk.Params, Batch], jnp.ndarray]
OptState = Any
ClientState = namedtuple("ClientState", "next_idx\
                                         epoch_count\
                                         similarity",
                         defaults=[int(0), int(0), float(1)]
                        )


# define hyperparameters format.
ServerHyperParams = namedtuple("ServerHyperParams", "num_rounds\
                                                     max_batches_per_round\
                                                     max_epochs_per_round\
                                                     batch_size\
                                                     seed")

# Define the aggregator format
SimilarityAggregatorHyperParams = namedtuple("SimilarityAggregatorHyperParams", "distance_penalty_factor")
AverageAggregatorHyperParams = namedtuple("AverageAggregatorHyperParams", "")
AggregatorHyperParams = Union[SimilarityAggregatorHyperParams, AverageAggregatorHyperParams]
AggregatorFunction = Callable[[Sequence[hk.Params], Sequence[ClientState], AggregatorHyperParams], Tuple[hk.Params, Sequence[ClientState]]]
class Aggregator(NamedTuple):
    aggregator_function: AggregatorFunction
    aggregator_hyperparams: AggregatorHyperParams

# message to the client from server.
ClientMessage = namedtuple("ClientMessage", "params\
                                             opt_init_input")

# message to the server from client.
ServerMessage = namedtuple("ServerMessage", "aggregator_input\
                                             stateupdater_input")


# message to the server from client for book keeping.
DiagnosticsMessage = namedtuple("DiagnosticsMessage", "train_loss\
                                                       train_acc\
                                                       test_loss\
                                                       test_acc\
                                                       weight")
ClientOutput = Tuple[ServerMessage, DiagnosticsMessage]

# State storing the diagnostics history
dstate_fields = ["train_loss_global",
                 "train_accuracy_global",
                 "test_loss",
                 "test_accuracy",
                 "similarities",
                 "epoch_counts"
                ]
DiagnosticsState = namedtuple("DiagnosticsState", dstate_fields,
                              defaults= [list()]*len(dstate_fields)
                             )

Clients_X_y = Sequence[Tuple[np.ndarray, np.ndarray]]