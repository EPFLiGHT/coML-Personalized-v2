import tensorflow as tf

def prep_make_ds(X, y, batch_size):
    def make_ds(idx):
        return tf.data.Dataset.from_tensor_slices({'features': X[idx], 'label':y[idx]})\
                 .batch(batch_size)
    return make_ds