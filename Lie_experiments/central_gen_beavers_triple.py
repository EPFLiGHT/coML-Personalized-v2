"""
This script generates Beaver's triple in a centralized way using `c_i = a_i * sum_j b_j`

For each party output a dictionary : {"a": ..., "b": ..., "c": ...}

Notes:
1. Decentralized generation: fig 7. of "Overdrive: making SPDZ great again".
1. Use integer instead of float.
2. This method is intended for testing. As such, it is not secure!
    - np.random with a seed input from user is not safe for cryptographic use
    - c_i = a_i * sum_j b_j ==> Each party can reconstruct b (the shared secret) from its share of c and a.
"""
import argparse
import numpy as np
import os
import pickle

parser = argparse.ArgumentParser(description='Generate Beavers triples.')
parser.add_argument('-N', type=int, default=10,
                    help='Number of Beavers triples to generate.')
parser.add_argument('-NP', type=int, default=2,
                    help='Number of parties.')
parser.add_argument('-f', type=str, default='/datasets/bv',
                    help='The directory of generated files: e.g. bv/1.pickle, bv/2.pickle')
parser.add_argument('-seed', type=int, default=0,
                    help='Seed.')
args = parser.parse_args()

np.random.seed(args.seed)

# raise FileExistsError if exists
os.mkdir(args.f)

a = np.random.uniform(low=0.0, high=1.0, size=(args.N, args.NP))
b = np.random.uniform(low=0.0, high=1.0, size=(args.N, args.NP))
c = a * np.c_[b.sum(axis=1)]

for i in range(args.NP):
    with open(args.f + "/{}.pickle".format(i + 1), 'wb') as f:
        pickle.dump({"a": a[:, i], "b": b[:, i], "c": c[:, i]}, f)
