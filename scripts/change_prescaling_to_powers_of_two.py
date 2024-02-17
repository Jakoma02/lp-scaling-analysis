import argparse
import pickle
import numpy as np

from sage.all import vector

def round_vector_to_powers_of_two(v):
    mantissas, exponents = np.frexp(v)
    mantissas.fill(0.5)
    new_v = vector(np.ldexp(mantissas, exponents))
    return new_v


parser = argparse.ArgumentParser()
parser.add_argument("--in", dest="in_file", help="input prescaling vector pickle")
parser.add_argument("--out", dest="out_file", help="output prescaling vector pickle")

args = parser.parse_args()

with open(args.in_file, "br") as in_f:
    old_ds = pickle.load(in_f)

new_ds = round_vector_to_powers_of_two(old_ds)

with open(args.out_file, "wb") as out_f:
    pickle.dump(new_ds, out_f)