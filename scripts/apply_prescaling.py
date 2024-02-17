import circuit_ineq
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--in", dest="in_mps", help="input mps file",
                    required=True)
parser.add_argument("--out", dest="out_mps", help="output mps file",
                    required=True)
parser.add_argument("--scaling-file", dest="scaling_file_path",
                    help="pickled MPS prescaling", required=True)
parser.add_argument("--in-fixed-mps", dest="in_fixed_mps", action="store_true",
                    help="use the fixed MPS format for input file")
parser.add_argument("--out-fixed-mps", dest="out_fixed_mps", action="store_true",
                    help="use the fixed MPS format for output file")
parser.add_argument("--use-gurobi", dest="use_gurobi", action="store_true",
                    help="preparse the MPS with gurobi before loading it with GLPK")
args = parser.parse_args()


with open(args.scaling_file_path, "rb") as scaling_file:
    ds = pickle.load(scaling_file)

in_free_mps = not args.in_fixed_mps
out_free_mps = not args.out_fixed_mps
orig_lp = circuit_ineq.read_mps(args.in_mps, free=in_free_mps, use_gurobi=args.use_gurobi)

semistandard_lp = circuit_ineq.convert_to_semi_standard(*orig_lp)
prescaled_lp = circuit_ineq.to_preconditioned_mip(*semistandard_lp, ds)

circuit_ineq.write_mps(*prescaled_lp, args.out_mps, free=out_free_mps)
