import circuit_ineq
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--in", dest="in_mps", help="input mps file",
                    required=True)
parser.add_argument("--out", dest="out_mps", help="output mps file",
                    required=True)
parser.add_argument("--fixed-mps", dest="fixed_mps", action="store_true",
                    help="use the fixed MPS format")
parser.add_argument("--use-gurobi", dest="use_gurobi", action="store_true",
                    help="preparse the MPS with gurobi before loading it with GLPK")
args = parser.parse_args()


free_mps = not args.fixed_mps
orig_lp = circuit_ineq.read_mps(args.in_mps, free=free_mps, use_gurobi=args.use_gurobi)

semistandard_lp = circuit_ineq.convert_to_semi_standard(*orig_lp)

circuit_ineq.write_mps(*semistandard_lp, args.out_mps, free=free_mps)
