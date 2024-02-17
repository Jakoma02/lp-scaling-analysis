import circuit_ineq
import argparse
import pickle

def find_d(in_filename, check_sanity=False, free_mps=True, use_gurobi=False):
    lp = circuit_ineq.read_mps(in_filename, free=free_mps, use_gurobi=use_gurobi)
    c, A_ub, b_ub, A_eq, b_eq, bounds, integrality = lp
    ss_c, ss_A, ss_b, ss_bounds, ss_integrality = circuit_ineq.convert_to_semi_standard(c, A_ub, b_ub, A_eq, b_eq, bounds, integrality)

    ds, _, _ = circuit_ineq.estimate_optimal_preconditioning_ds(ss_A, check_sanity)
    return ds


parser = argparse.ArgumentParser()
parser.add_argument("--in", dest="in_mps", help="Input mps file",
                    required=True)
parser.add_argument("--out", dest="out_file", help="Output file (pickle)",
                    required=True)
parser.add_argument("--check-sanity", dest="check_sanity", action="store_true",
                    help="Run some basic test to determine whether the solution makes any sense")
parser.add_argument("--fixed-mps", action="store_true",
                    help="use the fixed MPS format")
parser.add_argument("--use-gurobi", dest="use_gurobi", action="store_true",
                    help="preparse the MPS with gurobi before loading it with GLPK")
args = parser.parse_args()

ds = find_d(args.in_mps, args.check_sanity, not args.fixed_mps, args.use_gurobi)
with open(args.out_file, "wb") as out_f:
    pickle.dump(ds, out_f)