import argparse
import csv
# This is pyglpk, not scikit-glpk
import glpk
import gc
import itertools as it
import numpy as np
import time
import os

import os.path

POSSIBLE_SETTINGS = list(it.product(
    ["interior", "simplex", "exact"],  # method
    ["auto", "no"],  # scaling
    ["orig", "prescaled", "pow2_prescaled"],  # version
))


def basename(orig_filename):
    return orig_filename[:-len(".mps")]


def measure_one(mps_path, scaling="auto", method="simplex", freemps=True):
    if freemps:
        lp = glpk.LPX(freemps=mps_path)
    else:
        lp = glpk.LPX(mps=mps_path)

    if scaling == "auto":
        lp.scale()
    elif scaling == "no":
        lp.unscale()
    else:
        raise ValueError(f"invalid scaling '{scaling}'")

    gc.disable()
    start = time.perf_counter()
    if method == "simplex":
        solver_err = lp.simplex()
    elif method == "interior":
        solver_err = lp.interior()
    elif method == "exact":
        solver_err = lp.exact()
    else:
        raise ValueError(f"invalid method '{method}'")
    end = time.perf_counter()
    gc.enable()

    obj_value = lp.obj.value if solver_err is None else None
    duration = end - start
    return duration, obj_value

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig-dir", dest="orig_dir")
    parser.add_argument("--prescaled-dir", dest="prescaled_dir")
    parser.add_argument("--pow2-prescaled-dir", dest="pow2_prescaled_dir")
    parser.add_argument("--results-file", dest="results_file")
    parser.add_argument("--basename")
    parser.add_argument("--scaling")
    parser.add_argument("--method")
    parser.add_argument("--version")
    parser.add_argument("--fixed-mps", dest="fixed_mps", action="store_true")

    args = parser.parse_args()

    if args.version == "orig":
        mps_path = os.path.join(args.orig_dir, f"{args.basename}.mps")
    elif args.version == "prescaled":
        mps_path = os.path.join(args.prescaled_dir, f"{args.basename}-prescaled.mps")
    elif args.version == "pow2_prescaled":
        mps_path = os.path.join(args.pow2_prescaled_dir, f"{args.basename}-pow2-prescaled.mps")
    else:
        raise ValueError(f"invalid version '{args.version}'")

    duration, obj_value = measure_one(mps_path, args.scaling, args.method, not args.fixed_mps)

    with open(args.results_file, "a", newline="") as results_f:
        fieldnames = ["problem", "version", "scaling", "method", "time", "objective_value"]
        writer = csv.DictWriter(results_f, fieldnames=fieldnames)
        writer.writerow({
            "problem": args.basename,
            "version": args.version,
            "scaling": args.scaling,
            "method": args.method,
            "time": duration,
            "objective_value": obj_value,
        })