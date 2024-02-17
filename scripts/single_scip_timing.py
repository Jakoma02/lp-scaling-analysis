import argparse
import csv
import pyscipopt
import gc
import itertools as it
import time
import os

import os.path

POSSIBLE_SETTINGS = list(it.product(
    ["orig", "prescaled", "pow2_prescaled"],  # version
))


def basename(orig_filename):
    return orig_filename[:-len(".mps")]


def measure_one(mps_path):
    m = pyscipopt.Model()
    m.setParam("lp/scaling", 0)  # disable scaling
    m.readProblem(mps_path)

    gc.disable()
    start = time.perf_counter()
    m.optimize()
    end = time.perf_counter()
    gc.enable()

    obj_value = m.getObjVal() if m.getStatus() == "optimal" else None
    duration = end - start
    return duration, obj_value

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig-dir", dest="orig_dir")
    parser.add_argument("--prescaled-dir", dest="prescaled_dir")
    parser.add_argument("--pow2-prescaled-dir", dest="pow2_prescaled_dir")
    parser.add_argument("--results-file", dest="results_file")
    parser.add_argument("--basename")
    parser.add_argument("--version")

    args = parser.parse_args()

    if args.version == "orig":
        mps_path = os.path.join(args.orig_dir, f"{args.basename}.mps")
    elif args.version == "prescaled":
        mps_path = os.path.join(args.prescaled_dir, f"{args.basename}-prescaled.mps")
    elif args.version == "pow2_prescaled":
        mps_path = os.path.join(args.pow2_prescaled_dir, f"{args.basename}-pow2-prescaled.mps")
    else:
        raise ValueError(f"invalid version '{args.version}'")

    duration, obj_value = measure_one(mps_path)

    with open(args.results_file, "a", newline="") as results_f:
        fieldnames = ["problem", "version", "time", "objective_value"]
        writer = csv.DictWriter(results_f, fieldnames=fieldnames)
        writer.writerow({
            "problem": args.basename,
            "version": args.version,
            "time": duration,
            "objective_value": obj_value,
        })