import argparse
import csv
import gc
import itertools as it
import time
import os

import os.path
import pexpect as p

from fractions import Fraction

SCIP_BINARY_NAME = "scip-exact"
SCIP_PROMPT = "SCIP>"

POSSIBLE_SETTINGS = list(it.product(
    ["orig", "prescaled", "pow2_prescaled"],  # version
))

def basename(orig_filename):
    return orig_filename[:-len(".mps")]


def setup_scip_process(timeout):
    pcs = p.spawn(SCIP_BINARY_NAME, timeout=timeout)
    pcs.expect(SCIP_PROMPT)

    # disable scaling
    pcs.sendline("set lp advanced scaling 0")
    pcs.expect(SCIP_PROMPT)

    # enable exact solving
    pcs.sendline("set exact enabled true")
    pcs.expect(SCIP_PROMPT)

    return pcs


def measure_one(mps_path, timeout):
    pcs = setup_scip_process(timeout)

    pcs.sendline(f"read {mps_path}")
    pcs.expect("original problem has")
    pcs.expect(SCIP_PROMPT)

    gc.disable()
    start = time.perf_counter()
    pcs.sendline("optimize")
    pcs.expect(r"Solving Time \(sec\) : (.*?)\r\n")
    end = time.perf_counter()
    gc.enable()

    pcs.terminate()

    reported_time = float(pcs.match.group(1))
    measured_time = end - start
    obj_value = float(Fraction(pcs.match.group(1).decode("utf8")))

    return reported_time, measured_time, obj_value

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig-dir", dest="orig_dir")
    parser.add_argument("--prescaled-dir", dest="prescaled_dir")
    parser.add_argument("--pow2-prescaled-dir", dest="pow2_prescaled_dir")
    parser.add_argument("--results-file", dest="results_file")
    parser.add_argument("--basename")
    parser.add_argument("--version")
    parser.add_argument("--timeout", type=int, default=600)

    args = parser.parse_args()

    if args.version == "orig":
        mps_path = os.path.join(args.orig_dir, f"{args.basename}.mps")
    elif args.version == "prescaled":
        mps_path = os.path.join(args.prescaled_dir, f"{args.basename}-prescaled.mps")
    elif args.version == "pow2_prescaled":
        mps_path = os.path.join(args.pow2_prescaled_dir, f"{args.basename}-pow2-prescaled.mps")
    else:
        raise ValueError(f"invalid version '{args.version}'")

    reported_time, measured_time, obj_value = measure_one(mps_path, args.timeout)
    print("read")

    with open(args.results_file, "a", newline="") as results_f:
        fieldnames = ["problem", "version", "reported_time", "measured_time", "objective_value"]
        writer = csv.DictWriter(results_f, fieldnames=fieldnames)
        writer.writerow({
            "problem": args.basename,
            "version": args.version,
            "reported_time": reported_time,
            "measured_time": measured_time,
            "objective_value": obj_value,
        })

    print("written")
