import circuit_ineq

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

MAX_PROCESSES = 4
SUFFIX = "seminormal_relax"


def process(in_filename, out_filename):
    mip = circuit_ineq.read_mps(in_filename)
    semi_standard = circuit_ineq.convert_to_semi_standard(*mip)
    lp = circuit_ineq.lp_relaxation(*semi_standard)
    circuit_ineq.write_mps(*lp, filename=out_filename)

    return in_filename, out_filename


def to_out_filename(filename):
    extension = None
    if len(filename) >= 7 and filename[-7:] == ".mps.gz":
        extension = ".mps.gz"
    else:
        extension = ".mps"  # Kind of assuming...

    return filename[:-len(extension)] + "_" + SUFFIX + ".mps.gz"


parser = argparse.ArgumentParser()
parser.add_argument("--in", dest="in_dir", type=str,
                    help="path to input dir", required=True)
parser.add_argument("--out", dest="out_dir", type=str,
                    help="path to output dir", required=True)

args = parser.parse_args()


in_filenames = os.listdir(args.in_dir)
out_filenames = map(to_out_filename, in_filenames)

in_paths = (os.path.join(args.in_dir, x) for x in in_filenames)
out_paths = (os.path.join(args.out_dir, x) for x in out_filenames)

paths = list(zip(in_paths, out_paths))
paths.sort(key=lambda x: os.stat(x[0]).st_size)

paths = paths[-200:]

with ProcessPoolExecutor(max_workers=MAX_PROCESSES) as executor:
    jobs = []
    for in_f, out_f in paths:
        jobs.append(executor.submit(process, in_f, out_f))

    total = len(paths)
    done = 0

    for out in as_completed(jobs):
        in_f, out_f = out.result()
        done += 1
        print(f"[{done}/{total}] Processing of \"{in_f}\" to \"{out_f}\" done")