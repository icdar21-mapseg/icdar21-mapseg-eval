import argparse
import pandas as pd
from progress.bar import Bar
from pathlib import Path

from . import hausdorff

from skimage.io import imread


def task_02(A, B):
    def run_hausdorff(A, B):
        A = imread(A, as_gray=True)
        B = imread(B, as_gray=True)
        return hausdorff(A, B)

    A = Path(A)
    B = Path(B)
    if A.is_file() and B.is_file():
        mode = "file"
    elif A.is_dir() and B.is_dir():
        mode = "dir"
    else:
        raise ValueError("Invalid inputs")

    if mode == "file":
        print(run_hausdorff(A, B))

    if mode == "dir":
        A = sorted(A.glob("*-OUTPUT-GT.png"))
        B = sorted(B.glob("*-OUTPUT-GT.png"))  # FIXME GT -> PRED (tolerate both?)

        A_stems = [str(x.name) for x in A]
        B_stems = [str(x.name) for x in B]
        if not A_stems or not (B_stems):
            raise RuntimeError("Empty reference and/or result directory.")

        diff = set(A_stems) ^ set(B_stems)
        if diff:
            raise RuntimeError(f"No reference or result for {diff}.")

        bar = Bar("Processing", max=len(A))
        results = []
        for a, b in zip(A, B):
            # Do some work
            results.append(run_hausdorff(a, b))
            bar.next()
        bar.finish()

        df = pd.DataFrame({"Filename": A_stems, "Error": results})
        print(df)


parser = argparse.ArgumentParser(prog="icdar21-mapseg-eval")
subparsers = parser.add_subparsers()

# TASK 02
parser_b = subparsers.add_parser("T02", help="Task 2 - Segment Map Content Area")
parser_b.add_argument("A", help="Path to the reference segmentation")
parser_b.add_argument("B", help="Path to the predicted segmentation")
parser_b.set_defaults(task=2)

args = parser.parse_args()
if args.task == 2:
    task_02(args.A, args.B)
