import argparse
from pathlib import Path
import json

import pandas as pd
from progress.bar import Bar
from skimage.io import imread
import numpy as np

from . import hausdorff
from . import eval_pt_detect, show_predictions_classified, plot_f_vs_dist_curve


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


def task_03(A, B, output):
    radius_limit = 50
    beta = 0.5

    def run_eval_pt_detect(ft, fp, output):
        expected = np.loadtxt(ft, delimiter=",", skiprows=1)
        predicted = np.loadtxt(fp, delimiter=",", skiprows=1)
        area, df, df_dbg = eval_pt_detect(expected, predicted, radius_limit, beta=beta)
        df["File"] = fp.stem
        df_dbg["File"] = fp.stem
        df.to_csv(output / fp.with_suffix("eval.csv"))
        df_dbg.to_csv(output / fp.with_suffix("plot.csv"))
        plot_f_vs_dist_curve(df_dbg, radius_limit, beta, filename=output/fp.with_suffix("plot.pdf"))
        show_predictions_classified(expected, df, radius_limit, filename=output/fp.with_suffix("clf.pdf"))
        return area, df, df_dbg

    A = Path(A)
    B = Path(B)
    output = Path(output)
    if A.is_file() and B.is_file():
        mode = "file"
    elif A.is_dir() and B.is_dir():
        mode = "dir"
    else:
        raise ValueError("Invalid inputs")

    if mode == "file":
        score, _, _ = run_eval_pt_detect(A, B, output)
        print(f"{B} - Score: {score:0.3f}")

    if mode == "dir":
        A = sorted(A.glob("*-OUTPUT-GT.csv"))
        B = sorted(B.glob("*-OUTPUT-GT.csv"))

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
            score, _, _ = run_eval_pt_detect(a, b, output)
            results.append(score)
        bar.finish()

        df = pd.DataFrame({"Filename": A_stems, "Score": results})
        print(df)
        global_score = df["Score"].mean()
        print("==============================")
        print(f"Global score for task 3: {global_score:0.3f}")
        print("==============================")
        df.to_csv(output / "global.csv")
        with open(output / "global_score.json") as outfile:
            json.dump({"score": global_score}, outfile)


parser = argparse.ArgumentParser(prog="icdar21-mapseg-eval")
subparsers = parser.add_subparsers()

# TASK 02
parser_b = subparsers.add_parser("T02", help="Task 2 - Segment Map Content Area")
parser_b.add_argument("A", help="Path to the reference segmentation")
parser_b.add_argument("B", help="Path to the predicted segmentation")
parser_b.set_defaults(task=2)
# TASK 03
parser_b = subparsers.add_parser("T03", help="Task 3 - Detect graticule lines intersections")
parser_b.add_argument("A", help="Path to the reference detection")
parser_b.add_argument("B", help="Path to the predicted detection")
parser_b.add_argument("output", help="Path to output directory")
parser_b.set_defaults(task=3)

args = parser.parse_args()
if args.task == 2:
    task_02(args.A, args.B)
elif args.task == 3:
    task_03(args.A, args.B, args.output)
