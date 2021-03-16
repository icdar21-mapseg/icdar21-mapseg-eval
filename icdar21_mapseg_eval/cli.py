import argparse
from pathlib import Path
import json
import sys
import warnings

import pandas as pd
from progress.bar import Bar
from skimage.io import imread
import numpy as np

from . import hausdorff
from . import eval_pt_detect, show_predictions_classified, plot_f_vs_dist_curve
from . import COCO


def task_01(A, B):
    def run_COCO(A, B):
        A = imread(A, as_gray=True)
        B = imread(B, as_gray=True)
        return COCO(A, B)

    A = Path(A)
    B = Path(B)
    if A.is_file() and B.is_file():
        mode = "file"
    elif A.is_dir() and B.is_dir():
        mode = "dir"
    else:
        raise ValueError("Invalid inputs")

    if mode == "file":
        print(run_COCO(A, B))

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
            results.append(run_COCO(a, b))
            bar.next()
        bar.finish()

        df = pd.DataFrame({"Filename": A_stems, "COCO PQ": results})
        print(df)


def check_create_output(directory: Path):
    if directory.exists():
        if not directory.is_dir():
            raise ValueError(f"{directory} exists and is not a directory.")
        else:
            warnings.warn(f"{directory} already exist.")
    else:
        directory.mkdir(parents=True, exist_ok=True)


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
            results.append(run_hausdorff(a, b))
            bar.next()
        bar.finish()

        df = pd.DataFrame({"Filename": A_stems, "Error": results})
        print(df)


def run_eval_pt_detect(ft: Path, fp: Path, output: Path, *, radius_limit, beta):
    expected = np.loadtxt(ft, delimiter=",", skiprows=1)
    predicted = np.loadtxt(fp, delimiter=",", skiprows=1)
    area, df, df_dbg = eval_pt_detect(expected, predicted, radius_limit, beta=beta)
    df["Reference"] = ft.stem
    df["Prediction"] = fp.stem
    df_dbg["Reference"] = ft.stem
    df_dbg["Prediction"] = fp.stem
    basename_out = fp.stem
    df.to_csv(output / f"{basename_out}.eval.csv")
    df_dbg.to_csv(output / f"{basename_out}.plot.csv")
    plot_f_vs_dist_curve(
        df_dbg, radius_limit, beta, filename=output / f"{basename_out}.plot.pdf"
    )
    show_predictions_classified(
        expected, df, radius_limit, filename=output / f"{basename_out}.clf.pdf"
    )
    return area, df, df_dbg


def task_03(A, B, output):
    radius_limit = 50
    beta = 0.5

    A = Path(A)
    B = Path(B)
    output = Path(output)
    if A.is_file() and B.is_file():
        mode = "file"
    elif A.is_dir() and B.is_dir():
        mode = "dir"
    else:
        raise ValueError("Invalid inputs: must be either two files or two directories.")

    check_create_output(output)

    if mode == "file":
        score, _, _ = run_eval_pt_detect(A, B, output, radius_limit=radius_limit, beta=beta)
        print(f"{B} - Score: {score:0.3f}")

    if mode == "dir":
        As = sorted(A.glob("*-OUTPUT-GT.csv"))
        Bs = sorted(B.glob("*-OUTPUT-PRED.csv"))
        if len(Bs) == 0:
            # Tolerate bad filenames
            warnings.warn("Prediction files should be named NNN-OUTPUT-PRED.csv and not NNN-OUTPUT-GT.csv.")
            Bs = sorted(B.glob("*-OUTPUT-GT.csv"))

        A_names = [str(x.name) for x in As]
        B_names = [str(x.name) for x in Bs]
        if not A_names or not B_names:
            raise RuntimeError("Empty reference and/or result directory.")

        diff = set([n[:3] for n in A_names]) ^ set([n[:3] for n in B_names])
        if diff:
            raise RuntimeError(f"No reference or result for {sorted(list(diff))}.")

        bar = Bar("Processing", max=len(As))
        results = []
        for a, b in zip(As, Bs):
            # Do some work
            score, _, _ = run_eval_pt_detect(a, b, output, radius_limit=radius_limit, beta=beta)
            results.append(score)
            bar.next()
        bar.finish()

        df = pd.DataFrame(
            {"Reference": A_names, "Predictions": B_names, "Score": results}
        )
        df.set_index(["Reference", "Predictions"], inplace=True)
        print(df)
        global_score = df["Score"].mean()
        print("==============================")
        print(f"Global score for task 3: {global_score:0.3f}")
        print("==============================")
        df.to_csv(output / f"global_rad:{radius_limit}_beta:{beta:0.2f}.csv")
        data = {
            "global_mean_score": global_score,
            "references": [str(p) for p in As],
            "predictions": [str(p) for p in Bs],
            "radius_limit": radius_limit,
            "beta": beta,
        }
        with open(output / "global_score.json", "w") as outfile:
            json.dump(data, outfile)


def main():
    parser = argparse.ArgumentParser(prog="icdar21-mapseg-eval")
    subparsers = parser.add_subparsers()

    # TASK 02
    parser_a = subparsers.add_parser("T1", help="Task 1 - Detect Building Blocks")
    parser_a.add_argument("A", help="Path to the reference segmentation")
    parser_a.add_argument("B", help="Path to the predicted segmentation")
    parser_a.set_defaults(task=1)

    # TASK 02
    parser_b = subparsers.add_parser("T2", help="Task 2 - Segment Map Content Area")
    parser_b.add_argument("A", help="Path to the reference segmentation")
    parser_b.add_argument("B", help="Path to the predicted segmentation")
    parser_b.set_defaults(task=2)

    # TASK 03
    parser_b = subparsers.add_parser("T3", help="Task 3 - Detect graticule lines intersections")
    parser_b.add_argument("A", help="Path to the reference detection")
    parser_b.add_argument("B", help="Path to the predicted detection")
    parser_b.add_argument("output", help="Path to output directory")
    parser_b.set_defaults(task=3)

    args = parser.parse_args()
    if "task" not in args:
        print("Please select a task to evaluate.", file=sys.stderr)
        parser.print_help()

    if args.task == 1:
        task_01(args.A, args.B)
    elif args.task == 2:
        task_02(args.A, args.B)
    elif args.task == 3:
        task_03(args.A, args.B, args.output)
