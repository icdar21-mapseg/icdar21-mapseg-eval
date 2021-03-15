import numpy as np
from scipy.cluster.vq import vq
import matplotlib.pyplot as plt
import pandas as pd


def f_beta(tp, fn, fp, beta):
    """Direct implementation of https://en.wikipedia.org/wiki/F1_score#Definition"""
    return (1 + beta ** 2) * tp / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp)


def __identity(x):
    return x


def eval_pt_detect(
    expected,
    predicted,
    radius_limit,
    *,
    beta=0.5,
    transfer_fn=None,
    debug=False,
    show_plot=False,
):
    """transfer_fn must be a 0-intersecting, monotonous, increasing, continuous, positive-definite function."""

    def dbg_print(*args, **kwargs):
        if debug:
            print(*args, **kwargs)

    # for non linear response over distances, if needed
    f = transfer_fn
    if f is None:
        f = __identity

    count_expected = len(expected)
    count_predicted = len(predicted)

    # Step 1: identify closest expected points and the distance to them
    pred_to_targets, distances = vq(predicted, expected)
    order = np.argsort(distances)

    # Step 2: compute f_beta score as we increase the distance tolerance to consider matches as good
    # `seen`: Internal state to check whether we already matched some ground truth point
    seen = np.zeros(count_expected, dtype=bool)

    # Loop over predicted points
    area = 0.0
    prev_ax = 0.0
    prev_ay = 0.0
    dbg_x = [0.0]
    dbg_y = [0.0]  # f_beta(0, …)
    tps = []
    fps = []
    fns = []
    precisions = []
    recalls = []
    f_scores = []
    sorted_dst = []
    error_types = []  # 0: match, 1: extra match, 2: too far
    pred_xs = []
    pred_ys = []
    for tid, dst, pid in zip(pred_to_targets[order], distances[order], order):
        is_match = False
        pred_x, pred_y = predicted[pid]
        dbg_print(
            f"Considering predicted point {pid:02d}: ({pred_x:7.1f},{pred_y:7.1f})"
        )
        # Stop when is makes no sense to go further
        if dst > radius_limit:  # on the edge = OK here
            dbg_print(f"Reached radius limit ({radius_limit:.0f})")
            # All upcoming points will be counted as false positives
            error_types.append(2)
        else:
            dbg_print(
                f"\tMatches gt point {tid:02d}: ({expected[tid,0]:7.1f},{expected[tid,1]:7.1f}) @ {dst:.1f}"
            )
            if seen[tid]:
                dbg_print("\tEXTRA MATCH (noise)")
                error_types.append(1)
            else:
                dbg_print("\tFirst match (good)")
                error_types.append(0)
                is_match = True
            # Mark ground truth point as matched (at most once)
            seen[tid] = True
        # Count errors
        tp = np.sum(seen)
        fn = count_expected - tp
        fp = count_predicted - tp
        precision = (tp / count_predicted) if count_predicted != 0 else 0
        recall = (tp / count_expected) if count_expected != 0 else 0
        f_score = f_beta(tp, fn, fp, beta)
        # Save values
        pred_xs.append(pred_x)
        pred_ys.append(pred_y)
        sorted_dst.append(dst)
        tps.append(tp)
        fns.append(fn)
        fps.append(fp)
        precisions.append(precision)
        recalls.append(recall)
        f_scores.append(f_score)
        if is_match:
            # Compute trapezoidal area
            ax = f(dst) / f(radius_limit)
            ay = f_score
            dx = ax - prev_ax
            dy = (ay + prev_ay) / 2
            area += dx * dy
            dbg_print(
                f"\ttp:{tp:3d}, fn:{fn:3d}, fp:{fp:3d}, |expt|:{count_expected}, |pred|:{count_predicted},"
                f" x:{ax:.2f}, y:{ay:.2f}, area:{area:.3f}"
            )
            prev_ax = ax
            prev_ay = ay
            # Add debug points
            dbg_x.append(ax)
            dbg_y.append(ay)

    # add last rectangle up to the right limit
    area += (1.0 - prev_ax) * prev_ay
    dbg_x.append(1.0)
    dbg_y.append(prev_ay)
    dbg_print("Propagating to right limit")
    dbg_print(f"x:{1.0:.2f}, y:{prev_ay:.2f}, area:{area:.2f}")
    dbg_print(f"Score: {area:0.3f}")

    df = pd.DataFrame(
        {
            "Distance": sorted_dst,
            "Precision": precisions,
            "Recall": recalls,
            "F-beta": f_scores,
            "True Positives": tps,
            "False Positives": fps,
            "False Negatives": fns,
            "Error type": error_types,
            "Prediction x": pred_xs,
            "Prediction y": pred_ys,
        }
    )
    df = df.set_index("Distance")

    df_dbg = pd.DataFrame(
        {
            "Normalized distance": dbg_x,
            "F-beta": dbg_y,
        }
    )

    if show_plot:
        plot_f_vs_dist_curve(df_dbg, radius_limit, beta)

    return area, df, df_dbg


def plot_f_vs_dist_curve(df_dbg, radius_limit, beta, filename=None):
    plt.figure(figsize=(10, 10))
    plt.plot(df_dbg["Normalized distance"], df_dbg["F-beta"], "-x")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel(f"Normalized distance (0-{radius_limit:.1f} pixels)")
    plt.ylabel("$F_{\\beta=" f"{beta:0.2f}" "}$ score")
    plt.title("Evaluation curve")
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()


def show_predictions(expected, predicted, radius_limit, filename=None):
    _figure, axes = plt.subplots(figsize=(10, 10))
    for xy in expected:
        c = plt.Circle(
            xy, radius_limit, fill=False, linestyle="--", color="black", alpha=0.5
        )
        axes.add_patch(c)
    plt.plot(expected[:, 0], expected[:, 1], "og", label="ground truth")
    plt.plot(predicted[:, 0], predicted[:, 1], "xb", label="predicted")
    axes.set_aspect(1)
    plt.legend()
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()


def show_predictions_classified(expected, df, radius_limit, filename=None):
    _figure, axes = plt.subplots(figsize=(10, 10))
    for xy in expected:
        c = plt.Circle(
            xy, radius_limit, fill=False, linestyle="--", color="black", alpha=0.5
        )
        axes.add_patch(c)
    plt.plot(expected[:, 0], expected[:, 1], "og", label="ground truth")
    matches = df[df["Error type"] == 0]
    extra = df[df["Error type"] == 1]
    too_far = df[df["Error type"] == 2]
    plt.plot(
        too_far["Prediction x"],
        too_far["Prediction y"],
        "xk",
        label="predicted (too far)",
        alpha=0.8,
    )
    plt.plot(
        extra["Prediction x"],
        extra["Prediction y"],
        "xr",
        label="predicted (extra)",
        alpha=0.8,
    )
    plt.plot(
        matches["Prediction x"],
        matches["Prediction y"],
        "xb",
        label="predicted (match)",
        alpha=0.8,
    )
    axes.set_aspect(1)
    plt.legend()
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()
