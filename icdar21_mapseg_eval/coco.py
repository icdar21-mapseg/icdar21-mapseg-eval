import numpy as np
from skimage.measure import label as labelize
from . import iou


def _deduce_mode_1(A):
    label_types = (np.uint16, np.int16, np.uint32, np.int32, np.uint64, np.int64)
    if A.ndim == 2:
        if A.dtype == "bool":
            return "segmentation"
        elif A.dtype in label_types:
            return "labelmap"
        elif A.dtype == "uint8":
            a_label_count = np.unique(A)
            if len(a_label_count) > 2:
                return "labelmap"
            else:
                return "segmentation"
    if A.ndim == 1:
        return "iou_array"
    return None

def _deduce_mode(A, B):
    mode_A = _deduce_mode_1(A)
    mode_B = _deduce_mode_1(B)

    if not mode_A:
        raise ValueError(f"Unable to deduce computation mode for COCO metric ({A.ndim}:{A.dtype})")
    if not mode_B:
        raise ValueError(f"Unable to deduce computation mode for COCO metric ({B.ndim}:{B.dtype})")


    if (mode_A == "iou_array") != (mode_B == "iou_array"):
        raise ValueError(f"Incompatible deduced mode for COCO metric: {mode_A} vs {mode_B}")

    return mode_A, mode_B



def _compute_labelmap(A):
    A = labelize(A.astype(np.uint8), connectivity=1)  # , ltype=cv2.CV_16U
    return A


def _compute_iou(A, B):
    hist_inter_2d = iou.intersections(A, B)
    iou_a, iou_b = iou.compute_IoUs(hist_inter_2d)
    return iou_a, iou_b


def COCO(A: np.ndarray, B: np.ndarray, mode=None, ignore_zero=True, plot=None):
    """
    Compute the COCO metric of one segmentation vs another segmentation

    plot: Path to output file for plot, or `None` if deactivated.
    """
    A = np.asarray(A)
    B = np.asarray(B)

    if mode not in {None, "segmentation", "labelmap", "iou_array"}:
        raise ValueError(f"Invalid mode '{mode}'.")

    if not mode:
        modeA, modeB = _deduce_mode(A, B)
    else:
        modeA = modeB = mode


    if modeA == "segmentation": A = _compute_labelmap(A)
    if modeB == "segmentation": B = _compute_labelmap(B)

    if modeA != "iou_array" and modeB != "iou_array":
        A, B = _compute_iou(A, B)

    if ignore_zero:
        A = A[1:]
        B = B[1:]

    #print(f"Number of labels in A: {A.size}")
    #print(f"Number of labels in B: {B.size}")

    if B.size == 0:
        print("Warning: empty prediction. Setting scores to 0 and skipping plot generation.")
        return 0., 0., 0.

    df = iou.compute_matching_scores(A, B)
    if plot:
        iou.plot_scores(df, out=plot)

    COCO_SQ = df["IoU"].mean() if len(df) > 0 else 0
    COCO_RQ = df["F-score"].iloc[0] if len(df) > 0 else 0
    COCO_PQ = COCO_SQ * COCO_RQ
    return COCO_PQ, COCO_SQ, COCO_RQ
