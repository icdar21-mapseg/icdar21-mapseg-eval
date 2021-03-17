import numpy as np
from skimage.measure import label as labelize
from . import iou


def _deduce_mode(A, B):
    if A.ndim == 2 and B.ndim == 2:
        if A.dtype == "bool" and B.dtype == "bool":
            return "segmentation"
        elif A.dtype in (np.int16, np.int32, np.int64) and B.dtype in (
            np.int16,
            np.int32,
            np.int64,
        ):
            return "labelmap"
        elif A.dtype == "uint8" and B.dtype == "uint8":
            a_label_count = np.unique(A)
            b_label_count = np.unique(B)
            if len(a_label_count) > 2 or len(b_label_count) > 2:
                return "labelmap"
            else:
                return "segmentation"
    if A.ndim == 1 and B.ndim == 1:
        return "iou_array"
    return None


def _compute_labelmap(A):
    A = labelize(A.astype(np.uint8), connectivity=1)  # , ltype=cv2.CV_16U
    return A


def _compute_iou(A, B):
    hist_inter_2d = iou.intersections(A, B)
    iou_a, iou_b = iou.compute_IoUs(hist_inter_2d)
    return iou_a, iou_b


def COCO(A: np.ndarray, B: np.ndarray, mode=None, ignore_zero=True, plot=False):
    """
    Compute the COCO metric of one segmentation vs another segmentation
    """
    A = np.asarray(A)
    B = np.asarray(B)

    mode = mode or _deduce_mode(A, B)
    if not mode:
        raise ValueError(f"Unable to deduce computation mode for COCO metric ({A.ndim}:{A.dtype} and {B.dtype}).")
    if mode not in {"segmentation", "labelmap", "iou_array"}:
        raise ValueError(f"Invalid mode '{mode}'.")

    if mode == "segmentation":
        A = _compute_labelmap(A)
        B = _compute_labelmap(B)
        mode = "labelmap"

    if mode == "labelmap":
        A, B = _compute_iou(A, B)
        mode = "iou_array"

    if ignore_zero:
        A = A[1:]
        B = B[1:]

    df = iou.compute_matching_scores(A, B)
    if plot:
        iou.plot_scores(df)

    COCO_SQ = df["IoU"].mean()
    COCO_RQ = df["F-score"].iloc[0]
    COCO_PQ = COCO_SQ * COCO_RQ
    return COCO_PQ, COCO_SQ, COCO_RQ
