from icdar21_mapseg_eval import COCO
import numpy as np


def test_COCO_1():
    P = 4
    T = 5
    A = np.array([0.6, 0.7, 0.8, 0.94])
    B = np.array([0.2, 0.6, 0.7, 0.8, 0.94])

    PQ, SQ, RQ = COCO(A, B, ignore_zero=False, plot=False)

    fscore_05 = 2 * 4 / (4 + 5)
    assert(RQ == fscore_05)
    assert(SQ == A.mean())
    assert(PQ == SQ * RQ)


def test_COCO_labels_dtypes():
    for data_type in (np.uint16, np.int16, np.uint32, np.int32, np.uint64, np.int64):
        A = np.array([[1, 2, 0, 0], [1, 2, 0, 0]], dtype=data_type)
        B = A.copy()

        PQ, SQ, RQ = COCO(A, B, ignore_zero=False, plot=False)

        assert(RQ == 1)
        assert(SQ == 1)
        assert(PQ == 1)


def test_COCO_seg():
    for data_type in (np.uint8, "bool"):
        A = np.array([[1, 1, 0, 0], [1, 1, 0, 0]], dtype=data_type)
        B = A.copy()

        PQ, SQ, RQ = COCO(A, B, ignore_zero=False, plot=False)

        assert(RQ == 1)
        assert(SQ == 1)
        assert(PQ == 1)
