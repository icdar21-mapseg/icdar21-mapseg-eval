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
