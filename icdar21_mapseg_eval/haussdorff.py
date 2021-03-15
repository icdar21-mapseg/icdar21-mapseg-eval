import numpy as np
import skimage.morphology as morph
from scipy.ndimage import distance_transform_edt as edt



def hausdorff(A: np.ndarray, B: np.ndarray, percentile=95):

    """
    Compute the 95%-Hausdorff distance between two binary images


    Parameters
    ----------
    A, B : np.ndarray
           Binary images
    percentile : int
           The percentile of the Hausdorff distance when computing distances from the boundaries
           (Default = 95)
    """

    if not isinstance(A, np.ndarray):
        raise TypeError("A is not a ndarray")
    if not isinstance(B, np.ndarray):
        raise TypeError("B is not a ndarray")
    if not (A.ndim == 2 and B.ndim == 2):
        raise ValueError("A and B should be 2D")
    if not A.shape == B.shape:
        raise ValueError(f"A and B have incompatible shape {A.shape} vs {B.shape}")
    if not 0 <= percentile <= 100:
        raise ValueError(f"Percentile shall be between 0 and 100 (current is {percentile})")
    h, w = A.shape

    A = np.asarray(A, dtype=bool)
    B = np.asarray(B, dtype=bool)
    A = np.pad(A, pad_width=1)
    B = np.pad(B, pad_width=1)

    ext_grad_A = np.logical_xor(morph.binary_erosion(A), A)
    ext_grad_B = np.logical_xor(morph.binary_erosion(B), B)

    EA = edt(np.logical_not(A), return_distances=True)
    EB = edt(np.logical_not(B), return_distances=True)

    B_095 = np.percentile(EA[ext_grad_B], q=percentile)
    A_095 = np.percentile(EB[ext_grad_A], q=percentile)
    #print("A percentile {:.2f}".format(A_095))
    #print("B percentile {:.2f}".format(B_095))
    return max(A_095, B_095)






def test_haussdorf():
    A = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=bool)

    B = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
    ], dtype=bool)



    assert(hausdorff(A, B, percentile=100) == ((2 * 3**2) ** 0.5))
