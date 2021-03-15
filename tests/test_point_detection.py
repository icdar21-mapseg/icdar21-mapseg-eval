import numpy as np

from icdar21_mapseg_eval.point_detection import eval_pt_detect

radius_limit = 118

ground_truth = np.float32([
    [5710, 1170],  # A
    [8080, 1170],  # B
    [3330, 3530],  # C
    [5710, 3550],  # D
    [8085, 3540],  # E
    [3327, 5922],  # F
    [5715, 5940],  # G
    [8085, 5942]]) # H

predicted = np.float32([
    # A
    [5710, 1170], # exact match
    # B
    [8080 + 2*radius_limit, 1170+2*radius_limit], # Match outside acceptable area
    # C
    [3330+10, 3530+10], # multiple acceptable matches
    [3330-10, 3530-10],
    [3330+10, 3530+0],
    [3330+10, 3530+30],
    # D
    [5710+10, 3550-10], # 1 good match
    # E
    [8085+radius_limit, 3540], # far match, on the edge
    # F
    # Nothing, no match
    # G and H
    [(5715+8085)/2, (5940+5942)/2] # point on the perpendicular bisector of the two points
])


def test_eval_working01():
    area, _df, _ = eval_pt_detect(ground_truth, predicted, radius_limit, beta=0.5, debug=True)
    assert(np.abs(area - 0.374) < 0.01)


def test_empty_pred():
    area, _df, _ = eval_pt_detect(np.array([(0,0), (1,1)]), np.empty((0,2)), 10, beta=0.5, debug=True)
    assert(np.isclose(area, 0))

def test_empty_gt():
    area, _df, _ = eval_pt_detect(np.empty((0,2)), np.array([(0,0), (1,1)]), 10, beta=0.5, debug=True)
    assert(np.isclose(area, 0))

def test_both_empty():
    area, _df, _ = eval_pt_detect(np.empty((0,2)), np.empty((0,2)), 10, beta=0.5, debug=True)
    assert(np.isclose(area, 0))

def test_missing01():
    area, _df, _ = eval_pt_detect(np.array([(0,0), (1,1)]), np.array([(0,0)]), 10, beta=0.5, debug=True)
    assert(np.abs(area - 0.833) < 0.01)

def test_missing02():
    area, _df, _ = eval_pt_detect(np.array([(0,0), (1,1)]), np.array([(1,1)]), 10, beta=0.5, debug=True)
    assert(np.abs(area - 0.833) < 0.01)

def test_extra():
    area, _df, _ = eval_pt_detect(np.array([(0,0)]), np.array([(0,0), (1,1)]), 10, beta=0.5, debug=True)
    assert(np.abs(area - 0.556) < 0.01)

def test_limit():
    area, _df, _ = eval_pt_detect(np.array([(0,0)]), np.array([(10,0)]), 10, beta=0.5, debug=True)
    assert(np.abs(area - 0.50) < 0.01)

def test_duplicate():
    area, _, _ = eval_pt_detect(np.array([(0,0)]), np.array([(0,0), (0,0)]), 10, beta=0.5, debug=True)
    assert(np.abs(area - 0.556) < 0.01)

def test_order01():
    prev_area = None
    for radius_limit in [1, 10, 100, 500, 1000]:
        area, _, _ = eval_pt_detect(ground_truth, predicted, radius_limit, beta=0.5, debug=False)
        if prev_area is not None:
            assert(prev_area <= area)
        prev_area = area

def test_gt_vs_gt():
    np.random.seed(42)
    predictions = np.float32(np.random.randint(0,10000,(100,2)))
    area, _, _ = eval_pt_detect(predictions, predictions, 0.001, beta=0.5, debug=False)
    assert(np.isclose(area, 1))
    
def test_order02():
    np.random.seed(42)
    a = np.float32(np.random.randint(0,10000,(100,2)))
    b = np.float32(np.random.randint(0,10000,(100,2)))
    prev_area = None
    for radius_limit in [1, 10, 100, 500, 1000]:
        area, _, _ = eval_pt_detect(a, b, radius_limit, beta=0.5, debug=False)
        if prev_area is not None:
            assert(prev_area <= area)
        prev_area = area
