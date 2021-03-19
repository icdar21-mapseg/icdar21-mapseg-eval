# ICDAR 21 MapSeg Evaluation Tools

Welcome to the repository of the evaluation tools for ICDAR 2021 Competition in Historical Map Segmentation (MapSeg).

Here you can learn how to install and use the evaluation tools.
Feel free to report any problem in the issue tracker, and we will fix it as soon as possible.

## Installation
The evaluation tools are packaged as a Python package.

You need Python >= 3.7.1 to run these tools.

You can install the evaluation tools with the following command:
```shell
pip install icdar21-mapseg-eval
```

You can test your installation by running the following command which should display the online help of the command line tool:
```shell
icdar21-mapseg-eval --help
```

## Usage
There is a single command line program to run the evaluation for all 3 tasks.
To select the task you want to run the evaluation for, you need to follow this syntax:
```shell
icdar21-mapseg-eval {T1,T2,T3} path_to_reference path_to_prediction path_to_outputs
```
where:

- `{T1,T2,T3}` means select either `T1` or `T2` or `T3`.
- `path_to_reference` is the path to either a file or a directory.
- `path_to_prediction` is the path to either a file or a directory.
- `path_to_outputs` is the path to the directory where results will be stored.

You will find in the sections below sample usage for each task.


Should you need extra features, you may also import the package and use the exported functions.
```python
# Example for Task 1
from skimage.io import imread
from icdar21_mapseg_eval import COCO
task1_metrics = COCO(imread(ref_seg, as_gray=True), imread(pred_seg, as_gray=True))
print(task1_metrics)
# Example for Task 2
from skimage.io import imread
from icdar21_mapseg_eval import hausdorff
task2_metrics = hausdorff(imread(ref_seg, as_gray=True), imread(pred_seg, as_gray=True))
print(task2_metrics)
# Example for Task 3
import numpy as np
from icdar21_mapseg_eval import eval_pt_detect
task3_metrics = eval_pt_detect(
        np.loadtxt(ref_det, delimiter=",", skiprows=1), 
        np.loadtxt(pred_det, delimiter=",", skiprows=1),
        radius_limit=50, beta=0.5)  # values used for the competition
print(task3_metrics)
```

## Evaluation tool of task 1: Detect building blocks
*Please refer to https://icdar21-mapseg.github.io/tasks/task1/ for the description of the task.*

### Metric
This tool computes the [COCO PQ score](https://cocodataset.org/#panoptic-eval) associated to the instance segmentation returned by your system.
Please note that as we have only 1 "thing" class and not "stuff" class, we provide indicators only for the building blocks class.
These simplifications required a custom implementation which is fully compliant with the COCO PQ evaluation code.
We report COCO PQ (overall quality), COCO SQ (segmentation quality) and COCO RQ (detection/recognition quality) indicators.
For each of those, the values range from 0 (worst) to 1 (best).

For more details about the metric, please refer to the [evaluation details for task 1](https://icdar21-mapseg.github.io/tasks/task1/#metrics).

### Tool sample usage
The script supports comparing either:

* a predicted segmentation to a reference segmentation (as two binary images in PNG or two label maps in TIFF16).
* a reference directory to a reference segmentation.  
  In this case, reference files are expected to end with ``-OUTPUT-GT.png``, and prediction files with ``-OUTPUT-PRED.png`` or ``-OUTPUT-*.tiff``.


Comparing two files:

```
$ icdar21-mapseg-eval T1 201-OUTPUT-GT.png 201-OUTPUT-PRED.png output_dir
201-OUTPUT-PRED.png - COCO PQ 1.00 = 1.00 SQ * 1.00 RQ
```

Comparing two directories:

```
$ icdar21-mapseg-eval T1 1-detbblocks/validation/ mypred/t1/validation/ output_dir
Processing |################################| 1/1
                                       COCO PQ  COCO SQ  COCO RQ
Reference         Prediction                                  
201-OUTPUT-GT.png 201-OUTPUT-PRED.png      1.0      1.0      1.0
==============================
Global score for task 1: 1.000
============================
```

### Files generated in output folder
The output directory will contain something like:

```
201-OUTPUT-GT.plot.pdf 
global_coco.csv        
global_score.json      
```

Detail:
- `global_coco.csv`:  
  COCO metrics for each image.
- `global_score.json`:  
  Easy to parse file for global score with a summary of files analyzed.
- `NNN-OUTPUT-PRED.plot.pdf`:  
  Plot of the F-score against all IoU thresholds (COCO PQ is the area under the F-score curve + the value of the F-score at 0.5).


## Evaluation tool of task 2: Segment map content area
*Please refer to https://icdar21-mapseg.github.io/tasks/task2/ for the description of the task.*

### Metric
This tool computes the 95% [Haussdorff distance](https://en.wikipedia.org/wiki/Hausdorff_distance) (HD95) between two binary images.
This measures how the outline of the reference area and the predicted area are distant.
HD95 values range from 0 (perfect detection) to large values as it is an error measure.

For more details about the metric, please refer to the [evaluation details for task 2](https://icdar21-mapseg.github.io/tasks/task2/#metrics).

### Tool sample usage
The script supports comparing either:

* a predicted segmentation to a reference segmentation (as two binary images)
* a reference directory to a reference segmentation
  In this case, reference files are expected to end with ``-OUTPUT-GT.png``, and prediction files with ``-OUTPUT-PRED.png``.


Comparing two files:

```
$ icdar21-mapseg-eval T2 201-OUTPUT-GT.png 201-OUTPUT-PRED.png output_dir
201-OUTPUT-PRED.png - Haussdorff95 = 0.00
```

Comparing two directories:

```
$ icdar21-mapseg-eval T2 ./2-segmaparea/validation mypred/t2/validation output_dir
.../PIL/Image.py:2847: DecompressionBombWarning: Image size (137239200 pixels) exceeds limit of 89478485 pixels, could be decompression bomb DOS attack.
Processing |################################| 6/6
                                     Error
Reference         Prediction              
201-OUTPUT-GT.png 201-OUTPUT-GT.png    0.0
202-OUTPUT-GT.png 202-OUTPUT-GT.png    0.0
203-OUTPUT-GT.png 203-OUTPUT-GT.png    0.0
204-OUTPUT-GT.png 204-OUTPUT-GT.png    0.0
205-OUTPUT-GT.png 205-OUTPUT-GT.png    0.0
206-OUTPUT-GT.png 206-OUTPUT-GT.png    0.0
==============================
Global error for task 2: 0.000
```

:warning: Because the PNG files are large, you may get a warning from PIP that you can safely ignore:  
`DecompressionBombWarning: Image size (137239200 pixels) exceeds limit of 89478485 pixels, could be decompression bomb DOS attack.`

### Files generated in output folder
When processing directories, the output directory will contain the following files:

- `global_error.json`:  
  Easy to parse file for global score with a summary of files analyzed.
- `global_hd95.csv`:  
  HD95 metrics for each image.


## Evaluation tool for task 3: Locate graticule lines intersections
*Please refer to https://icdar21-mapseg.github.io/tasks/task3/ for the description of the task.*

### Metric
This tool computes an aggregate indicator of detection and localization accuracy for each set of points (map sheet).
More precisely, we compute and plot the F-score (with beta=0.5 to penalize false detections more than missed elements) of the correctly detected points across a range of distance thresholds (50 pixels here).
The global indicator is the average of all individual scores.

For more details about the metric, please refer to the [evaluation details for task 3](https://icdar21-mapseg.github.io/tasks/task3/#metrics).


### Tool sample usage
The script supports comparing either:

* a predicted detection to a reference detection (as two CSV files)
* a reference directory to a reference detection
  In this case, reference files are expected to end with ``-OUTPUT-GT.csv``, and prediction files with ``-OUTPUT-PRED.csv``.

Comparing two files:

```
$ icdar21-mapseg-eval T3 201-OUTPUT-GT.csv 201-OUTPUT-PRED.csv output_dir
201-OUTPUT-PRED.csv - Score: 1.000
```

Comparing two directories:

```
$ icdar21-mapseg-eval T3 ./3-locglinesinter/validation mypred/t3/validation output_dir
Processing |################################| 6/6
                                       Score
Reference         Predictions               
201-OUTPUT-GT.csv 201-OUTPUT-PRED.csv    1.0
202-OUTPUT-GT.csv 202-OUTPUT-PRED.csv    1.0
203-OUTPUT-GT.csv 203-OUTPUT-PRED.csv    1.0
204-OUTPUT-GT.csv 204-OUTPUT-PRED.csv    1.0
205-OUTPUT-GT.csv 205-OUTPUT-PRED.csv    1.0
206-OUTPUT-GT.csv 206-OUTPUT-PRED.csv    1.0
==============================
Global score for task 3: 1.000
==============================
```


### Files generated in output folder
The output directory will contain something like:
```
201-OUTPUT-PRED.clf.pdf 
201-OUTPUT-PRED.eval.csv
201-OUTPUT-PRED.plot.csv
201-OUTPUT-PRED.plot.pdf
...
global_rad:50_beta:0.50.csv
global_score.json
```

Detail:
- `global_rad:50_beta:0.50.csv`:  
  global score for each pair of files (ground truth, prediction).
- `global_score.json`:  
  Easy to parse file for global score with a summary of files analyzed, and values for evaluation parameters.
- `nnn-OUTPUT-PRED.eval.csv`:  
  CSV file with all intermediate metrics (precision, recall, f_beta, tps, fns, fps, etc.) computed for each detected point.
- `nnn-OUTPUT-PRED.plot.csv`:  
  Source values used to generate the curve to plot.
- `nnn-OUTPUT-PRED.plot.pdf`:  
  Plot of the curve used to compute the global metric.
- `nnn-OUTPUT-PRED.clf.pdf `:  
  A visualization of predictions and their error classification against the ground truth.

You can check the [Demo analysis notebook for task 3](https://github.com/icdar21-mapseg/icdar21-mapseg-eval/blob/main/notebooks/task3_point_detect_eval_demo.ipynb) for further details about the evaluation tools for task 3.  