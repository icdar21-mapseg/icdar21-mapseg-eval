# ICDAR 21 MapSeg Evaluation Tools

Welcome to the repository of the evaluation tools for ICDAR 2021 Competition in Historical Map Segmentation (MapSeg).

Here you can learn how to install and use the evaluation tools.
Feel free to report any problem in the issue tracker, and we will fix it as soon as possible.

## Installation
The evaluation tools are packaged as a Python package.

You need a Python >= 3.7.1 to run these tools.

You can install the evaluation tools with the following command:
```shell
pip install icdar21-mapseg-eval
```

You can test your installation by running the following command which should display the online help of the command line tool:
```shell
icdar21-mapseg-eval --help
```

Should you need extra features, you may also import the package and use the exported functions.
```python
from icdar21_mapseg_eval import COCO
task1_metrics = COCO(ground_truth_segmentation, predicted_segmentation)
print(task1_metrics)
```

## Usage
There is a single command line program to run the evaluation for all 3 tasks.
To select the task you want to run the evaluation for, you need to follow this syntax:
```shell
icdar21-mapseg-eval {T1,T2,T3} path_to_reference path_to_prediction path_to_outputs
```
where:

- `{T1,T2,T3}` means "select either `T1` or `T2` or `T3`.
- `path_to_reference` is the path to either a file or a directory.
- `path_to_prediction` is the path to either a file or a directory.
- `path_to_outputs` is the path to the directory where results will be stored.


### Evaluation tool of task 1: Detect building blocks
Please refer to https://icdar21-mapseg.github.io/tasks/task1/ for the description of the task.

This tool computes the [COCO PQ score](https://cocodataset.org/#panoptic-eval) associated to the instance segmentation returned by your system.
Please note that as we have only 1 "thing" class and not "stuff" class, we provide indicators only for the building blocks class.
We have a custom implementation which is fully compliant with the COCO PQ evaluation code.
The need for a custom evaluation was driven by data format incompatibilities.

The script supports comparing either:

* a predicted segmentation to a reference segmentation (as two binary images in PNG or two label maps in TIFF16).
* a reference directory to a reference segmentation.  
  In this case, reference files are expected to end with ``-OUTPUT-GT.png``, and prediction files with ``-OUTPUT-PRED.png`` or ``-OUTPUT-*.tiff``.


Comparing two files:

```
> icdar21-mapseg-eval T1 reference.png predicted.png
COCO PQ  COCO SQ  COCO RQ
1.0      1.0      1.0
```

Comparing two directories:

```
> icdar21-mapseg-eval T1 ./1-detbblocks/validation ./prediction
Processing |################################| 6/6
            Filename  COCO PQ  COCO SQ  COCO RQ
0  201-OUTPUT-GT.png    1.0    1.0      1.0
```


### Evaluation tool of task 2: Segment map content area
Please refer to https://icdar21-mapseg.github.io/tasks/task2/ for the description of the task.

This tool computes the 95% Haussdorff (HD) between two binary images.


The script supports comparing either:

* a predicted segmentation to a reference segmentation (as two binary images)
* a reference directory to a reference segmentation
  In this case, reference files are expected to end with ``-OUTPUT-GT.png``, and prediction files with ``-OUTPUT-PRED.png``.


Comparing two files:

```
> icdar21-mapseg-eval T2 reference.png predicted.png
1.6
```

Comparing two directories:

```
> icdar21-mapseg-eval T2 ./2-segmaparea/validation ./prediction
Processing |################################| 6/6
            Filename  Error
0  201-OUTPUT-GT.png    0.0
1  202-OUTPUT-GT.png    0.0
2  203-OUTPUT-GT.png    0.0
3  204-OUTPUT-GT.png    0.0
4  205-OUTPUT-GT.png    0.0
5  206-OUTPUT-GT.png    0.0
```


### Evaluation tool for task 3: Locate graticule lines intersections
Please refer to https://icdar21-mapseg.github.io/tasks/task3/ for the description of the task.

This tool computes an aggregate indicator of detection and localization accuracy for each set of points (map sheet).
The global indicator is the average of all individual scores.

The tool is called by passing the "T3" option to the command line.
You can either process two individual outputs, or two directories containing multiple results.


Here is a sample showing how you can compare the predictions generated in the directory `validation_pred` 
against the ground truth in `validation_gt` and store all results and analysis under `output_results_dir`.
```
> icdar21_mapseg_eval T3 validation_gt  validation_pred output_results_dir
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

The directory `output_results_dir` will contain something like:
```
201-OUTPUT-PRED.clf.pdf 
201-OUTPUT-PRED.eval.csv
201-OUTPUT-PRED.plot.csv
201-OUTPUT-PRED.plot.pdf
...
global_rad:50_beta:0.50.csv
global_score.json
```

Content:
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

You can check the [Demo analysis notebook for task 3](task3_point_detect_eval_demo.ipynb) for further details about the evaluation tools for task 3 we provide.