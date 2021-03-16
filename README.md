# icdar21-mapseg-eval
Evaluation tools for participants to the ICDAR 21 MapSeg competition.


## 



## Evaluation tool of task 2

See https://icdar21-mapseg.github.io/tasks/task2/ the description of the task. It computes the 95% Haussdorf (HD)
between two binary images.


The script supports comparing either:

* a predicted segmentation to a reference segmentation (as two binary images)
* a reference directory to a reference segmentation (in this case, files are expected
  to have the same names and to finish with ``-OUTPUT-GT.png``.


Comparing two files:

```bash
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


## Evaluation tool for task 3: Detection of graticule lines intersections
See https://icdar21-mapseg.github.io/tasks/task3/ the description of the task.

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

You can check the [Demo analysis notebook for task 3](task3_point_detect_eval_demo.ipynb) for further details about the tools we provide.