# icdar21-mapseg-eval
Evaluation tools for participants to the ICDAR 21 MapSeg competition.


## 



## Evaluation tool of task 02

See https://icdar21-mapseg.github.io/tasks/task2/ the description of the task. It computes the 95% Haussdorf (HD)
between two binary images.


The script supports comparing either:

* a predicted segmentation to a reference segmentation (as two binary images)
* a reference directory to a reference segmentation (in this case, files are expected
  to have the same names and to finish with ``-OUTPUT-GT.png``.


Comparing two files:

```bash
> icdar21-mapseg-eval T02 reference.png predicted.png
1.6
```

Comparing two directories:

```bash
> icdar21-mapseg-eval T02 ./2-segmaparea/validation ./prediction
Processing |################################| 6/6
            Filename  Error
0  201-OUTPUT-GT.png    0.0
1  202-OUTPUT-GT.png    0.0
2  203-OUTPUT-GT.png    0.0
3  204-OUTPUT-GT.png    0.0
4  205-OUTPUT-GT.png    0.0
5  206-OUTPUT-GT.png    0.0
```
