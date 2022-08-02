# Fair TREC 2021 Public Code

This code is example implementations of the TREC 2021 metrics.  To make it work,
you need to put data in `data`, as downloaded from TREC, and the runs in `runs`.

The `environment.yml` file defines a Conda environment that contains all required
dependencies.  It is **strongly recommended** to use this environment for running
the metric code.

If you store your runs in `runs` with a `.gz` extension, they will be found by
the evaluation code to run your own evaluations.

**Note:** The Task 1 metrics are ready to use, but take significant time and
memory to run. Still finishing the Task 2 metric updates and working on Task 1
performance improvements.

**Note:** the final metrics use gender data.  See the overview paper for crucial
notes on the limitations, warnings, and limitations, and ethical considerations
of this data.

Save the data files from [Fair TREC](https://fair-trec.github.io) into the
`data` directory before running this code.  You can find all data files at
<https://data.boisestate.edu/library/Ekstrand/TRECFairRanking/>.

## Oracle Runs

The `oracle-runs.py` file generates oracle runs from the training queries, with
a specified precision.  For example, to generate runs for Task 2 with precision
0.9, run:

    python .\oracle-runs.py -p 0.9 --task2 -o runs/task2-prec09.tsv

These are useful for seeing the output format, and for testing metrics.

## Metrics and Evaluation

The metrics are defined in modules under `wptrec`.  They require alignment data that
is computed by the following scripts and stored in `data/metric-tables`:

- `PageAlignments.py` computes the per-page alignments and saves them to disk
- `Task1Alignment.py` uses the page alignments, qrels, and background distributions
  to compute Task 1 target distributions.
- `Task2Alignment.py` does the same thing for Task 2.

Each of these scripts is paired with a Jupyter notebook with nicely-rendered outputs
and explanations (the Python file is actually the Jupytext plain-text version). We
recommend the notebook for editing, but the Python script can be run directly to
produce the outputs.

The Evaluation notebooks use this alignment data to evaluate runs.

All task-specific notebooks have a `DATA_MODE` constant that controls whether they
work on the training data or the eval data.

You can download precomputed metric tables for the training data from the `metric-tables` 
folder in <https://drive.google.com/drive/folders/1kjw0HgRor2aEupYyie5orB7t_22khTeD>.

## License

All code is licensed under the MIT License.