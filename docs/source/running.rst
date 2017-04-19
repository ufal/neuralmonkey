.. _running:

=======================================
Use SGE cluster array job for inference
=======================================

To speed up the inference, the ``neuralmonkey-run`` binary provides the
``--grid`` option, which can be used when running the program as a SGE array
job.

The ``run`` script make use of the ``SGE_TASK_ID`` and ``SGE_TASK_STEPSIZE``
environment variables that are set in each computing node of the array job.
If the ``--grid`` option is supplied and these variables are present, it runs
the inference only on a subset of the dataset, specified by the variables.

Consider this example ``test_data.ini``::

  [main]
  test_datasets=[<dataset>]
  variables=["path/to/variables.data"]

  [dataset]
  class=dataset.load_dataset_from_files
  s_source="data/source.en"
  s_target_out="out/target.de"

If we want to run a model configured in ``model.ini`` on this dataset, we can
do::

  neuralmonkey-run model.ini test_data.ini

And the program executes the model on the dataset loaded from
``data/source.en`` and stores the results in ``out/target.de``.

If the source file is large or if you use a slow inference method (such as beam
search), you may want to split the source file into smaller parts and execute
the model on all of them in parallel. If you have access to a SGE cluster, you
don't have to do it manually - just create an array job and supply the
``--grid`` option to the program. Now, suppose that the source file contains
100,000 sentences and you want to split it to 100 parts and run it on
cluster. To accomplish this, just run::

  qsub <qsub_options> -t 1-100000:1000 -b y \
  "neuralmonkey-run --grid model.ini test_data.ini"

This will submit 100 jobs to your cluster. Each job will use its
``SGE_TASK_ID`` and ``SGE_TASK_STEPSIZE`` parameters to determine its part of
the data to process. It then runs the inference only on the subset of the
dataset and stores the result in a suffixed file.

For example, if the ``SGE_TASK_ID`` is 3, the ``SGE_TASK_STEPSIZE`` is 100, and
the ``--grid`` option is specified, the inference will be run on lines 201 to
300 of the file ``data/source.en`` and the output will be written to
``out/target.de.0000000200``.

After all the jobs are finished, you just need to manually run::

  cat out/target.de.* > out/target.de

and delete the intermediate files. (Careful when your file has more than 10^10
lines - you need to concatenate the intermediate files in the right order!)
