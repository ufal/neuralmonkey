# Examples

This directory contains example configuration files for toy experiments with
Neuralmonkey.

These experiments use example data, which can be downloaded by the download
script inside the example's dedicated data subdirectory. To prepare the data
for e.g. the tagging example, just run the script from inside the
`data/tagging` directory.

- `translation.ini` an example translation model. For running this model on
  GPU, it should have at least 6 GB RAM.

- `tagging.ini` an example part-of-speech tagger model.

- `language_model.ini` and example INI for training a language model.

There are also additional examples in the `_old` directory but they have not
been updated for a while and therefore won't be functional without further
edits.