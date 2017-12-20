# inaturalist_observations
iNaturalist Observation Analysis

Convert a database archive to a dataset.
```
python dataset.py \
--archive_dir <path to archive dir> \
--output_dir <path to output directory> \
--convert_leaf_node_keys_to_integers \
--flat_taxonomy
```

Train a model
```
python train.py \
--dataset_path <path to a training dataset> \
--output_dir <path to a directory to save the model>
```