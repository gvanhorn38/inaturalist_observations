# inaturalist_observations
iNaturalist Observation Analysis

Clone the [crowdsourcing repo](https://github.com/gvanhorn38/crowdsourcing) and add it to your python path.

Convert a database archive to a dataset.
```
python dataset.py \
--archive_dir <path to archive dir> \
--output_dir <path to output directory> \
--convert_leaf_node_keys_to_integers \
--flat_taxonomy \
--max_observations_per_species 1000 \
--max_species 100 \
--add_empirical_class_priors
```

Train a model
```
python train.py \
--dataset_path <path to a training dataset> \
--output_dir <path to a directory to save the model and worker skills>
```

Test a model
```
python test.py \
--model_path <path to a trained model> \
--dataset_path <path to a testing dataset> \
--output_dir <path to a directory to save the observation risks>
```