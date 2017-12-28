# inaturalist_observations
iNaturalist Observation Analysis

Clone the [crowdsourcing repo](https://github.com/gvanhorn38/crowdsourcing) and add it to your python path.

Create the json dataset files from an iNat archive.
```
python dataset.py \
--archive_dir /Users/GVH/Desktop/inat_odes/archive \
--output_dir /Users/GVH/Desktop/inat_odes/datasets
```

Predict image labels.
```
python train.py \
--dataset_path <path to the observation label prediction dataset> \
--output_dir <path to a directory to save the model> \
--verification_task
```

Learn worker skills, and use the predicted images labels as fixed.
```
python train.py \
--dataset_path <path to the observation label prediction dataset> \
--output_dir <path to a directory to save the model> \
--verification_task \
--combined_labels <path to the model that was trained to predict the image labels>
```

Test a model by predicting the risks of the observations.
```
python test.py \
--model_path <path to a trained model> \
--dataset_path <path to a testing dataset> \
--output_dir <path to a directory to save the observation risks> \
--verification_task
```