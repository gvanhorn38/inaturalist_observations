from __future__ import absolute_import, division, print_function

import argparse
import cProfile
import json
import os
import time

import numpy as np

from crowdsourcing.annotations.classification import multiclass_single_binomial as MSB

def test(model_path, dataset_path, output_dir):

    # Load the trained model
    trained_model = MSB.CrowdDatasetMulticlassSingleBinomial()
    trained_model.load(model_path)

    # BUG: the class_prob and class_prob_prior dict keys get converted to strings.
    # Lets change them back to ints
    trained_model.class_probs = {int(k) : v for k, v in trained_model.class_probs.items()}
    trained_model.class_probs_prior = {int(k) : v for k, v in trained_model.class_probs_prior.items()}

    test_dataset = MSB.CrowdDatasetMulticlassSingleBinomial()

    # Copy over the learned worker params
    test_dataset.workers = trained_model.workers
    for worker in test_dataset.workers.itervalues():
        worker.finished=True

    # Load in the test images and annotations and any new workers
    test_dataset.load(dataset_path, sort_annos=True, overwrite_workers=False)

    # Copy over the learned global parameters (need to do this after loading the test dataset)
    test_dataset.copy_parameters_from(trained_model)

    # Initialize any new workers
    test_dataset.initialize_parameters(avoid_if_finished=True)

    # Predict the image labels
    for image_id, image in test_dataset.images.iteritems():
      image.predict_true_labels(avoid_if_finished=False)

    # Save the risks
    image_risks = [(image_id, image.risk) for image_id, image in test_dataset.images.iteritems()]
    image_risks.sort(key=lambda x: x[1])
    image_risks.reverse()
    with open(os.path.join(output_dir, 'observation_risks.txt'), 'w') as f:
      for image_id, risk in image_risks:
        print("%s\t%0.5f" % (image_id, risk), file=f)
    with open(os.path.join(output_dir, 'observation_risks.json'), 'w') as f:
      json.dump(image_risks, f)

def parse_args():

    parser = argparse.ArgumentParser(description='Test the person classifier')

    parser.add_argument('--model_path', dest='model_path',
                        help='Path to a trained model.', type=str,
                        required=True)

    parser.add_argument('--dataset_path', dest='dataset_path',
                        help='Path to the testing dataset json file.', type=str,
                        required=True)

    parser.add_argument('--output_dir', dest='output_dir',
                          help='Path to an output directory to save the observation risks.', type=str,
                          required=True)

    args = parser.parse_args()
    return args

def main():

    args = parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    test(args.model_path, args.dataset_path, args.output_dir)

if __name__ == '__main__':

    main()