from __future__ import absolute_import, division, print_function

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

from crowdsourcing.annotations.classification import multiclass_single_binomial_nt as MSB

def parse_args():

    parser = argparse.ArgumentParser(description='Test the person classifier')

    parser.add_argument('--model_path', dest='model_path',
                        help='Path to a trained model.', type=str,
                        required=True)

    parser.add_argument('--priors', dest='priors_path',
                        help='Path to a trained model.', type=str,
                        required=False, default=None)

    args = parser.parse_args()
    return args

def main():

    args = parse_args()
    model_path = args.model_path

    model = MSB.CrowdDatasetMulticlass()
    model.load(model_path,
        load_dataset=True,
        load_workers=True,
        load_images=False,
        load_annos=False,
        load_gt_annos=False,
        load_combined_labels=False
    )

    prios_path = args.priors_path
    if prios_path is not None:
        with open(prios_path) as f:
            prior_data = json.load(f)

        inat_taxon_id_to_class_label = {i : taxon_id for i, taxon_id in enumerate(prior_data.keys())}
        class_probs = np.array(prior_data.values())

        model.num_classes = len(prios_path)
        model.inat_taxon_id_to_class_label = inat_taxon_id_to_class_label
        model.class_probs = class_probs

    image_id = 0
    while True:
        image_id += 1
        image = model._CrowdImageClass_(image_id, model)
        model.images[image_id] = image

        print("####################")
        print("Image: %d" % (image_id,))
        while True:
            print("Input worker id and then their identification")
            worker_id = str(raw_input("Worker id: "))
            if worker_id == '':
                break
            taxon_id = str(raw_input("Taxon id: "))
            if taxon_id == '':
                break
            if taxon_id not in model.inat_taxon_id_to_class_label:
                print("ERROR: Taxon id %s does not exist in the priors" % (taxon_id,))
                continue
            label = model.inat_taxon_id_to_class_label[taxon_id]

            print("Adding label %d (taxon id %s) with prob %0.4f for worker %s to image %d" % (label, taxon_id, model.class_probs[label], worker_id, image_id))

            if worker_id not in model.workers:
                print("Creating a new worker")
                worker = model._CrowdWorkerClass_(worker_id, test_dataset)
                worker.prob_correct = model.prob_correct
                worker.prob_trust = model.prob_trust
                model.workers[worker_id] = worker
            else:
                print("Found existing worker")
                worker = model.workers[worker_id]
            print("Worker prob correct: %0.3f" % (worker.prob_correct,))
            if model.model_worker_trust:
                print("Worker prob trust: %0.3f" % (worker.prob_trust,))

            anno = model._CrowdLabelClass_(image, worker, label)
            image.z[worker_id] = anno
            image.workers.append(worker_id)
            worker.images[image_id] = image

            image.predict_true_labels(avoid_if_finished=False)
            print()
            print("Pred label: %d\tRisk: %0.5f" % (image.y.label, image.risk))
            print()
        print("####################")
        print()


if __name__ == '__main__':

    main()