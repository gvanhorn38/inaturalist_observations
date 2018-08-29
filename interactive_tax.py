from __future__ import absolute_import, division, print_function

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

from crowdsourcing.annotations.classification import multiclass_single_binomial as MSB

def parse_args():

    parser = argparse.ArgumentParser(description='Interactively test the person classifier')

    parser.add_argument('--model_path', dest='model_path',
                        help='Path to a trained model.', type=str,
                        required=True)

    args = parser.parse_args()
    return args

def main():

    args = parse_args()
    model_path = args.model_path

    model = MSB.CrowdDatasetMulticlassSingleBinomial()
    model.load(model_path,
        load_dataset=True,
        load_workers=True,
        load_images=False,
        load_annos=False,
        load_gt_annos=False,
        load_combined_labels=False
    )

    # Initialize the class priors.
    # NOTE: this might be different than the train dataset!
    if hasattr(model, 'global_class_priors'):
        class_probs = model.global_class_priors
    else:
        #assert False
        #class_probs = np.ones(model.num_classes) * (1. / model.num_classes)
        class_probs = {node.key : 1. / model.num_classes for node in model.taxonomy.leaf_nodes()}

    model.class_probs = class_probs
    model.class_probs_prior = class_probs

    model.initialize_default_priors()
    model.initialize_data_structures()

    if hasattr(model, 'inat_taxon_id_to_class_label'):
        inat_taxon_id_to_class_label = model.inat_taxon_id_to_class_label
    else:
        # Assume that the node keys are the inat taxon ids
        inat_taxon_id_to_class_label = {k : k for k in model.taxonomy.nodes}

        #print("ERROR: inat_taxon_id_to_class_label needs to be present")
        #return

    label_to_inat_taxon_id = {v : k for k, v in model.inat_taxon_id_to_class_label.items()}

    image_id = 0
    while True:
        image_id += 1
        image = model._CrowdImageClass_(image_id, model)
        model.images[image_id] = image

        print("####################")
        print("Image: %d" % (image_id,))
        while True:
            print("-----------------")
            print("Input worker id and then their identification (or the empty string to move to the next image)")
            worker_id = str(raw_input("Worker id: "))
            if worker_id == '':
                break
            taxon_id = str(raw_input("Taxon id: "))
            if taxon_id == '':
                break
            if taxon_id not in inat_taxon_id_to_class_label:
                print("ERROR: Taxon id %s does not exist in the priors" % (taxon_id,))
                continue

            label = inat_taxon_id_to_class_label[taxon_id]
            node = model.taxonomy.nodes[label]

            print("Worker %s adding an identification of taxon %s with prior = %0.4f" % (worker_id, taxon_id, node.data['prob']))

            if worker_id not in model.workers:
                print("Creating a new worker")
                worker = model._CrowdWorkerClass_(worker_id, model)
                worker.skill_vector = np.copy(model.default_skill_vector)
                worker.prob_trust = model.prob_trust
                model.workers[worker_id] = worker
            else:
                print("Found existing worker")
                worker = model.workers[worker_id]


            print("Worker %s Skills:" % worker_id)
            skill_str = "\tR"
            z_integer_id = model.orig_node_key_to_integer_id[label]
            z_node_list = model.root_to_node_path_list[z_integer_id]
            for child_node_index in range(1, len(z_node_list)):
                z_parent_node = z_node_list[child_node_index - 1]
                skill_vector_index = model.internal_node_integer_id_to_skill_vector_index[z_parent_node]
                skill = worker.skill_vector[skill_vector_index]

                child_inat_id = label_to_inat_taxon_id[model.integer_id_to_orig_node_key[z_node_list[child_node_index]]]
                ind = "\n\t" + "   " * (child_node_index - 1) + "|"
                skill_str += ind + "->(taxon id %s, skill %0.3f)" % (child_inat_id, skill)
            print(skill_str)
            #if model.model_worker_trust:
            #    print("\tWorker prob trust: %0.3f" % (worker.prob_trust,))


            # Create the annotation
            anno = model._CrowdLabelClass_(image, worker, label)
            image.z[worker_id] = anno
            image.workers.append(worker_id)
            worker.images[image_id] = image

            # Predict the true label
            class_log_likelihoods = image.predict_true_labels(avoid_if_finished=False)
            node_probabilities = image.compute_probability_of_each_node(class_log_likelihoods)

            y_label = image.y.label
            y_node = model.taxonomy.nodes[y_label]

            print()
            print("Predicted taxon: %s\tRisk: %0.5f" % (label_to_inat_taxon_id[y_label], image.risk))
            print("Taxonomic Probabilities:")
            tax_prob_str = ""
            y_integer_id = model.orig_node_key_to_integer_id[y_label]
            y_node_list = model.root_to_node_path_list[y_integer_id]
            for node_index in range(len(y_node_list)):
                prob = node_probabilities[y_node_list[node_index]]
                inat_id = label_to_inat_taxon_id[model.integer_id_to_orig_node_key[y_node_list[node_index]]]

                if node_index == 0:
                    ind = "\t"
                    tax_prob_str += ind + "(taxon id %s, prob %0.3f)" % (inat_id, prob)
                else:
                    ind = "\n\t" + "   " * (node_index) + "|"
                    tax_prob_str += ind + "->(taxon id %s, prob %0.3f)" % (inat_id, prob)

            # the y_node is always a leaf node...
            # if not y_node.is_leaf:

            #     offset = len(y_node_list)
            #     tax_prob_str += "\n" + "\t" + "   " * offset + "---Children Under The Predicted Node---"

            #     for child_key in y_node.children:
            #         child_integer_id = model.orig_node_key_to_integer_id[child_key]
            #         prob = node_probabilities[child_integer_id]
            #         inat_id = label_to_inat_taxon_id[model.integer_id_to_orig_node_key[child_integer_id]]

            #         ind = "\n\t" + "   " * (offset) + "|"
            #         tax_prob_str += ind + "->(taxon id %s, prob %0.3f)" % (inat_id, prob)

            print(tax_prob_str)

            # Print out the sibling probabilities
            y_parent_node = y_node.parent
            for child_node in y_parent_node.children.values():
                integer_id = model.orig_node_key_to_integer_id[child_node.key]
                prob = node_probabilities[integer_id]

                inat_id = label_to_inat_taxon_id[child_node.key]
                print("Sibling node %s: prob %0.5f" % (inat_id, prob))

            print()
            #print("Pred label: %s\tRisk: %0.5f" % (image.y.label, image.risk))
            #print()
        print("####################")
        print()


if __name__ == '__main__':

    main()