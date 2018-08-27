from __future__ import absolute_import, division, print_function

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

from crowdsourcing.annotations.classification import multiclass_single_binomial as MSB

# https://gist.github.com/vladignatyev/06860ec2040cb497f0f3
def progress_bar(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percentage_complete = round(100.0 * count / float(total), ndigits=1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percentage_complete, '%', status))
    sys.stdout.flush()

def test(model_path, dataset_path, output_dir, verification_task=False):

    print("##############################")
    print("Loading Dataset")
    print()
    s = time.time()

    # Load the trained model
    trained_model = MSB.CrowdDatasetMulticlassSingleBinomial()
    trained_model.load(model_path)

    # BUG: the class_prob and class_prob_prior dict keys get converted to strings.
    # Lets change them back to ints
    #trained_model.class_probs = {int(k) : v for k, v in trained_model.class_probs.items()}
    #trained_model.class_probs_prior = {int(k) : v for k, v in trained_model.class_probs_prior.items()}

    test_dataset = MSB.CrowdDatasetMulticlassSingleBinomial()

    # Copy over the learned worker params
    test_dataset.workers = trained_model.workers
    del trained_model

    # Mark the workers' parameters as finished
    for worker in test_dataset.workers.itervalues():
        worker.params = test_dataset
        worker.finished=True

    # Load in the test images and annotations and any new workers
    test_dataset.load(dataset_path, sort_annos=True, overwrite_workers=False)
    test_dataset.model_worker_trust = verification_task

    e = time.time()
    t = e - s
    print("Loading time: %0.2f seconds (%0.2f minutes) (%0.2f hours)" % (t, t / 60., t / 3600.))
    print()

    print("##############################")
    print("Initializing Dataset")
    print()
    s = time.time()

    # Initialize the class priors.
    # NOTE: this might be different than the train dataset!
    if hasattr(test_dataset, 'global_class_priors'):
        #class_probs = np.clip(test_dataset.global_class_priors, 0.00000001, 0.99999)
        class_probs = test_dataset.global_class_priors
    else:
        #assert False
        #class_probs = np.ones(test_dataset.num_classes) * (1. / test_dataset.num_classes)
        class_probs = {node.key : 1. / test_dataset.num_classes for node in test_dataset.taxonomy.leaf_nodes()}

    test_dataset.class_probs = class_probs
    test_dataset.class_probs_prior = class_probs

    test_dataset.initialize_default_priors()
    test_dataset.initialize_data_structures()


    # Initialize any new workers
    test_dataset.initialize_parameters(avoid_if_finished=True)

    e = time.time()
    t = e - s
    print("Initialization time: %0.2f seconds (%0.2f minutes) (%0.2f hours)" % (t, t / 60., t / 3600.))
    print()

    print("##############################")
    print("Predicting Image Labels & Risks")
    print()
    s = time.time()

    total_images = len(test_dataset.images)
    node_probs_per_image = np.empty((total_images, len(test_dataset.taxonomy.nodes)), dtype=np.float32)
    node_probs_image_ids = []

    # Predict the image labels

    i = 0
    progress_bar(i, total_images)
    for image in test_dataset.images.itervalues():
        class_log_probs = image.predict_true_labels(avoid_if_finished=False)
        node_probs_per_image[i] = image.compute_probability_of_each_node(class_log_probs)

        if not np.isclose(node_probs_per_image[i][0], 1):
            print("ERROR: Root probability not close to 1: %s (%0.5f)" % (image.id, node_probs_per_image[i][0]))

        node_probs_image_ids.append(image.id)
        i += 1
        if i % 1000 == 0:
            progress_bar(i, total_images, "%d images finished" % (i,))
    print()
    e = time.time()
    t = e - s
    print("Predition time: %0.2f seconds (%0.2f minutes) (%0.2f hours)" % (t, t / 60., t / 3600.))
    print()

    print("##############################")
    print("Saving Predictions")
    print()
    s = time.time()

    # Save the node probs
    node_ids = [node.key for node in test_dataset.taxonomy.breadth_first_traversal()]
    np.savez(os.path.join(output_dir, 'observation_node_probs.npz'),
        node_probs=node_probs_per_image,
        image_ids=np.array(node_probs_image_ids),
        node_ids=np.array(node_ids)
    )

    # Save the risks
    image_risks = [(image_id, image.risk) for image_id, image in test_dataset.images.iteritems()]
    image_risks.sort(key=lambda x: x[1])
    image_risks.reverse()
    with open(os.path.join(output_dir, 'observation_risks.txt'), 'w') as f:
      for image_id, risk in image_risks:
        print("%s\t%0.5f" % (image_id, risk), file=f)
    with open(os.path.join(output_dir, 'observation_risks.json'), 'w') as f:
      json.dump(image_risks, f)


    # Make some urls for visualization
    group_size = 25
    risk_groups = [image_risks[i:i+group_size] for i in range(0,len(image_risks), group_size)]
    with open(os.path.join(output_dir, 'identify_urls.txt'), 'w') as f:
        for risk_group in risk_groups:
            obs_ids = ','.join([x[0] for x in risk_group])
            print("https://www.inaturalist.org/observations/identify?reviewed=any&quality_grade=needs_id,research&id=%s" % (obs_ids,), file=f)


    # Make a csv file that contains the observation url, the risk, and identification count.
    ob_url_str = 'https://www.inaturalist.org/observations/%s'

    if hasattr(test_dataset, 'inat_taxon_id_to_class_label'):
        class_label_to_inat_taxon_id = {v : k for k, v in test_dataset.inat_taxon_id_to_class_label.iteritems()}
        header = ["Observation ID", "Risk", "Pred Label", "Number of Identifications", "URL"]
        image_data = [(image_id, image.risk, class_label_to_inat_taxon_id[image.y.label], len(image.z), ob_url_str % (image_id,))
                      for image_id, image in test_dataset.images.iteritems()]

    else:
        header = ["Observation ID", "Risk", "Number of Identifications", "URL"]
        image_data = [(image_id, image.risk, len(image.z), ob_url_str % (image_id,))
                      for image_id, image in test_dataset.images.iteritems()]

    image_data.sort(key=lambda x: x[1])
    image_data.reverse()
    with open(os.path.join(output_dir, 'observation_data.csv'), 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(header)
        csv_writer.writerows(image_data)


    # Make a file that prints out the sequence of events for each observations.
    if hasattr(test_dataset, 'inat_taxon_id_to_class_label'):

        class_label_to_inat_taxon_id = {v : k for k, v in test_dataset.inat_taxon_id_to_class_label.iteritems()}
        integer_id_to_inat_taxon_id = {k : class_label_to_inat_taxon_id[v] for k, v in test_dataset.integer_id_to_orig_node_key.items()}

        with open(os.path.join(output_dir, 'observation_seq_events.txt'), 'w') as f:
            print("[Image ID] => (Worker ID, Skill Traversal) => ...  ==> [Predicted Taxon ID, Risk]", file=f)
            # use the same order as the csv file
            for i in range(len(image_data)):
                image_id = image_data[i][0]
                image = test_dataset.images[image_id]

                seq_str = "Image [%s]" % (image_id,)
                for anno in image.z.values():
                    worker = anno.worker
                    taxon_id = class_label_to_inat_taxon_id[anno.label]

                    # worker skill str
                    worker_skill_traversal_str = "R"
                    z_integer_id = test_dataset.orig_node_key_to_integer_id[anno.label]
                    z_node_list = test_dataset.root_to_node_path_list[z_integer_id]
                    for child_node_index in range(1, len(z_node_list)):
                        z_parent_node = z_node_list[child_node_index - 1]
                        skill_vector_index = test_dataset.internal_node_integer_id_to_skill_vector_index[z_parent_node]
                        skill = worker.skill_vector[skill_vector_index]

                        inat_taxon_id = integer_id_to_inat_taxon_id[z_node_list[child_node_index]]
                        taxon_prior = test_dataset.node_priors[z_node_list[child_node_index]]
                        node_is_leaf = test_dataset.taxonomy.nodes[test_dataset.integer_id_to_orig_node_key[z_node_list[child_node_index]]].is_leaf
                        worker_skill_traversal_str += "->(n: %s (%s), s: %0.3f, p: %0.3f)" % (inat_taxon_id, "L" if node_is_leaf else "I", skill, taxon_prior)

                    # => (worker_id, label, prob_correct, prob_trust)
                    #if verification_task:
                    #    seq_str += "\n\t Worker %s (t %0.3f) => %s" % (worker.id, worker_skill_traversal_str)
                    #else:
                    seq_str += "\n\t Worker %s => %s" % (worker.id, worker_skill_traversal_str)

                # ==> (predicted label, risk)
                pred_taxon_id = class_label_to_inat_taxon_id[image.y.label]
                seq_str += "\nPrediction ==> [%s, %0.3f]\n" % (pred_taxon_id, image.risk)

                print(seq_str, file=f)


    e = time.time()
    t = e - s
    print("Saving time: %0.2f seconds (%0.2f minutes) (%0.2f hours)" % (t, t / 60., t / 3600.))
    print()

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

    parser.add_argument('--verification_task', dest='verification_task',
                        help='Model the labels as a verification task.',
                        required=False, action='store_true', default=False)

    args = parser.parse_args()
    return args

def main():

    args = parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    test(args.model_path, args.dataset_path, args.output_dir, args.verification_task)

if __name__ == '__main__':

    main()