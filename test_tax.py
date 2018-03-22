from __future__ import absolute_import, division, print_function

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

from crowdsourcing.annotations.classification import multiclass_single_binomial_tax_c as MSB

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
        assert False
        class_probs = np.ones(test_dataset.num_classes) * (1. / test_dataset.num_classes)

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
    # if hasattr(test_dataset, 'inat_taxon_id_to_class_label'):
    #     class_label_to_inat_taxon_id = {v : k for k, v in test_dataset.inat_taxon_id_to_class_label.iteritems()}

    #     with open(os.path.join(output_dir, 'observation_seq_events.txt'), 'w') as f:
    #         print("[Image ID] => (Worker ID, Prob Correct, <Prob Trust>, Taxon ID, Prior Taxon Prob) => ...  ==> [Predicted Taxon ID, Risk]", file=f)
    #         # use the same order as the csv file
    #         for i in range(len(image_data)):
    #             image_id = image_data[i][0]
    #             image = test_dataset.images[image_id]

    #             seq_str = "[%s]" % (image_id,)
    #             for anno in image.z.values():
    #                 worker = anno.worker
    #                 taxon_id = class_label_to_inat_taxon_id[anno.label]
    #                 taxon_prior = test_dataset.class_probs[anno.label]
    #                 # => (worker_id, label, prob_correct, prob_trust)
    #                 if verification_task:
    #                     seq_str += " => (%s, %0.3f, %0.3f, %s, %0.4f)" % (worker.id, worker.prob_correct, worker.prob_trust, taxon_id, taxon_prior)
    #                 else:
    #                     seq_str += " => (%s, %s, %0.3f)" % (worker.id, worker.prob_correct, taxon_id, taxon_prior)

    #             # ==> (predicted label, risk)
    #             pred_taxon_id = class_label_to_inat_taxon_id[image.y.label]
    #             seq_str += " ==> [%s, %0.3f]" % (pred_taxon_id, image.risk)

    #             print(seq_str, file=f)


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