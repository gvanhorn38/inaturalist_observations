from __future__ import absolute_import, division, print_function

import argparse
import cProfile
import csv
import os
import time

import numpy as np

from crowdsourcing.annotations.classification import multiclass_single_binomial_tax as MSB


PROB_CORRECT_PRIOR = 0.8
PROB_CORRECT = 0.8 # 0.1 #
POOLED_PRIOR_STRENGTH = 10
GLOBAL_PRIOR_STRENGTH = 2.5
GLOBAL_CLASS_PROB_PRIOR_STRENGTH = 5
GLOBAL_TRUST_PRIOR_STRENGTH = 15
POOLED_TRUST_STENGTH = 10
PROB_TRUST_PRIOR = 0.8
PROB_TRUST = 0.8 # 0.999999 #

def train(dataset_path, output_dir, estimate_priors_automatically=False, verification_task=False, combined_labels_path=None):

    dataset = MSB.CrowdDatasetMulticlassSingleBinomial(
        name='inat_single_binomial',
        learn_worker_params=True,
        learn_image_params=False,
        computer_vision_predictor=None,
        image_dir=None,
        min_risk=0.05,
        estimate_priors_automatically=estimate_priors_automatically,

        class_probs=None,
        class_probs_prior_beta = GLOBAL_CLASS_PROB_PRIOR_STRENGTH,
        class_probs_prior = None,

        prob_correct_prior_beta = GLOBAL_PRIOR_STRENGTH,
        prob_correct_prior = PROB_CORRECT_PRIOR,
        prob_correct_beta = POOLED_PRIOR_STRENGTH,
        prob_correct = PROB_CORRECT,

        prob_trust_prior_beta = GLOBAL_TRUST_PRIOR_STRENGTH,
        prob_trust_prior = PROB_TRUST_PRIOR,
        prob_trust_beta=POOLED_TRUST_STENGTH,
        prob_trust=PROB_TRUST,

        model_worker_trust=verification_task,

        debug=2
    )

    print("##############################")
    print("Loading Dataset")
    print()
    s = time.time()
    dataset.load(dataset_path, sort_annos=verification_task, )

    # Load in combined labels that were predicted previously.
    if combined_labels_path is not None:
        print("Loading combined labels")
        dataset.load(combined_labels_path,
                     overwrite_workers=False,
                     load_dataset=False,
                     load_workers=False,
                     load_images=False,
                     load_annos=False,
                     load_gt_annos=False,
                     load_combined_labels=True)

        # Mark the images as finished.
        nf = 0
        for image in dataset.images.itervalues():
            if image.y is not None:
                image.finished = True
                nf += 1
        print("Loaded labels for %d / %d images" % (nf, len(dataset.images)))
        print()

    e = time.time()
    t = e - s
    print("Loading time: %0.2f seconds (%0.2f minutes) (%0.2f hours)" % (t, t / 60., t / 3600.))
    print()

    print("##############################")
    print("Initializing Dataset")
    print()
    s = time.time()
    if hasattr(dataset, 'global_class_priors'):
        #class_probs = np.clip(dataset.global_class_priors, 0.00000001, 0.99999)
        class_probs = dataset.global_class_priors
    else:
        assert False
        class_probs = np.ones(dataset.num_classes) * (1. / dataset.num_classes)

    dataset.class_probs = class_probs
    dataset.class_probs_prior = class_probs

    dataset.initialize_default_priors()
    dataset.initialize_data_structures()

    e = time.time()
    t = e - s
    print("Initialization time: %0.2f seconds (%0.2f minutes) (%0.2f hours)" % (t, t / 60., t / 3600.))
    print()

    print("##############################")
    print("Estimating Parameters")
    print()
    s = time.time()
    dataset.estimate_parameters(avoid_if_finished=True)
    e = time.time()
    t = e - s
    print("Estimation time: %0.2f seconds (%0.2f minutes) (%0.2f hours)" % (t, t / 60., t / 3600.))
    print()

    print("##############################")
    print("Saving Learned Parameters")
    print()
    s = time.time()
    model_path = os.path.join(output_dir, 'model.json')
    dataset.save(model_path)
    e = time.time()
    t = e - s
    print("Saving time: %0.2f seconds (%0.2f minutes) (%0.2f hours)" % (t, t / 60., t / 3600.))
    print()

    # Not sure what we want to do here...
    # Make a csv file with the user parameter estimates
    # if verification_task:
    #     header = ['User ID', 'Prob Correct', 'Prob Trust Others']
    #     user_skills = [(worker_id, worker.prob_correct, worker.prob_trust) for worker_id, worker in dataset.workers.iteritems()]
    # else:
    #     header = ['User ID', 'Prob Correct']
    #     user_skills = [(worker_id, worker.prob_correct) for worker_id, worker in dataset.workers.iteritems()]
    # user_skills.sort(key=lambda x: x[1])
    # user_skills.reverse()
    # with open(os.path.join(output_dir, 'user_skills.csv'), 'w') as f:
    #     csv_writer = csv.writer(f)
    #     csv_writer.writerow(header)
    #     csv_writer.writerows(user_skills)

def parse_args():

    parser = argparse.ArgumentParser(description='Test the person classifier')

    parser.add_argument('--dataset_path', dest='dataset_path',
                        help='Path to the training dataset json file.', type=str,
                        required=True)

    parser.add_argument('--output_dir', dest='output_dir',
                          help='Path to an output directory to save the model.', type=str,
                          required=True)

    parser.add_argument('--estimate_priors_automatically', dest='estimate_priors_automatically',
                        help='Estimate the global priors automatically.',
                        required=False, action='store_true', default=False)

    parser.add_argument('--verification_task', dest='verification_task',
                        help='Model the labels as a verification task.',
                        required=False, action='store_true', default=False)

    parser.add_argument('--combined_labels', dest='combined_labels_path',
                        help='Path to a file that contains predicted labels for these images. This will mark the images as finished.',
                        type=str, required=False, default=None)

    parser.add_argument('--profile', dest='profile',
                        help='Run the code through cProfile',
                        required=False, action='store_true', default=False)

    args = parser.parse_args()
    return args

def main():

    args = parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.profile:
        pr = cProfile.Profile()
        pr.enable()
        train(args.dataset_path, args.output_dir,
            estimate_priors_automatically=args.estimate_priors_automatically,
            verification_task=args.verification_task
        )
        pr.disable()
        pr.print_stats(sort='time')
    else:
        train(args.dataset_path, args.output_dir,
            estimate_priors_automatically=args.estimate_priors_automatically,
            verification_task=args.verification_task,
            combined_labels_path=args.combined_labels_path
        )

if __name__ == '__main__':

    main()