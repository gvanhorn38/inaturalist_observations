from __future__ import absolute_import, division, print_function

import argparse
import cProfile
import os
import time

import numpy as np

from crowdsourcing.annotations.classification import multiclass_single_binomial as MSB


ESTIMATE_PRIORS_AUTOMATICALLY = False
PROB_CORRECT_PRIOR = 0.8
POOLED_PRIOR_STRENGTH = 10
GLOBAL_PRIOR_STRENGTH = 2.5
GLOBAL_CLASS_PROB_PRIOR_STRENGTH = 5
GLOBAL_TRUST_PRIOR_STRENGTH = 15
POOLED_TRUST_STENGTH = 10
PROB_TRUST_PRIOR = 0.8
PROB_TRUST = 0.8

VERIFICATION_TASK = False
DEPENDENT_LABELS = True

def train(dataset_path, output_dir):

    dataset = MSB.CrowdDatasetMulticlassSingleBinomial(
        name='inat_single_binomial',
        learn_worker_params=True,
        learn_image_params=False,
        computer_vision_predictor=None,
        image_dir=None,
        min_risk=0.05,
        estimate_priors_automatically=ESTIMATE_PRIORS_AUTOMATICALLY,

        class_probs=None,
        class_probs_prior_beta = GLOBAL_CLASS_PROB_PRIOR_STRENGTH,
        class_probs_prior = None,

        prob_correct_prior_beta = GLOBAL_PRIOR_STRENGTH,
        prob_correct_prior = PROB_CORRECT_PRIOR,
        prob_correct_beta = POOLED_PRIOR_STRENGTH,

        prob_trust_prior_beta = GLOBAL_TRUST_PRIOR_STRENGTH,
        prob_trust_prior = PROB_TRUST_PRIOR,
        prob_trust_beta=POOLED_TRUST_STENGTH,
        prob_trust=PROB_TRUST,

        model_worker_trust=VERIFICATION_TASK,
        recursive_trust=DEPENDENT_LABELS,

        debug=2
    )

    print("##############################")
    print("Loading Dataset")
    print()
    s = time.time()
    dataset.load(dataset_path, sort_annos=VERIFICATION_TASK)
    e = time.time()
    t = e - s
    print("Loading time: %0.2f seconds (%0.2f minutes) (%0.2f hours)" % (t, t / 60., t / 3600.))
    print()

    print("##############################")
    print("Initializing Dataset")
    print()
    s = time.time()
    if hasattr(dataset, 'global_class_probs'):
        class_probs = np.clip(dataset.global_class_probs, 0.00000001, 0.99999)
    else:
        class_probs = np.ones(dataset.num_classes) * (1. / dataset.num_classes)

    dataset.class_probs = {cid : p for cid, p in enumerate(class_probs)}
    dataset.class_probs_prior = {cid : p for cid, p in enumerate(class_probs)}

    dataset.initialize_default_priors()
    e = time.time()
    t = e - s
    print("Initialization time: %0.2f seconds (%0.2f minutes) (%0.2f hours)" % (t, t / 60., t / 3600.))
    print()

    print("##############################")
    print("Estimating Parameters")
    print()
    s = time.time()
    dataset.estimate_parameters(avoid_if_finished=False)
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

    # Make a file with the user skills
    user_skills = [(worker_id, worker.taxonomy.root_node.data['prob_correct']) for worker_id, worker in dataset.workers.iteritems()]
    user_skills.sort(key=lambda x: x[1])
    user_skills.reverse()
    with open(os.path.join(output_dir, 'user_skill.txt'), 'w') as f:
        for worker_id, skill in user_skills:
            print("%s\t%0.5f" % (worker_id, skill), file=f)

def parse_args():

    parser = argparse.ArgumentParser(description='Test the person classifier')

    parser.add_argument('--dataset_path', dest='dataset_path',
                        help='Path to the training dataset json file.', type=str,
                        required=True)

    parser.add_argument('--output_dir', dest='output_dir',
                          help='Path to an output directory to save the model.', type=str,
                          required=True)

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
        train(args.dataset_path, args.output_dir)
        pr.disable()
        pr.print_stats(sort='time')
    else:
        train(args.dataset_path, args.output_dir)

if __name__ == '__main__':

    main()