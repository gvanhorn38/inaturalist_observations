"""
Data Formats:

identification{
  u'category': u'improving',
  u'created_at': u'2011-12-04 05:08:30.290905',
  u'current': u't',
  u'id': u'53829',
  u'observation_id': u'41053',
  u'taxon_change_id': None,
  u'taxon_id': u'47792',
  u'user_id': u'3891'
}

observation{
  u'community_taxon_id': u'47792',
  u'created_at': u'2011-12-04 05:08:29.229652',
  u'id': u'41053',
  u'latitude': u'38.1850031143',
  u'longitude': u'-122.1861015833',
  u'user_id': u'3891'
}

users = [<user_id>]

observation_photo{
  u'medium_url': u'http://static.inaturalist.org/photos/1596277/medium.jpg?1425545517',
  u'observation_id': u'1272361'
}

taxa{
  u'ancestry': u'48460',
  u'iconic_taxon_id': u'1',
  u'id': u'1',
  u'is_active': u't',
  u'rank': u'kingdom'
}

"""



from __future__ import absolute_import, division, print_function

import argparse
from collections import Counter
import copy
import datetime
import json
import os
import random
import sys

import numpy as np

from crowdsourcing.util.taxonomy import Node, Taxonomy

class iNaturalistDataset():

  def __init__(self, observations, observation_photos, identifications, taxa, users):
    self.observations = observations
    self.observation_photos = observation_photos
    self.identifications = identifications
    self.taxa = taxa
    self.users = users

  def simple_stats(self):

    print("##############################")
    print("Archive Stats")
    print()

    # Print the number of observers and the number of identifers
    num_observations = len(self.observations)
    num_observers = len(set([obs['user_id'] for obs in self.observations]))
    num_identifications = len(self.identifications)
    num_identifiers = len(set([iden['user_id'] for iden in self.identifications]))

    print("Number of observers: %d" % num_observers)
    print("Number of observations: %d" % num_observations)
    print("Number of identifications: %d" % num_identifications)
    print("Number of identifiers: %d" % num_identifiers)


    # How many identifications have been made by people?
    user_identification_counts = {}
    for ident in self.identifications:
      user_id = ident['user_id']
      user_identification_counts.setdefault(user_id, 0)
      user_identification_counts[user_id] += 1

    identification_counts = np.array(user_identification_counts.values())
    print("Mean identifications per user: %0.3f" % np.mean(identification_counts))
    print("Median identifications per user: %d" % np.median(identification_counts))
    print("Max identifications per user: %d" % np.max(identification_counts))

    edges = [0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    for edge in edges:
      print("Number of users with >= %d identifications: %d" % (edge, np.sum(identification_counts >= edge)))

    print()
    number_of_non_current_taxa = len([t for t in self.taxa if t['is_active'] == 't'])
    print("Number of taxa: %d" % (len(self.taxa),))
    print("Number of non-active taxa: %d" % (number_of_non_current_taxa,))


    # Breakdown the identifications by taxonomic rank
    taxon_id_to_rank = {taxon['id'] : taxon['rank'] for taxon in self.taxa}
    rank_identification_counts = {}
    for ident in self.identifications:
      taxon_id = ident['taxon_id']
      rank = taxon_id_to_rank[taxon_id]
      rank_identification_counts.setdefault(rank, 0)
      rank_identification_counts[rank] += 1

    print()
    print("Number of identifications at different ranks:")
    for rank, count in rank_identification_counts.items():
      print("%s : %d" % (rank, count))

    # Print some info about the taxonomy itself:
    taxon_id_to_taxon = {taxon['id'] : taxon for taxon in self.taxa}
    rank_counts = Counter([r for taxon_id, r in taxon_id_to_rank.items() if taxon_id_to_taxon[taxon_id]['is_active'] == 't'])
    print()
    print("Taxonomy Info (only active nodes)")
    for rank, count in rank_counts.items():
      print("%s: %d nodes" % (rank, count))


    # How many observations finish with a community id label (or something that we can use as a groundtruth proxy?)
    obs_with_cid = []
    cid_ranks = []
    for obs in self.observations:
      try:
        int(obs['community_taxon_id'])
        obs_with_cid.append(obs)
        cid_ranks.append(taxon_id_to_rank[obs['community_taxon_id']])
      except:
        pass
    print()
    print("%d / %d observations have community ids" % (len(obs_with_cid), len(self.observations)))
    cid_rank_counts = Counter(cid_ranks)
    for rank, count in cid_rank_counts.iteritems():
      print("%d community ids at at rank %s" % (count, rank))

    ob_id_to_ob = {}
    obs_ids_with_species_cid = []
    for obs in obs_with_cid:
      if obs['community_taxon_id'] in taxon_id_to_rank:
        if taxon_id_to_rank[obs['community_taxon_id']] == 'species':
          obs_ids_with_species_cid.append(obs['id'])
          ob_id_to_ob[obs['id']] = obs

    # For the observations with a community id at species level, how many species level identifications are there?
    #obs_ids = {ob['id'] : [] for ob in obs_with_cid}

    obs_ids = {ob_id : [] for ob_id in obs_ids_with_species_cid}
    ids_to_obs = {}
    for identification in self.identifications:
      if identification['observation_id'] in obs_ids:
        if identification['current'] == 't':
          if taxon_id_to_rank[identification['taxon_id']] == 'species' or taxon_id_to_rank[identification['taxon_id']] == 'subspecies':
            obs_ids[identification['observation_id']].append(identification)

    num_ids = [len(ids) for ob, ids in obs_ids.iteritems()]
    print()
    print("Stats for obs with community id at species and identifications at species or lower:")
    print("Mean identifications per obs: %0.3f" % np.mean(num_ids))
    print("Median identifications per obs: %d" % np.median(num_ids))
    print("Max identifications per obs: %d" % np.max(num_ids))
    print("Min identifications per obs: %d" % np.min(num_ids))

    bad_obs = []
    for ob, ids in obs_ids.iteritems():
      if len(ids) < 2:
        bad_obs.append(ob)
    print()
    print("There are potentially %d bad observations, these have less than 2 ids" % (len(bad_obs),))
    print(bad_obs)

    # How many observations have multiple identifications from the same person?
    ob_to_user_ids = {ob['id'] : [] for ob in self.observations}
    for ident in self.identifications:
      ob_id = ident['observation_id']
      user_id = ident['user_id']
      ob_to_user_ids[ob_id].append(user_id)

    has_duplicate_users = 0
    for ob_id, user_ids in ob_to_user_ids.iteritems():
      c = Counter(user_ids)
      for user_id, count in c.iteritems():
        if count > 1:
          has_duplicate_users += 1
          break

    print("%d / %d observations have duplicate identifiers" % (has_duplicate_users, len(ob_to_user_ids),))

    print()
    print("End Archive Stats")
    print("##############################")

  def get_image_urls_for_observations(self, observations):
    ob_id_to_urls = {ob['id'] : [] for ob in observations}
    for ob_image in self.observation_photos:
      if ob_image['observation_id'] in ob_id_to_urls:
        ob_id_to_urls[ob_image['observation_id']].append(ob_image['medium_url'])

    # for ob_id, urls in ob_id_to_urls.iteritems():
    #   if len(urls) == 0:
    #     print("ERROR: an image was not found for observation %s" % (ob_id,))
    #     print()

    return ob_id_to_urls

  def enforce_observations_have_images(self, observations, identifications=None):
    """ Remove observations that we don't have images for. Remove identifications that point to
    those observations.
    """
    ob_id_to_urls = self.get_image_urls_for_observations(observations)

    ob_ids_to_keep = set()
    for ob_id, urls in ob_id_to_urls.iteritems():
      if len(urls) > 0:
        ob_ids_to_keep.add(ob_id)

    filtered_observations = [obs for obs in observations if obs['id'] in ob_ids_to_keep]
    if identifications is not None:
      filtered_identifications = [iden for iden in identifications if iden['observation_id'] in ob_ids_to_keep]
    else:
      filtered_identifications = None
    return filtered_observations, filtered_identifications

  def get_taxon_working_set(self, species_ids):
    """ Return a set of all taxon ids that are ancestors of the species ids. This includes the `species_ids`
    """
    taxon_id_to_taxon = {taxon['id'] : taxon for taxon in self.taxa}
    taxon_id_working_set = set(species_ids)
    for taxon_id in species_ids:
      taxon = taxon_id_to_taxon[taxon_id]
      ancestry = taxon['ancestry']
      ancestry_ids = ancestry.split('/')
      for ancestor_id in ancestry_ids:
        taxon_id_working_set.add(ancestor_id)
    return taxon_id_working_set

  def enforce_min_species_identification_count_per_observation(self, observations, identifications, min_species_identifications_per_observation=5):
    """ Remove observations that do not have enough species level identifications.
    Remove identifications that point to the removed observations
    """
    if min_species_identifications_per_observation is None or min_species_identifications_per_observation < 0:
      return observations, identifications, False

    taxon_id_to_rank = {taxon['id'] : taxon['rank'] for taxon in self.taxa}

    obs_species_counts = {ob['id'] : 0 for ob in observations}
    for identification in identifications:
      ob_id = identification['observation_id']
      taxon_id = identification['taxon_id']
      if taxon_id_to_rank[taxon_id] == 'species':
        obs_species_counts[ob_id] += 1
    filtered_observations = [obs for obs in observations if obs_species_counts[obs['id']] >= min_species_identifications_per_observation]

    if len(filtered_observations) == len(observations):
      return observations, identifications, False

    obs_ids = set([obs['id'] for obs in filtered_observations])
    filtered_identifications = []
    for identification in identifications:
      ob_id = identification['observation_id']
      if ob_id in obs_ids:
        filtered_identifications.append(identification)

    return filtered_observations, filtered_identifications, True


  def enforce_min_observation_count_per_species(self, observations, identifications, min_observations_per_species=10, allow_non_species_identifications=False):
    """ Remove species that do not have enough observations.
    Remove identifications that point to the removed species.
    """

    if min_observations_per_species is None or min_observations_per_species < 0:
      return observations, identifications, False

    species_taxon_ids = set([obs['community_taxon_id'] for obs in observations])
    species_obs_counts = {taxon_id : 0 for taxon_id in species_taxon_ids}
    for obs in observations:
      taxon_id = obs['community_taxon_id']
      species_obs_counts[taxon_id] += 1

    filtered_species = set([taxon_id for taxon_id, count in species_obs_counts.items() if count > min_observations_per_species])
    if len(filtered_species) == len(species_taxon_ids):
      return observations, identifications, False

    filtered_observations = [obs for obs in observations if obs['community_taxon_id'] in filtered_species]
    obs_ids = set([obs['id'] for obs in filtered_observations])

    if allow_non_species_identifications:
      taxon_working_set = self.get_taxon_working_set(filtered_species)
    else:
      taxon_working_set = filtered_species

    filtered_identifications = []
    for identification in identifications:
      ob_id = identification['observation_id']
      taxon_id = identification['taxon_id']
      if ob_id in obs_ids and taxon_id in taxon_working_set:
        filtered_identifications.append(identification)

    return filtered_observations, filtered_identifications, True

  def enforce_max_observation_count_per_species(self, observations, identifications, max_observations_per_species=30, sort_by_diversity=True):
    """Restrict the number of observations per taxa. Choose observations that have the most diverse set of species labels or choose
    observations that have the most identifications.
    """
    if max_observations_per_species == None or max_observations_per_species < 0:
      return observations, identifications

    species_taxon_ids = set([obs['community_taxon_id'] for obs in observations])
    ob_id_to_species_idens = {ob['id'] : [] for ob in observations}
    for ident in identifications:
      if ident['taxon_id'] in species_taxon_ids:
        ob_id_to_species_idens[ident['observation_id']].append(ident)

    obs_id_to_var = {obs['id'] : 0 for obs in observations}
    for ob_id, idents in ob_id_to_species_idens.iteritems():
      if sort_by_diversity:
        labels = set([ident['taxon_id'] for ident in idents])
        obs_id_to_var[ob_id] = len(labels)
      else:
        obs_id_to_var[ob_id] = len(idents)

    species_taxon_ids = set([obs['community_taxon_id'] for obs in observations])
    taxa_id_to_obs_ids = {taxon_id : [] for taxon_id in species_taxon_ids}
    for obs in observations:
      taxa_id_to_obs_ids[obs['community_taxon_id']].append(obs['id'])

    selected_obs_ids = []
    for taxa_id, obs_ids in taxa_id_to_obs_ids.iteritems():
      obs_id_var = [(ob_id, obs_id_to_var[ob_id]) for ob_id in obs_ids]
      obs_id_var.sort(key=lambda x: x[1])
      obs_id_var.reverse()
      selected_obs_ids += [x[0] for x in obs_id_var[:max_observations_per_species]]
    selected_obs_ids = set(selected_obs_ids)

    filtered_observations = [ob for ob in observations if ob['id'] in selected_obs_ids]
    filtered_identifications = [iden for iden in identifications if iden['observation_id'] in selected_obs_ids]

    return filtered_observations, filtered_identifications

  def enforce_max_species(self, observations, identifications, species_stats, max_species, allow_non_species_identifications=False):
    """ Select a subset of species, choosing those with lower worker accuracy.
    Return the observations for the subset of species. Also remove any identifications that pointed to
    removed species.
    """

    if max_species is None or max_species < 0:
      return observations, identifications, False

    if len(species_stats) <= max_species:
      return observations, identifications, False

    species_acc = [(taxon_id, stats['acc']) for taxon_id, stats in species_stats.items()]
    species_acc.sort(key=lambda x:x[1])
    selected_species_ids = set([x[0] for x in species_acc[:max_species]])

    filtered_observations = [ob for ob in observations if ob['community_taxon_id'] in selected_species_ids]
    obs_ids = set([ob['id'] for ob in filtered_observations])

    if allow_non_species_identifications:
      taxon_working_set = self.get_taxon_working_set(selected_species_ids)
    else:
      taxon_working_set = selected_species_ids

    filtered_identifications = []
    for identification in identifications:
      ob_id = identification['observation_id']
      taxon_id = identification['taxon_id']
      if ob_id in obs_ids and taxon_id in taxon_working_set:
        filtered_identifications.append(identification)

    return filtered_observations, filtered_identifications, True

  def enforce_max_identifications_per_observation(self, observations, identifications, max_identifications_per_observation=None):
    """ We want a subset of identifications that maximizes the number of
    identifications per worker and that keeps the diversity of identifications.
    So prioritize the identifications that have different taxa, and then
    prioritize the workers that we have seen before.
    """
    if max_identifications_per_observation is None or max_identifications_per_observation < 0:
      return observations, identifications

    ob_id_to_idens = {ob['id'] : [] for ob in observations}
    for identification in identifications:
      ob_id_to_idens[identification['observation_id']].append(identification)

    worker_pool = set()
    identification_ids_to_keep = []
    for ob_id, idens in ob_id_to_idens.iteritems():

      # Keep all of the identifications
      if len(idens) <= max_identifications_per_observation:
        for iden in idens:
          worker_pool.add(iden['user_id'])
          identification_ids_to_keep.append(iden['id'])
        continue

      # We need to sample from the identifications
      # group the idens by taxa
      taxa_to_idens = {}
      for iden in idens:
        taxon_id = iden['taxon_id']
        taxa_to_idens.setdefault(taxon_id, [])
        taxa_to_idens[taxon_id].append(iden)

      num_sampled = 0
      selected_iden_ids = set()
      while num_sampled < max_identifications_per_observation:
        old_num_sampled = num_sampled

        # Select an identification from each taxa
        for taxon_id, all_idens in taxa_to_idens.iteritems():
          # get the non-selected identifications
          non_selected_idens = [iden for iden in all_idens if iden['id'] not in selected_iden_ids]
          if len(non_selected_idens) == 0:
            continue

          # Bias towards workers we have already seen
          iden_ids_to_sample = [(iden['id'], iden['user_id']) for iden in non_selected_idens if iden['user_id'] in worker_pool]

          if len(iden_ids_to_sample) > 0:
            iden_id, worker_id = random.choice(iden_ids_to_sample)
          else:
            iden_id, worker_id = random.choice([(iden['id'], iden['user_id']) for iden in non_selected_idens])

          identification_ids_to_keep.append(iden_id)
          worker_pool.add(worker_id)
          selected_iden_ids.add(iden_id)
          num_sampled += 1

          if num_sampled == max_identifications_per_observation:
            break

        # Not sure how this is possible
        if old_num_sampled == num_sampled:
          print("ERROR: weird bug in sampling identifications, %d idens sampled" % (num_sampled,))
          print(len([iden['id'] for iden in idens]))
          print(len(set([iden['id'] for iden in idens])))
          print
          break

        if num_sampled == max_identifications_per_observation:
            break

    identification_ids_to_keep = set(identification_ids_to_keep)
    filtered_identifications = [iden for iden in identifications if iden['id'] in identification_ids_to_keep]

    return observations, filtered_identifications

  def create_example_images(self, observations, taxon_id_to_class_label, num_examples=5):
    """ Grab images from observations other than `observations` to use as initial training data
    for the computer vision system.
    """

    exclude_ob_ids = set([ob['id'] for ob in observations])
    taxon_ids = set([ob['community_taxon_id'] for ob in observations])
    taxon_id_to_obs = {taxon_id : [] for taxon_id in taxon_ids}
    for ob in self.observations:
      if ob['id'] not in exclude_ob_ids and ob['community_taxon_id'] in taxon_ids:
        taxon_id_to_obs[ob['community_taxon_id']].append(ob)

    example_images = []
    for taxon_id, available_obs in taxon_id_to_obs.iteritems():

      ob_id_to_image_urls = self.get_image_urls_for_observations(available_obs)
      ob_id_to_ob = {ob['id'] : ob for ob in available_obs}
      obs_with_images = [ob_id_to_ob[ob_id] for ob_id, urls in ob_id_to_image_urls.items() if len(urls) > 0]

      if len(obs_with_images) < num_examples:
        print("WARNING: only %d observations available (with images) as examples for taxon %s, not %d as req." % (len(available_obs), taxon_id, num_examples))
      num_to_sample = min(num_examples, len(obs_with_images))
      if num_to_sample > 0:
        example_obs = random.sample(obs_with_images, num_to_sample)

        for ob in example_obs:
          example_images.append({
            'id' : ob['id'],
            'created_at' : ob['created_at'],
            'url' : ob_id_to_image_urls[ob['id']][0],
            'urls' : ob_id_to_image_urls[ob['id']],
            'gt_label' : taxon_id_to_class_label[taxon_id]
          })

    return example_images


  def analyze_observations(self, observations, identifications):
    """ Print out some useful stats for these observations.
    """

    print("Observation Stats")
    print("%d total observations" % (len(observations),))
    print("%d total identifications" % (len(identifications),))
    print()

    # what is the identification diversity for these observations?
    ob_id_to_taxon_iden = {ob['id'] : set() for ob in observations}
    for iden in identifications:
      ob_id_to_taxon_iden[iden['observation_id']].add(iden['taxon_id'])

    iden_diversity = [len(x[1]) for x in ob_id_to_taxon_iden.items()]

    print("Observation Identification Diversities")
    print("Max diversity: %d" % (np.max(iden_diversity),))
    print("Min diversity: %d" % (np.min(iden_diversity),))
    print("Mean diversity: %d" % (np.mean(iden_diversity),))
    print("Median diversity: %d" % (np.median(iden_diversity),))
    print("(Number of Unique Species Identifications, Number of obervations that have that many unique species identifications)")
    print(sorted(Counter(iden_diversity).items(), key=lambda x: x[0]))
    print()

    # how many identifications does each observation have?
    ob_id_to_iden_count = {ob['id'] : 0 for ob in observations}
    for iden in identifications:
      ob_id_to_iden_count[iden['observation_id']] += 1
    iden_counts = ob_id_to_iden_count.values()
    print("Observation Identification Counts")
    print("Max identifications: %d" % (np.max(iden_counts),))
    print("Min identifications: %d" % (np.min(iden_counts),))
    print("Mean identifications: %d" % (np.mean(iden_counts),))
    print("Median identifications: %d" % (np.median(iden_counts),))
    print("(Number of Identifications, Number of observations that have that many identifications)")
    print(sorted(Counter(iden_counts).items(), key=lambda x: x[0]))
    print()

    # how many identifications per worker?
    worker_ids = set([iden['user_id'] for iden in identifications])
    worker_id_to_iden_count = {wid : 0 for wid in worker_ids}
    for iden in identifications:
      worker_id_to_iden_count[iden['user_id']] += 1
    iden_counts = worker_id_to_iden_count.values()
    print("Worker Identification Counts")
    print("Number of workers: %d" % (len(worker_id_to_iden_count),))
    print("Max identifications: %d" % (np.max(iden_counts),))
    print("Min identifications: %d" % (np.min(iden_counts),))
    print("Mean identifications: %d" % (np.mean(iden_counts),))
    print("Median identifications: %d" % (np.median(iden_counts),))
    print("(Number of identifications, Number of people who have made that many identifications)")
    print(sorted(Counter(iden_counts).items(), key=lambda x: x[0]))
    print()

    # how many taxa has each worker contributed to?
    ob_id_to_community_id = {ob['id'] : ob['community_taxon_id'] for ob in observations}
    worker_id_to_taxa_count = {wid : set() for wid in worker_ids}
    for iden in identifications:
      worker_id_to_taxa_count[iden['user_id']].add(ob_id_to_community_id[iden['observation_id']])
    taxa_counts = [len(x[1]) for x in worker_id_to_taxa_count.items()]
    print("Worker Taxa Counts")
    print("Max taxa: %d" % (np.max(taxa_counts),))
    print("Min taxa: %d" % (np.min(taxa_counts),))
    print("Mean taxa: %d" % (np.mean(taxa_counts),))
    print("Median taxa: %d" % (np.median(taxa_counts),))
    print("(Number of Taxa, Number of people who have observed that many taxa)")
    print(sorted(Counter(taxa_counts).items(), key=lambda x: x[0]))
    print()

  def construct_dataset(self,
    observations,
    identifications,
    allow_non_species_identifications=False,
    convert_leaf_node_keys_to_integers=False,
    flat_taxonomy=False,
    add_ground_truth_labels=False,
    add_empirical_class_priors=False):

    # Lets make sure the observations are unique
    ob_ids = set([ob['id'] for ob in observations])
    assert len(ob_ids) == len(observations), "Error: observation ids are not unique"

    species_taxon_ids = set([obs['community_taxon_id'] for obs in observations])

    ob_id_to_identifications = {ob['id'] : [] for ob in observations}
    for ident in identifications:
      ob_id_to_identifications[ident['observation_id']].append(ident)

    # Lets make sure that each observation has at least 2 identifications
    for ob_id, idents in ob_id_to_identifications.iteritems():
      assert len(idents) >= 2, "Error: observation has less than 2 identifications"

    # Lets also make sure that each identification is from the set of observed classes
    if allow_non_species_identifications:
      taxon_working_set = self.get_taxon_working_set(species_taxon_ids)
      for ident in identifications:
        assert ident['taxon_id'] in taxon_working_set, "Error: identification is 'out of world'"
    else:
      for ident in identifications:
        assert ident['taxon_id'] in species_taxon_ids, "Error: identification is 'out of world'"

    # Lets make sure that the workers for each observation are unique
    for ob_id, idents in ob_id_to_identifications.iteritems():
      user_ids = [ident['user_id'] for ident in idents]
      assert len(user_ids) == len(set(user_ids)), "Error: duplicate worker for an observation."

    if convert_leaf_node_keys_to_integers:
      taxon_id_to_class_label = {taxon_id : label for label, taxon_id in enumerate(species_taxon_ids)}
    else:
      taxon_id_to_class_label = {taxon_id : taxon_id for taxon_id in species_taxon_ids}

    taxonomy_data, old_node_key_to_new_node_key = self.build_taxonomy(taxon_id_to_class_label, flat=flat_taxonomy)
    #print("Taxonomy")
    #for node in taxonomy_data:
    #  print(node)
    #print()
    #print()

    dataset = {
      'num_classes' : len(species_taxon_ids),
      'inat_taxon_id_to_class_label' : taxon_id_to_class_label,
      'taxonomy_data' : taxonomy_data,
      'old_node_key_to_new_node_key' : old_node_key_to_new_node_key
    }

    if add_empirical_class_priors:
      num_classes = dataset['num_classes']
      class_counts = np.zeros(num_classes, dtype=np.float)
      for ob in observations:
        label = taxon_id_to_class_label[ob['community_taxon_id']]
        class_counts[label] += 1
      class_probs = class_counts / np.sum(class_counts)
      dataset['global_class_probs'] = class_probs.tolist()


    gt_labels = []
    if add_ground_truth_labels:
      for ob in observations:
        gt_labels.append({
          'image_id' : ob['id'],
          'label' : {
            'gtype' : 'multiclass',
            'label' : taxon_id_to_class_label[ob['community_taxon_id']]
          }
        })

    workers = {}
    for ident in identifications:
      if ident['user_id'] not in workers:
        workers[ident['user_id']] = {
          'id' : ident['user_id']
        }

    ob_id_to_image_urls = self.get_image_urls_for_observations(observations)

    images = {}
    for ob in observations:
      images[ob['id']] = {
        'id' : ob['id'],
        'created_at' : ob['created_at'],
        'url' : ob_id_to_image_urls[ob['id']][0],
        'urls' : ob_id_to_image_urls[ob['id']]
      }

    annos = []
    for ident in identifications:

      taxon_id = ident['taxon_id']
      if taxon_id in taxon_id_to_class_label:
        worker_label = taxon_id_to_class_label[taxon_id]
      elif taxon_id in old_node_key_to_new_node_key:
        worker_label = old_node_key_to_new_node_key[taxon_id]
      else:
        worker_label = taxon_id

      annos.append({
        'anno' : {
          'gtype' : 'multiclass',
          'label' : worker_label
        },
        'image_id' : ident['observation_id'],
        'worker_id' : ident['user_id'],
        'created_at' : ident['created_at'],
        'id' : ident['id']
      })

    dataset = {
      'dataset' : dataset,
      'gt_labels' : gt_labels,
      'workers' : workers,
      'images' : images,
      'annos' : annos
    }

    return dataset

  def analyze_species_performance(self, observations, identifications):

    taxon_ids = set([obs['community_taxon_id'] for obs in observations])
    taxon_id_to_rank = {taxon['id'] : taxon['rank'] for taxon in self.taxa}

    taxa_performance = {taxon_id : {
                          'num_correct' : 0,
                          'num_incorrect' : 0,
                          'users' : set(),
                          'num_obs' : 0,
                          'num_ids_not_at_species' : 0
                        } for taxon_id in taxon_ids}

    global_num_correct = 0
    global_num_incorrect = 0

    global_num_not_at_species = 0

    ob_id_to_ob = {ob['id'] : ob for ob in observations}
    ob_id_to_identifications = {ob['id'] : [] for ob in observations}
    for ident in identifications:
      ob_id_to_identifications[ident['observation_id']].append(ident)

    for ob_id, idens in ob_id_to_identifications.iteritems():
      gt_taxon_id = ob_id_to_ob[ob_id]['community_taxon_id']
      taxa_performance[gt_taxon_id]['num_obs'] += 1
      for iden in idens:

        taxa_performance[gt_taxon_id]['users'].add(iden['user_id'])
        rank = taxon_id_to_rank[iden['taxon_id']]
        if rank == 'species':

          worker_taxon_id = iden['taxon_id']

          if gt_taxon_id == worker_taxon_id:
            global_num_correct += 1
            taxa_performance[gt_taxon_id]['num_correct'] += 1
          else:
            global_num_incorrect += 1
            taxa_performance[gt_taxon_id]['num_incorrect'] += 1
        else:
          global_num_not_at_species += 1
          taxa_performance[gt_taxon_id]['num_ids_not_at_species'] += 1

    print("Basic Species Stats:")
    print("%d species" % (len(taxon_ids),))
    print("%d total identifications not at species" % (global_num_not_at_species,))
    print("%d total identifications at species" % (global_num_correct + global_num_incorrect,))
    print("%d correct identifications" % (global_num_correct,))
    print("%d incorrect identifications" % (global_num_incorrect,))
    print("Global accuracy: %0.3f" % (global_num_correct / float(global_num_correct + global_num_incorrect)))
    print()

    # Compute per species worker acc
    for taxon_id, stats in taxa_performance.iteritems():
      total = float(stats['num_correct'] + stats['num_incorrect'])
      acc = stats['num_correct'] / total
      stats['acc'] = acc

    taxa_stats = taxa_performance.items()
    taxa_stats.sort(key=lambda x: x[1]['acc'])
    print("{0:4s}\t{1:8s}\t{2:4s}\t{3:5s}\t{4:4s}\t{5:s}".format("", "ID", "ACC", "Users", "Obs", "Idens Not At Species"))
    for i, (taxon_id, stats) in enumerate(taxa_stats):
      acc = stats['acc']
      num_users = len(stats['users'])
      num_obs = stats['num_obs']
      num_idens_not_at_species = stats['num_ids_not_at_species']
      print("{0:4d}\t{1:8s}\t{2:.4f}\t{3:5d}\t{4:4d}\t{5:4d}".format(i+1, taxon_id, acc, num_users, num_obs, num_idens_not_at_species))
    print()

    return taxa_performance

  def compute_flat_gt_probs(self, dataset):
    """ Compute gt probabilities assuming a flat list of species
    """

    taxonomy = Taxonomy()
    taxonomy.load(dataset['dataset']['taxonomy_data'])
    taxonomy.finalize()

    ordered_leaf_node_ids = [node.key for node in taxonomy.leaf_nodes()]

    # For now, we need to assume that the leaf keys are integers
    for key in ordered_leaf_node_ids:
      assert type(key) == int, "We currently need leaf keys to be integers."

    leaf_node_ids = set(ordered_leaf_node_ids)
    num_classes = len(ordered_leaf_node_ids)

    leaf_node_id_to_order = {node_id : node_id for i, node_id in enumerate(ordered_leaf_node_ids)}
    #leaf_node_id_to_order = {node_id : i for i, node_id in enumerate(ordered_leaf_node_ids)}

    # Leaf node class probs
    # NOTE: now this is computed elsewhere
    # class_counts = np.zeros(num_classes, dtype=np.float)
    # for gt_label in dataset['gt_labels']:
    #   label = leaf_node_id_to_order[gt_label['label']['label']]
    #   class_counts[label] += 1
    # class_probs = class_counts / np.sum(class_counts)
    # dataset['dataset']['global_class_probs'] = class_probs.tolist()

    #print("Class Probs")
    #print(class_probs)

    # Single Binomial Model
    global_num_correct = 0.
    global_num_incorrect = 0.

    # Binomial Per Class Model
    per_class_correct = np.zeros(num_classes, dtype=np.float)
    per_class_incorrect = np.zeros(num_classes, dtype=np.float)

    # Multinomial Model
    per_class_confusion = np.zeros((num_classes, num_classes), dtype=np.float)

    image_id_to_gt_label = {x['image_id'] : leaf_node_id_to_order[x['label']['label']] for x in dataset['gt_labels']}
    for anno in dataset['annos']:
      gt_label = image_id_to_gt_label[anno['image_id']]
      worker_label = anno['anno']['label']

      # The worker label could be somewhere higher up in the taxonomy
      if worker_label in leaf_node_ids:
        worker_label = leaf_node_id_to_order[worker_label]

        per_class_confusion[gt_label][worker_label] += 1
        if gt_label == worker_label:
          global_num_correct += 1
          per_class_correct[gt_label] += 1
        else:
          global_num_incorrect += 1
          per_class_incorrect[gt_label] += 1

    prob_correct = global_num_correct / (global_num_correct + global_num_incorrect)
    #print("Prob correct %0.3f" % (prob_correct,))

    dataset['dataset']['global_prob_correct'] = prob_correct

    #print()
    per_class_prob_correct = per_class_correct / (per_class_correct + per_class_incorrect)
    #print("Per Class Prob Correct")
    #for i, p in enumerate(per_class_prob_correct):
    #  print("%2d: %0.3f" % (i, p))

    dataset['dataset']['global_per_class_prob_correct'] = per_class_prob_correct.tolist()

    #print()
    per_class_confusion_prob = per_class_confusion / per_class_confusion.sum(axis=1)[:,np.newaxis]
    np.set_printoptions(threshold='nan', precision=2, linewidth=200)
    #print("Per Class Confusion Prob")
    #print(per_class_confusion_prob)

    dataset['dataset']['global_per_class_confusion_prob'] = per_class_confusion_prob.tolist()


  def compute_taxonomic_gt_probs(self, dataset):

    taxonomy = Taxonomy()
    taxonomy.load(dataset['dataset']['taxonomy_data'])
    taxonomy.finalize()

    # for each node we want to compute the following:
    # prob correct
    # per child node prob correct
    # per child node confusion with the other children
    # per child distribution

    # Put some count variables on all of the nodes
    for node in taxonomy.breadth_first_traversal():
      node.data['num_correct'] = 0
      node.data['num_incorrect'] = 0
      if not node.is_leaf:
        num_children = len(node.children)
        node.data['per_child_correct'] = np.zeros(num_children, dtype=np.float)
        node.data['per_child_incorrect'] = np.zeros(num_children, dtype=np.float)
        node.data['per_child_confusion'] = np.zeros((num_children, num_children), dtype=np.float)
        node.data['per_child_incorrect_preference'] = np.zeros(num_children, dtype=np.float)
        node.data['per_child_counts'] = np.zeros(num_children, dtype=np.float)

    # Add the groundtruth counts for species occurences:
    for gt_label in dataset['gt_labels']:
      gt_key = gt_label['label']['label']
      gt_node = taxonomy.nodes[gt_key]

      for gt_ancestor in gt_node.ancestors:

        gt_ancestor_level = gt_ancestor.level
        gt_ancestor_node_at_level_plus_one = taxonomy.node_at_level_from_node(gt_ancestor_level + 1, gt_node)
        gt_ancestor.data['per_child_counts'][gt_ancestor_node_at_level_plus_one.order] += 1

    # Go through each annotation and update the counts on the node that corresponds to the
    # annotation and the ancestor nodes of that node
    image_id_to_gt_label = {x['image_id'] : x['label']['label'] for x in dataset['gt_labels']}
    for anno in dataset['annos']:

      gt_label = image_id_to_gt_label[anno['image_id']]
      gt_node = taxonomy.nodes[gt_label]
      gt_level = gt_node.level
      assert gt_node.is_leaf

      worker_label = anno['anno']['label']
      worker_node = taxonomy.nodes[worker_label]
      worker_level = worker_node.level

      # Update the leaf node single binomial model
      if gt_node == worker_node:
        gt_node.data['num_correct'] += 1
      else:
        gt_node.data['num_incorrect'] += 1

      # Update the gt ancestors `per_child_correct`, `per_child_incorrect` and `per_child_confusion`
      for gt_ancestor in gt_node.ancestors:

        gt_ancestor_level = gt_ancestor.level
        gt_ancestor_node_at_level_plus_one = taxonomy.node_at_level_from_node(gt_ancestor_level + 1, gt_node)

        # Is the worker label still "higher up" than the ancestor label?
        if worker_level < gt_ancestor_level:
          gt_ancestor.data['num_incorrect'] += 1
          gt_ancestor.data['per_child_incorrect'][gt_ancestor_node_at_level_plus_one.order] += 1
        else:
          worker_ancestor = taxonomy.node_at_level_from_node(gt_ancestor_level, worker_node)


          if gt_ancestor == worker_ancestor:

            gt_ancestor.data['num_correct'] += 1

            # If the worker label is at the root, then skip it
            if gt_ancestor_level + 1 > worker_level:
              continue

            worker_ancestor_node_at_level_plus_one = taxonomy.node_at_level_from_node(gt_ancestor_level + 1, worker_node)
            gt_ancestor.data['per_child_confusion'][gt_ancestor_node_at_level_plus_one.order][worker_ancestor_node_at_level_plus_one.order] += 1

            if gt_ancestor_node_at_level_plus_one == worker_ancestor_node_at_level_plus_one:
              gt_ancestor.data['per_child_correct'][gt_ancestor_node_at_level_plus_one.order] += 1
            else:
              gt_ancestor.data['per_child_incorrect'][gt_ancestor_node_at_level_plus_one.order] += 1

          else:
            gt_ancestor.data['num_incorrect'] += 1
            gt_ancestor.data['per_child_incorrect'][gt_ancestor_node_at_level_plus_one.order] += 1


      # Update the worker ancestors `per_child_incorrect_preference`
      for worker_ancestor in worker_node.ancestors:

        worker_ancestor_level = worker_ancestor.level
        worker_ancestor_node_at_level_plus_one = taxonomy.node_at_level_from_node(worker_ancestor_level + 1, worker_node)

        if gt_level < worker_ancestor_level:
           worker_ancestor.data['per_child_incorrect_preference'][worker_ancestor_node_at_level_plus_one.order] += 1

        else:
          gt_ancestor = taxonomy.node_at_level_from_node(worker_ancestor_level, gt_node)

          if gt_ancestor != worker_ancestor:

            worker_ancestor.data['per_child_incorrect_preference'][worker_ancestor_node_at_level_plus_one.order] += 1

    # Compute the probabilities
    for node in taxonomy.breadth_first_traversal():
      num_correct = node.data['num_correct']
      num_incorrect = node.data['num_incorrect']
      prob_correct = num_correct / float(num_correct + num_incorrect)
      node.data['global_prob_correct'] = prob_correct

      if not node.is_leaf:
        num_children = len(node.children)
        per_child_correct = node.data['per_child_correct']
        per_child_incorrect = node.data['per_child_incorrect']
        per_child_confusion = node.data['per_child_confusion']
        per_child_incorrect_preference = node.data['per_child_incorrect_preference']
        per_child_counts = node.data['per_child_counts']

        per_child_prob_correct = per_child_correct / (per_child_correct + per_child_incorrect)
        node.data['global_per_child_prob_correct'] = per_child_prob_correct.tolist()

        #print(per_child_prob_correct)

        per_child_confusion_prob = per_child_confusion / per_child_confusion.sum(axis=1)[:,np.newaxis]
        node.data['global_per_child_confusion_prob'] = per_child_confusion_prob.tolist()

        #print(per_child_confusion_prob)

        if np.sum(per_child_incorrect_preference) == 0:
          per_child_incorrect_preference = np.ones(num_children, dtype=np.float) * (1. / num_children)
        else:
          per_child_incorrect_preference = per_child_incorrect_preference /np.sum(per_child_incorrect_preference)
        node.data['global_per_child_incorrect_preference_prob'] = per_child_incorrect_preference.tolist()

        #print(per_child_incorrect_preference)

        per_child_probs = per_child_counts / np.sum(per_child_counts)
        node.data['global_per_child_probs'] = per_child_probs.tolist()

        #print(per_child_probs)

        del node.data['per_child_correct']
        del node.data['per_child_incorrect']
        del node.data['per_child_confusion']
        del node.data['per_child_incorrect_preference']
        del node.data['per_child_counts']

      del node.data['num_correct']
      del node.data['num_incorrect']

    taxonomy_data = taxonomy.export(export_data=True)
    dataset['dataset']['taxonomy_data'] = taxonomy_data

  def add_gt_probs_to_dataset(self, dataset):

    self.compute_flat_gt_probs(dataset)

    self.compute_taxonomic_gt_probs(dataset)


  def compress_nodes(self, node_dict):
    """ Remove linked lists and return a map from old nodes to new nodes.
    """

    for node_id, node in node_dict.iteritems():
      depth = 0
      parent_id = node['parent']
      while parent_id != None:
        depth += 1
        parent_id = node_dict[parent_id]['parent']
      node['depth'] = depth

    # Get all of the nodes that have exactly one child
    compression_candidates = [node for node in node_dict.values() if len(node['children']) == 1]
    compression_candidates.sort(key=lambda x: x['depth'])
    compression_candidates.reverse()

    old_node_key_to_new_node_key = {}

    for node in compression_candidates:
      node_id = node['key']
      parent_id = node['parent']
      child_id = node['children'][0]

      # Shift everything up the taxonomy
      del node_dict[node_id]
      old_node_key_to_new_node_key[node_id] = child_id

      # update the map (incase we are deleting something that was already mapped)
      items = old_node_key_to_new_node_key.items()
      for old_node_key, new_node_key in items:
        if new_node_key == node_id:
          old_node_key_to_new_node_key[old_node_key] = child_id

      if parent_id == None:
        # the root node has a single child. Make the child the root
        node_dict[child_id]['parent'] = None

      else:
        node_index_in_parent = node_dict[parent_id]['children'].index(node_id)
        node_dict[parent_id]['children'][node_index_in_parent] = child_id
        node_dict[child_id]['parent'] = parent_id

    return old_node_key_to_new_node_key

  def build_taxonomy(self, taxon_id_to_class_label, flat=False):

    taxon_id_to_taxon = {taxon['id'] : taxon for taxon in self.taxa}
    taxon_id_class_label = taxon_id_to_class_label.items()
    taxon_id_class_label.sort(key=lambda x: x[1])

    taxonomy_data = []
    old_node_key_to_new_node_key = {}

    if flat:
      root_key = 'root'
      root = {
        'key' : root_key,
        'data' : {},
        'parent' : None
      }
      taxonomy_data.append(root)

      for taxon_id, class_label in taxon_id_class_label:
        taxonomy_data.append({
          'key' : class_label,
          'data' : {'taxon_id' : taxon_id},
          'parent' : root_key
        })

    else:

      leaf_taxon_ids = taxon_id_to_class_label.keys()
      nodes = {}

      for taxon_id, class_label in taxon_id_class_label:

        nodes[class_label] = {
          'key' : class_label,
          'data' : {'taxon_id' : taxon_id},
          'parent' : None,
          'children' : []
        }

        taxon = taxon_id_to_taxon[taxon_id]
        ancestry = taxon['ancestry']
        ancestry_ids = ancestry.split('/')
        ancestry_ids.reverse()

        child_node = nodes[class_label]
        for ancestor_id in ancestry_ids:
          if ancestor_id not in nodes:
            nodes[ancestor_id] = {
              'key' : ancestor_id,
              'data' : {'taxon_id' : ancestor_id},
              'parent' : None,
              'children' : []
            }
          ancestor_node = nodes[ancestor_id]
          if child_node['key'] not in ancestor_node['children']:
            ancestor_node['children'].append(child_node['key'])
            child_node['parent'] = ancestor_node['key']

            child_node = ancestor_node
          else:
            break

      # Ensure that ony leaf nodes do not have children
      for node_id, node in nodes.iteritems():
        if node_id in taxon_id_to_class_label.values():
          assert len(node['children']) == 0
        else:
          assert len(node['children']) > 0

      # We can go back through and "compress" the taxonomy to remove linked lists.
      old_node_key_to_new_node_key = self.compress_nodes(nodes)

      taxonomy_data = []
      roots = [node for node in nodes.values() if node['parent'] is None]
      assert len(roots) == 1
      root = roots[0]
      queue = [roots[0]]
      while len(queue):
        node = queue.pop(0)
        taxonomy_data.append({
          'key' : node['key'],
          'parent' : node['parent'],
          'data' : node['data']
        })
        queue += [nodes[node_id] for node_id in node['children']]

    return taxonomy_data, old_node_key_to_new_node_key

  def build_lean_crowdsourcing_dataset(self,
    allow_non_species_identifications=False,
    min_species_identifications_per_observation=None,
    max_identifications_per_observation=None,
    min_observations_per_species=None,
    max_observations_per_species=None,
    target_species_ids=None,
    max_species=None,
    convert_leaf_node_keys_to_integers=False,
    select_observations_by_identification_diversity=True,
    flat_taxonomy=False,
    num_examples=5,
    use_current_identifications=False,
    add_empirical_probs_to_dataset=False,
    add_ground_truth_labels=False,
    add_empirical_class_priors=False
    ):

    taxon_id_to_taxon = {taxon['id'] : taxon for taxon in self.taxa}
    taxon_id_to_rank = {taxon['id'] : taxon['rank'] for taxon in self.taxa}

    # We want to remap subspecies identifications to species identifications
    subspecies_id_to_species = {}
    for taxon in self.taxa:
      if taxon['rank'] == 'subspecies':
        ancestry = taxon['ancestry']
        try:
          ancestry_ids = ancestry.split('/')
        except:
          print("WARNING: bad subspecies? No ancestors found.")
          print(taxon)
          print()
          continue
        species_id = ancestry_ids[-1]
        if species_id not in taxon_id_to_rank:
          continue
        if taxon_id_to_rank[species_id] != 'species':
          print("WARNING: bad subspecies? The parent rank is not `species`, it is %s" % (taxon_id_to_rank[species_id],))
          print(taxon)
          continue
        subspecies_id_to_species[taxon['id']] = species_id


    if target_species_ids != None:
      species_filter = set(target_species_ids)
      for species_id in target_species_ids:
        assert taxon_id_to_rank[species_id] == 'species'
        assert taxon_id_to_taxon[species_id]['is_active'] =='t'
    else:
      # everything
      species_filter = set([taxon_id for taxon_id in taxon_id_to_taxon if taxon_id_to_rank[taxon_id] == 'species' and taxon_id_to_taxon[taxon_id]['is_active'] =='t'])

    # Get the observations with community ids at the species level
    obs_with_cid_at_species = []
    for obs in self.observations:
      try:
        int(obs['community_taxon_id']) # ensure the field is there
        if obs['community_taxon_id'] in taxon_id_to_rank:
          if obs['community_taxon_id'] in species_filter:
            obs_with_cid_at_species.append(obs)
      except:
        pass

    # Filter out any observations that do not have images
    obs_with_cid_at_species, _ = self.enforce_observations_have_images(obs_with_cid_at_species, identifications=None)

    species_taxon_ids = set([obs['community_taxon_id'] for obs in obs_with_cid_at_species])
    print("Found %d species with observations whose community label is at the species level" % (len(species_taxon_ids),))

    # Restrict the identifications to those whose label is in this working set.
    taxon_id_working_set = set(species_taxon_ids)
    if allow_non_species_identifications:
      for taxon_id in species_taxon_ids:
        taxon = taxon_id_to_taxon[taxon_id]
        ancestry = taxon['ancestry']
        ancestry_ids = ancestry.split('/')
        for ancestor_id in ancestry_ids:
          taxon_id_working_set.add(ancestor_id)
    print("Found %d taxa in the working set." % (len(taxon_id_working_set),))

    # Add the identifications to the observations.
    # Make sure all identifications are in the working set of labels
    observation_ids = set([ob['id'] for ob in obs_with_cid_at_species])
    ob_id_to_ob = {ob['id'] : ob for ob in obs_with_cid_at_species}
    ob_id_to_idens = {ob['id'] : [] for ob in obs_with_cid_at_species}
    for identification in self.identifications:
      obs_id = identification['observation_id']
      # Is this an identification of an observation we care about
      if obs_id in observation_ids:

        taxon_id = identification['taxon_id']
        rank = taxon_id_to_rank[taxon_id]

        # remap subspecies to species
        if rank == 'subspecies' and taxon_id in subspecies_id_to_species:
          species_id = subspecies_id_to_species[taxon_id]
          if species_id in taxon_id_working_set:
            s_identification = copy.copy(identification)
            s_identification['taxon_id'] =species_id
            ob_id_to_idens[obs_id].append(s_identification)
        else:
          if taxon_id in taxon_id_working_set:
            ob_id_to_idens[obs_id].append(identification)

    # Analyze the identifications for each observation and try to come up with a good sequence of events
    filtered_observations = []
    filtered_identifications = []
    for ob_id, idens in ob_id_to_idens.iteritems():

      # group the identifications by user and process them individually
      user_id_to_idens = {}
      for identification in idens:
        user_id = identification['user_id']
        user_id_to_idens.setdefault(user_id, [])
        user_id_to_idens[user_id].append(identification)


      # Keep one identification per user
      selected_idens = []

      # Choose each user's current identification
      if use_current_identifications:
        for user_id in user_id_to_idens:
          user_idens = user_id_to_idens[user_id]
          current_iden = None
          # Sort the identifications by time
          for identification in user_idens:
            if identification['current'] == 't':
              current_iden = identification
              break
          if current_iden != None:
            selected_idens.append(current_iden)

      # Choose the first identification from a user
      else:

        for user_id in user_id_to_idens:
          user_idens = user_id_to_idens[user_id]

          # Sort the identifications by time
          for identification in user_idens:
            created_at = identification['created_at']
            try:
              cat = datetime.datetime.strptime(created_at, '%Y-%m-%d %H:%M:%S.%f')
            except:
              cat = datetime.datetime.strptime(created_at, '%Y-%m-%d %H:%M:%S')
            identification['time'] = cat
          user_idens.sort(key=lambda x: x['time'])

          # We want the first identification that user thought it was
          # We just need to make sure to handle taxonomy changes
          selected_iden = user_idens[0]
          user_changed_mind = selected_iden['taxon_change_id'] != None
          for identification in user_idens[1:]:
            is_current = identification['current'] == 't'
            category = identification['category']
            taxon_change_id = identification['taxon_change_id']

            # The taxonomy changed, lets use this identification
            if taxon_change_id != None:
              if user_changed_mind == False:
                selected_iden = identification

            # The user is changing their mind
            else:
              # We want to keep their original identification
              user_changed_mind=True

          assert selected_iden['taxon_id'] in taxon_id_working_set
          selected_idens.append(selected_iden)

      # Make sure this observation has enough identifications at the species level
      species_identification_count = 0
      for identification in selected_idens:
        taxon_id = identification['taxon_id']
        if taxon_id_to_rank[taxon_id] == 'species':
          species_identification_count += 1

      if min_species_identifications_per_observation != None:
        if species_identification_count < min_species_identifications_per_observation:
          continue

      filtered_observations.append(ob_id_to_ob[ob_id])
      filtered_identifications.extend(selected_idens)

    # After filtering the observations by number of identifications, it could be the case
    # that a species doesn't have enough observations. Remove those species.
    filtered_observations, filtered_identifications, did_filter = self.enforce_min_observation_count_per_species(
      filtered_observations,
      filtered_identifications,
      min_observations_per_species,
      allow_non_species_identifications)

    # Some species were removed. Iterate between checking identification counts per observation, and observation counts per species.
    if did_filter:
      while True:
        filtered_observations, filtered_identifications, did_filter = self.enforce_min_species_identification_count_per_observation(
          filtered_observations,
          filtered_identifications,
          min_species_identifications_per_observation)
        if not did_filter:
          break
        filtered_observations, filtered_identifications, did_filter = self.enforce_min_observation_count_per_species(
          filtered_observations,
          filtered_identifications,
          min_observations_per_species,
          allow_non_species_identifications)
        if not did_filter:
          break

    filtered_observations, filtered_identifications = self.enforce_max_observation_count_per_species(
      filtered_observations,
      filtered_identifications,
      max_observations_per_species,
      sort_by_diversity=select_observations_by_identification_diversity)

    species_taxon_ids = set([obs['community_taxon_id'] for obs in filtered_observations])
    print("Found %d species after filtering for identification counts and observation counts." % (len(species_taxon_ids),))
    taxon_id_working_set = set(species_taxon_ids)
    if allow_non_species_identifications:
      for taxon_id in species_taxon_ids:
        taxon = taxon_id_to_taxon[taxon_id]
        ancestry = taxon['ancestry']
        ancestry_ids = ancestry.split('/')
        for ancestor_id in ancestry_ids:
          taxon_id_working_set.add(ancestor_id)
    print("Found %d taxa in the working set." % (len(taxon_id_working_set),))

    print()
    print("Species stats prior to selecting subset.")
    species_stats = self.analyze_species_performance(filtered_observations, filtered_identifications)

    # Select a subset of species based on their worker accuracy.
    filtered_observations, filtered_identifications, did_filter = self.enforce_max_species(
      filtered_observations,
      filtered_identifications,
      species_stats,
      max_species,
      allow_non_species_identifications
    )

    # Some species were removed. Iterate between checking identification counts per observation, and observation counts per species.
    if did_filter:
      while True:
        filtered_observations, filtered_identifications, did_filter = self.enforce_min_species_identification_count_per_observation(
          filtered_observations,
          filtered_identifications,
          min_species_identifications_per_observation)
        if not did_filter:
          break
        filtered_observations, filtered_identifications, did_filter = self.enforce_min_observation_count_per_species(
          filtered_observations,
          filtered_identifications,
          min_observations_per_species,
          allow_non_species_identifications)
        if not did_filter:
          break

    print()
    print("Species stats after selecting subset.")
    species_stats = self.analyze_species_performance(filtered_observations, filtered_identifications)

    # Lets look at the diversity of the selected observations
    print("Observation Analysis before limiting the number of identifications")
    self.analyze_observations(filtered_observations, filtered_identifications)

    filtered_observations, filtered_identifications = self.enforce_max_identifications_per_observation(filtered_observations, filtered_identifications, max_identifications_per_observation=max_identifications_per_observation)

    print("Observation Analysis after limiting the number of identifications")
    self.analyze_observations(filtered_observations, filtered_identifications)

    print("Selected species ids")
    species_ids = list(set([obs['community_taxon_id'] for obs in filtered_observations]))
    print("%d species" % (len(species_ids),))
    print(species_ids)
    print()

    # We now have our final version of the observations and identifications.
    # Lets build the lean crowdsourcing dataset
    dataset = self.construct_dataset(
      filtered_observations,
      filtered_identifications,
      allow_non_species_identifications,
      convert_leaf_node_keys_to_integers,
      flat_taxonomy,
      add_ground_truth_labels,
      add_empirical_class_priors
    )

    # Compute ground truth probabilities for the dataset
    if add_empirical_probs_to_dataset:
      self.add_gt_probs_to_dataset(dataset)

    # make sure that the probs are the same in the case of a flat taxonomy
    if flat_taxonomy and add_empirical_probs_to_dataset:
      assert np.array_equal(dataset['dataset']['global_per_class_confusion_prob'], dataset['dataset']['taxonomy_data'][0]['data']['global_per_child_confusion_prob'])
      assert np.array_equal(dataset['dataset']['global_per_class_prob_correct'], dataset['dataset']['taxonomy_data'][0]['data']['global_per_child_prob_correct'])
      assert np.array_equal(dataset['dataset']['global_class_probs'], dataset['dataset']['taxonomy_data'][0]['data']['global_per_child_probs'])

    # Clear off any properties that are not json serializable
    for identification in filtered_identifications:
      if 'time' in identification:
        del identification['time']

    if num_examples > 0:
      example_images = self.create_example_images(filtered_observations, dataset['dataset']['inat_taxon_id_to_class_label'], num_examples=num_examples)
    else:
      example_images = []

    return dataset, filtered_observations, filtered_identifications, example_images


def parse_args():

    parser = argparse.ArgumentParser(description='Test the person classifier')

    parser.add_argument('--archive_dir', dest='archive_dir',
                        help='Path to the database archive directory.', type=str,
                        required=True)

    parser.add_argument('--output_dir', dest='output_dir',
                          help='Path to an output directory to save the datasets.', type=str,
                          required=True)

    parser.add_argument('--allow_non_species_identifications', dest='allow_non_species_identifications',
                        help='If True, then we will allow inner node identifications, otherwise they will be filtered out.',
                        required=False, action='store_true', default=False)

    parser.add_argument('--target_species_ids', dest='target_species_ids',
                        help='A comma separated list of species ids to use for the dataset. All other species in the archive will be ignored.', type=int,
                        nargs='+', required=False, default=None)

    parser.add_argument('--min_species_identifications_per_observation', dest='min_species_identifications_per_observation',
                        help='The minimum number of identifications needed for an observation to be included in the dataset.',
                        required=False, type=int, default=2)

    parser.add_argument('--max_identifications_per_observation', dest='max_identifications_per_observation',
                        help='The maximum number of identifications to include for an observation.',
                        required=False, type=int, default=None)

    parser.add_argument('--min_observations_per_species', dest='min_observations_per_species',
                        help='The minimum number of observations for a species to have to be included in the dataset.',
                        required=False, type=int, default=1)

    parser.add_argument('--max_observations_per_species', dest='max_observations_per_species',
                        help='The maximum number of observations for a species.',
                        required=False, type=int, default=None)

    parser.add_argument('--select_observations_by_identification_diversity', dest='select_observations_by_identification_diversity',
                        help='If true, then observations will be chosen by prefering those with more label diversity. Otherwise they will be chosen based on the number of identifications.',
                        required=False, action='store_true', default=False)

    parser.add_argument('--max_species', dest='max_species',
                        help='The maximum number of species to include in the dataset.',
                        required=False, type=int, default=None)

    parser.add_argument('--convert_leaf_node_keys_to_integers', dest='convert_leaf_node_keys_to_integers',
                        help='Convert the leaf node keys to be integers in the range [0, # leaf nodes)',
                        required=False, action='store_true', default=False)

    parser.add_argument('--flat_taxonomy', dest='flat_taxonomy',
                        help='Store the taxonomy as a flat list.',
                        required=False, action='store_true', default=False)

    parser.add_argument('--num_examples', dest='num_examples',
                        help='The number of example images to save for each species. These can be used to initialize a vision system.',
                        required=False, type=int, default=0)

    parser.add_argument('--add_empirical_probs_to_dataset', dest='add_empirical_probs_to_dataset',
                        help='Add empirical ground truth probabilities to the dataset.',
                        required=False, action='store_true', default=False)

    parser.add_argument('--add_ground_truth_labels', dest='add_ground_truth_labels',
                        help='Add ground truth labels (i.e. the community ids) to the dataset.',
                        required=False, action='store_true', default=False)

    parser.add_argument('--add_empirical_class_priors', dest='add_empirical_class_priors',
                        help='Add empirical class priors (computed from the community ids) to the dataset.',
                        required=False, action='store_true', default=False)

    args = parser.parse_args()
    return args



def main():

  args = parse_args()

  archive_dir = args.archive_dir
  output_dir = args.output_dir

  with open(os.path.join(archive_dir, 'observations.json')) as f:
    observations = json.load(f)

  with open(os.path.join(archive_dir, 'observation_photos.json')) as f:
    observation_photos = json.load(f)

  with open(os.path.join(archive_dir, 'identifications.json')) as f:
    identifications = json.load(f)

  with open(os.path.join(archive_dir, 'taxa.json')) as f:
    taxa = json.load(f)

  with open(os.path.join(archive_dir, 'users.json')) as f:
    users = json.load(f)

  inat = iNaturalistDataset(observations, observation_photos, identifications, taxa, users)

  inat.simple_stats()

  print("##############################")
  print("BUILDING TRAINING DATASET")
  print()

  # Build the train dataset
  train_dataset, train_observations, train_identifications, train_example_images = inat.build_lean_crowdsourcing_dataset(
    allow_non_species_identifications=args.allow_non_species_identifications,
    min_species_identifications_per_observation=args.min_species_identifications_per_observation,
    max_identifications_per_observation=args.max_identifications_per_observation,
    min_observations_per_species=args.min_observations_per_species,
    max_observations_per_species=args.max_observations_per_species,

    target_species_ids=args.target_species_ids,
    max_species=args.max_species,
    convert_leaf_node_keys_to_integers=args.convert_leaf_node_keys_to_integers,
    select_observations_by_identification_diversity=args.select_observations_by_identification_diversity,
    flat_taxonomy=args.flat_taxonomy,
    num_examples=args.num_examples,
    use_current_identifications=False,
    add_empirical_probs_to_dataset=args.add_empirical_probs_to_dataset,
    add_ground_truth_labels=args.add_ground_truth_labels,
    add_empirical_class_priors=args.add_empirical_class_priors
  )

  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  train_output_dir = os.path.join(output_dir, 'train')
  if not os.path.exists(train_output_dir):
    os.makedirs(train_output_dir)

  with open(os.path.join(train_output_dir, 'dataset.json'), 'w') as f:
    json.dump(train_dataset, f)
  with open(os.path.join(train_output_dir, 'observations.json'), 'w') as f:
    json.dump(train_observations, f)
  with open(os.path.join(train_output_dir, 'identifications.json'), 'w') as f:
    json.dump(train_identifications, f)
  with open(os.path.join(train_output_dir, 'example_images.json'), 'w') as f:
    json.dump(train_example_images, f)

  print()
  print("END BUILDING TRAINING DATASET")
  print("##############################")
  print()
  print("##############################")
  print("BUILDING TESTING DATASET")
  print()

  # Build the test dataset
  species_ids = list(set([ob['community_taxon_id'] for ob in train_observations]))
  test_dataset, test_observations, test_identifications, test_example_images = inat.build_lean_crowdsourcing_dataset(
    allow_non_species_identifications=args.allow_non_species_identifications,
    min_species_identifications_per_observation=args.min_species_identifications_per_observation,
    target_species_ids=species_ids,
    convert_leaf_node_keys_to_integers=args.convert_leaf_node_keys_to_integers,
    flat_taxonomy=args.flat_taxonomy,
    num_examples=0,
    use_current_identifications=True,
    add_empirical_probs_to_dataset=args.add_empirical_probs_to_dataset,
    add_ground_truth_labels=args.add_ground_truth_labels,
    add_empirical_class_priors=args.add_empirical_class_priors
  )

  test_output_dir = os.path.join(output_dir, 'test')
  if not os.path.exists(test_output_dir):
    os.makedirs(test_output_dir)

  with open(os.path.join(test_output_dir, 'dataset.json'), 'w') as f:
    json.dump(test_dataset, f)
  with open(os.path.join(test_output_dir, 'observations.json'), 'w') as f:
    json.dump(test_observations, f)
  with open(os.path.join(test_output_dir, 'identifications.json'), 'w') as f:
    json.dump(test_identifications, f)
  with open(os.path.join(test_output_dir, 'example_images.json'), 'w') as f:
    json.dump(test_example_images, f)

  print()
  print("END BUILDING TESTING DATASET")
  print("##############################")

if __name__ == '__main__':

  random.seed(1)
  np.random.seed(1)

  main()
