# contains helper functions for extraction of synonym pairs
import json
from tqdm import tqdm


def create_pairs_for_medmentions(mentions, target_ontology, train=True, outfile=''):

    pairs = []
    for mention, concept in tqdm(mentions.items()):
        exemplars = target_ontology[concept]
        if train:
            exemplars = [x for x in exemplars if x != mention]
        else:
            assert exemplars

        for exemplar in exemplars:
            pair = (concept, mention, exemplar)
            pairs.append(pair)

    if outfile:
        print('Saving...')
        with open(outfile, 'w') as f:
            json.dump(pairs, f)

    return pairs


def create_pairs_for_medmentions_zeroshot(mentions, target_ontology, outfile=''):

    pairs = []
    for mention, concept in tqdm(mentions.items()):
        exemplars = target_ontology[concept]
        assert exemplars

        pair = (concept, mention, sorted(exemplars))
        pairs.append(pair)

    if outfile:
        print('Saving...')
        with open(outfile, 'w') as f:
            json.dump(pairs, f)

    return pairs
