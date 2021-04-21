# contains helper functions for extraction of synonym pairs
import json
from tqdm import tqdm


def create_pairs_for_train(target_ontology, outfile=''):

    pairs = []
    for concept, exemplars in tqdm(target_ontology.items()):
        if len(exemplars) > 1:
            for exemplar in exemplars:
                other_exemplars = [x for x in exemplars if x != exemplar]
                for other_exemplar in other_exemplars:
                    pair = (concept, exemplar, other_exemplar)
                    pairs.append(pair)

    if outfile:
        print('Saving...')
        with open(outfile, 'w') as f:
            json.dump(pairs, f)

    return pairs


def create_pairs_for_test(references, target_ontology, snomed=False, outfile=''):

    pairs = []
    for reference, concept in tqdm(references.items()):
        try:
            other_exemplars = target_ontology[concept]
        except KeyError:
            if snomed:
                continue
            else:
                raise KeyError
        if not snomed:
            assert other_exemplars
        assert reference not in other_exemplars
        for other_exemplar in other_exemplars:
            pair = (concept, reference, other_exemplar)
            pairs.append(pair)

    if outfile:
        print('Saving...')
        with open(outfile, 'w') as f:
            json.dump(pairs, f)

    return pairs


def create_pairs_for_zeroshot(target_ontology, outfile=''):

    pairs = []
    for concept, exemplars in tqdm(target_ontology.items()):
        if len(exemplars) > 1:
            for exemplar in exemplars:
                other_exemplars = [x for x in exemplars if x != exemplar]
                if other_exemplars:
                    pair = (concept, exemplar, other_exemplars)
                    pairs.append(pair)

    if outfile:
        print('Saving...')
        with open(outfile, 'w') as f:
            json.dump(pairs, f)

    return pairs


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
